"""HEFGNet architecture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .backbones import convnextv2_base
from .modules import (
    DPSABlock,
    DetailFeaturePath,
    HierarchicalFeatureFusion,
    MSCEE,
    SegmentationHead,
    make_laplacian_pyramid,
)


LEGACY_PREFIX_MAP = {
    "net1.": "backbone.",
    "net2.": "deep_dpsa.",
    "net3.": "mid_dpsa.",
    "norm0.": "stage0_norm.",
    "norm1.": "stage1_norm.",
    "norm2.": "stage2_norm.",
    "ega.": "mscee.",
    "mv2.": "detail_path.downsample.",
    "mblock0.": "detail_path.stage0_enhancer.",
    "mblock1.": "detail_path.stage1_enhancer.",
    "mv21.": "detail_path.refine.",
    "mblock2.": "detail_path.fusion_enhancer.",
    "changec.": "detail_path.projection.",
    "changec1.": "deep_projection.",
    "ffm.": "fusion.",
    "head.": "segmentation_head.",
}


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state-dict-like mapping.")
    for key in ("model", "state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    state_dict = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean_key = key[len("module.") :] if key.startswith("module.") else key
        state_dict[clean_key] = value
    return state_dict


def convert_legacy_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    converted = {}
    for key, value in state_dict.items():
        key = key[len("module.") :] if key.startswith("module.") else key
        new_key = key
        for old_prefix, new_prefix in LEGACY_PREFIX_MAP.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix) :]
                break
        # These gates were registered in the experimental implementation but
        # were never used by its forward method.
        if ".gate_row." in new_key or ".gate_col." in new_key:
            continue
        converted[new_key] = value
    return converted


def rgb_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    if image.shape[1] != 3:
        raise ValueError("HEFGNet expects a three-channel RGB input.")
    weights = image.new_tensor((0.2989, 0.5870, 0.1140)).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


class HEFGNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = convnextv2_base()
        self.input_downsample = nn.AvgPool2d(kernel_size=2, stride=2)

        self.deep_dpsa = DPSABlock(1024, key_dim=128, num_heads=3)
        self.mid_dpsa = DPSABlock(512, key_dim=64, num_heads=3)
        self.stage0_norm = nn.LayerNorm(128)
        self.stage1_norm = nn.LayerNorm(256)
        self.stage2_norm = nn.LayerNorm(512)

        self.deep_projection = nn.Conv2d(1024, 512, kernel_size=1)
        self.mscee = MSCEE(512)
        self.detail_path = DetailFeaturePath()
        self.fusion = HierarchicalFeatureFusion(512)
        self.segmentation_head = SegmentationHead(num_classes)

    @staticmethod
    def _to_nchw(feature: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        feature = feature.permute(0, 2, 3, 1)
        return norm(feature).permute(0, 3, 1, 2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        grayscale = rgb_to_grayscale(image)
        laplacian_feature = make_laplacian_pyramid(grayscale, levels=3)[2]

        features = self.backbone(self.input_downsample(image))
        stage0 = self._to_nchw(features[0], self.stage0_norm)
        stage1 = self._to_nchw(features[1], self.stage1_norm)
        stage2 = self._to_nchw(features[2], self.stage2_norm)
        stage3 = features[3].permute(0, 3, 1, 2)

        deep = self.deep_projection(self.deep_dpsa(stage3))
        deep = F.interpolate(
            deep, size=stage2.shape[-2:], mode="bilinear", align_corners=False
        )
        mid = self.mid_dpsa(stage2)
        semantic = deep + mid
        boundary = self.mscee(laplacian_feature, mid, deep)
        detail = self.detail_path(stage0, stage1)
        output = self.fusion(detail, semantic, boundary)
        return self.segmentation_head(output)

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        strict: bool = True,
    ) -> nn.modules.module._IncompatibleKeys:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = convert_legacy_state_dict(_extract_state_dict(checkpoint))
        return self.load_state_dict(state_dict, strict=strict)

    def load_backbone_weights(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)
        if any(key.startswith("backbone.") for key in state_dict):
            state_dict = {
                key[len("backbone.") :]: value
                for key, value in state_dict.items()
                if key.startswith("backbone.")
            }
        elif any(key.startswith("net1.") for key in state_dict):
            state_dict = {
                key[len("net1.") :]: value
                for key, value in state_dict.items()
                if key.startswith("net1.")
            }
        self.backbone.load_state_dict(state_dict, strict=True)


def build_hefgnet(
    num_classes: int,
    backbone_weights: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> HEFGNet:
    model = HEFGNet(num_classes=num_classes)
    if backbone_weights is not None:
        model.load_backbone_weights(backbone_weights)
    if checkpoint is not None:
        model.load_checkpoint(checkpoint)
    return model
