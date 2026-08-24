"""Feature fusion and segmentation head."""

from __future__ import annotations

import torch
from torch import nn


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, ratio: int = 16) -> None:
        super().__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        scale = self.avg_pooling(x).view(batch, channels)
        scale = self.fc_layers(scale).view(batch, channels, 1, 1)
        return x * self.sigmoid(scale)


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batch, channels, height, width = x.shape
    x = x.view(batch, groups, channels // groups, height, width)
    x = x.transpose(1, 2).contiguous()
    return x.view(batch, channels, height, width)


class EfficientUpConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pwc = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_dwc(x)
        return self.pwc(channel_shuffle(x, self.in_channels))


class HierarchicalFeatureFusion(nn.Module):
    def __init__(self, channels: int = 512) -> None:
        super().__init__()
        self.conv_act = nn.Sequential(
            nn.Conv2d(3 * channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.ca = SqueezeExcitation(channels)
        self.attforcross = SqueezeExcitation(channels)
        self.attforvssm = SqueezeExcitation(channels)
        self.attbou = SqueezeExcitation(channels)
        self.up = EfficientUpConvolution(channels, channels, scale=2)
        self.up1 = EfficientUpConvolution(channels, channels, scale=2)

    def forward(
        self,
        detail: torch.Tensor,
        semantic: torch.Tensor,
        boundary: torch.Tensor,
    ) -> torch.Tensor:
        semantic = self.up(semantic)
        boundary = self.up1(boundary)
        fused = self.ca(self.conv_act(torch.cat((detail, semantic, boundary), dim=1)))
        return (
            fused
            + self.attforcross(detail)
            + self.attforvssm(semantic)
            + self.attbou(boundary)
        )


class SegmentationHead(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.UP_stage_00 = EfficientUpConvolution(512, 128, scale=2)
        self.UP_stage_0 = EfficientUpConvolution(128, 128, scale=4)
        self.UP_stage_1 = EfficientUpConvolution(128, 128, scale=2)
        self.cls_seg = nn.Conv2d(128, num_classes, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.UP_stage_00(x)
        x = self.UP_stage_0(x)
        x = self.UP_stage_1(x)
        return self.cls_seg(x)
