"""ConvNeXt V2 backbone adapted from the official Meta implementation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import DropPath, trunc_normal_


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ) -> None:
        super().__init__()
        if data_format not in {"channels_last", "channels_first"}:
            raise ValueError(f"Unsupported data format: {data_format}")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class GRN(nn.Module):
    """Global response normalization."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        response = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        normalized = response / (response.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * normalized) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class ConvNeXtV2(nn.Module):
    """ConvNeXt V2 feature extractor.

    The classification norm and head are retained to remain compatible with the
    ImageNet-22K pretrained checkpoint used in the experiments. The forward
    method returns the four hierarchical feature maps.
    """

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: tuple[int, ...] = (3, 3, 9, 3),
        dims: tuple[int, ...] = (96, 192, 384, 768),
        drop_path_rate: float = 0.1,
        head_init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            )
        )
        for index in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[index], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[index], dims[index + 1], kernel_size=2, stride=2),
                )
            )

        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        offset = 0
        for index, depth in enumerate(depths):
            self.stages.append(
                nn.Sequential(
                    *[
                        ConvNeXtV2Block(dims[index], rates[offset + block_index])
                        for block_index in range(depth)
                    ]
                )
            )
            offset += depth

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for index in range(4):
            x = self.downsample_layers[index](x)
            x = self.stages[index](x)
            if index == 3:
                x = self.norm(x.permute(0, 2, 3, 1))
            outputs.append(x)
        return outputs


def convnextv2_base() -> ConvNeXtV2:
    return ConvNeXtV2(
        depths=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
    )
