"""Detail feature path used by HEFGNet."""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn


def conv_1x1_bn(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(),
    )


def conv_nxn_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            kernel_size // 2,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(),
    )


class PreNorm(nn.Module):
    def __init__(self, dim: int, function: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if not (heads == 1 and dim_head == dim)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [
            rearrange(tensor, "b p n (h d) -> b p h n d", h=self.heads)
            for tensor in (q, k, v)
        ]
        attention = self.attend((q @ k.transpose(-1, -2)) * self.scale)
        output = attention @ v
        output = rearrange(output, "b p h n d -> b p n (h d)")
        return self.to_out(output)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attention, feed_forward in self.layers:
            x = attention(x) + x
            x = feed_forward(x) + x
        return x


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError("stride must be 1 or 2")
        hidden_dim = int(in_channels * expansion)
        self.use_residual = stride == 1 and in_channels == out_channels
        layers = []
        if expansion != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.SiLU(),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv(x)
        return x + output if self.use_residual else output


class MobileViTEnhancer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        channels: int,
        kernel_size: int,
        patch_size: tuple[int, int],
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ph, self.pw = patch_size
        self.conv1 = conv_nxn_bn(channels, channels, kernel_size)
        self.conv2 = conv_1x1_bn(channels, dim)
        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)
        self.conv3 = conv_1x1_bn(dim, channels)
        self.conv4 = conv_nxn_bn(2 * channels, channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv2(self.conv1(x))
        _, _, height, width = x.shape
        if height % self.ph or width % self.pw:
            raise ValueError(
                f"Feature size {(height, width)} is not divisible by patch size "
                f"{(self.ph, self.pw)}"
            )
        x = rearrange(
            x,
            "b d (h ph) (w pw) -> b (ph pw) (h w) d",
            ph=self.ph,
            pw=self.pw,
        )
        x = self.transformer(x)
        x = rearrange(
            x,
            "b (ph pw) (h w) d -> b d (h ph) (w pw)",
            h=height // self.ph,
            w=width // self.pw,
            ph=self.ph,
            pw=self.pw,
        )
        x = self.conv3(x)
        return self.conv4(torch.cat((x, residual), dim=1))


class DetailFeaturePath(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.downsample = InvertedResidual(128, 256, stride=2, expansion=2)
        self.stage0_enhancer = MobileViTEnhancer(64, 2, 256, 3, (2, 2), 128)
        self.stage1_enhancer = MobileViTEnhancer(64, 2, 256, 3, (2, 2), 128)
        self.refine = InvertedResidual(256, 256, stride=1, expansion=2)
        self.fusion_enhancer = MobileViTEnhancer(64, 2, 256, 3, (2, 2), 128)
        self.projection = nn.Conv2d(256, 512, kernel_size=1)

    def forward(
        self,
        stage0: torch.Tensor,
        stage1: torch.Tensor,
    ) -> torch.Tensor:
        stage0 = self.stage0_enhancer(self.downsample(stage0))
        stage1 = self.stage1_enhancer(stage1)
        fused = self.refine(stage0 + stage1)
        fused = self.fusion_enhancer(fused)
        return self.projection(fused)
