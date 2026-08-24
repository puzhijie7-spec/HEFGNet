"""Multi-Source Contextual Edge Enhancement (MSCEE)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def gaussian_kernel(
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    kernel = torch.tensor(
        [
            [1.0, 4.0, 6.0, 4.0, 1.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [6.0, 24.0, 36.0, 24.0, 6.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [1.0, 4.0, 6.0, 4.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    return (kernel / 256.0).repeat(channels, 1, 1, 1)


def gaussian_convolution(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    image = F.pad(image, (2, 2, 2, 2), mode="reflect")
    return F.conv2d(image, kernel, groups=image.shape[1])


def pyramid_downsample(x: torch.Tensor) -> torch.Tensor:
    return x[:, :, ::2, ::2]


def pyramid_upsample(x: torch.Tensor, channels: int) -> torch.Tensor:
    zeros = torch.zeros_like(x)
    x = torch.cat((x, zeros), dim=3)
    x = x.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3] // 2)
    x = x.permute(0, 1, 3, 2)
    zeros = torch.zeros_like(x)
    x = torch.cat((x, zeros), dim=3)
    x = x.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3] // 2)
    x = x.permute(0, 1, 3, 2)
    kernel = 4.0 * gaussian_kernel(channels, x.device, x.dtype)
    return gaussian_convolution(x, kernel)


def make_laplacian_pyramid(
    image: torch.Tensor,
    levels: int,
) -> list[torch.Tensor]:
    channels = image.shape[1]
    current = image
    pyramid = []
    for _ in range(levels):
        kernel = gaussian_kernel(channels, current.device, current.dtype)
        filtered = gaussian_convolution(current, kernel)
        downsampled = pyramid_downsample(filtered)
        upsampled = pyramid_upsample(downsampled, channels)
        if upsampled.shape[-2:] != current.shape[-2:]:
            upsampled = F.interpolate(upsampled, size=current.shape[-2:])
        pyramid.append(current - upsampled)
        current = downsampled
    pyramid.append(current)
    return pyramid


class ChannelGate(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_size = x.shape[-2:]
        average = self.mlp(F.avg_pool2d(x, spatial_size, stride=spatial_size))
        maximum = self.mlp(F.max_pool2d(x, spatial_size, stride=spatial_size))
        scale = torch.sigmoid(average + maximum).unsqueeze(2).unsqueeze(3)
        return x * scale


class SpatialGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compressed = torch.cat(
            (x.max(dim=1, keepdim=True)[0], x.mean(dim=1, keepdim=True)), dim=1
        )
        return x * torch.sigmoid(self.spatial(compressed))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.ChannelGate = ChannelGate(channels, reduction_ratio)
        self.SpatialGate = SpatialGate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.SpatialGate(self.ChannelGate(x))


class MSCEE(nn.Module):
    """Fuse reverse-semantic, semantic-boundary, and Laplacian cues."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(3 * channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.semantic_boundary_extractor = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Tanh(),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.cbam = CBAM(channels)

    def forward(
        self,
        laplacian_feature: torch.Tensor,
        semantic_feature: torch.Tensor,
        guidance_feature: torch.Tensor,
    ) -> torch.Tensor:
        residual = semantic_feature
        reverse_semantic = semantic_feature * (1.0 - torch.sigmoid(guidance_feature))
        semantic_boundary = self.semantic_boundary_extractor(guidance_feature)
        boundary_feature = semantic_feature * semantic_boundary

        laplacian_feature = F.interpolate(
            laplacian_feature,
            size=semantic_feature.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        high_frequency_feature = semantic_feature * laplacian_feature
        fused = torch.cat(
            (reverse_semantic, boundary_feature, high_frequency_feature), dim=1
        )
        fused = self.fusion_conv(fused)
        fused = fused * self.attention(fused)
        return self.cbam(fused + residual)
