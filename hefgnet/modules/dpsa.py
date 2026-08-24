"""Dual-Pooling Squeeze Attention (DPSA)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bn_weight_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.add_module(
            "c",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias=False,
            ),
        )
        batch_norm = nn.BatchNorm2d(out_channels)
        nn.init.constant_(batch_norm.weight, bn_weight_init)
        nn.init.constant_(batch_norm.bias, 0)
        self.add_module("bn", batch_norm)


class HardSigmoid(nn.Module):
    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + 3.0) / 6.0


class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, shape: int = 16) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, dim, shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        embedding = F.interpolate(
            self.pos_embed, size=length, mode="linear", align_corners=False
        )
        return x + embedding


class ConvolutionalMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = ConvBN(in_features, hidden_features)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            bias=True,
        )
        self.act = nn.ReLU()
        self.fc2 = ConvBN(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class DualPoolingSqueezeAttention(nn.Module):
    """Axial attention over average- and max-pooled representations."""

    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int,
        attention_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads
        self.value_dim = int(attention_ratio * key_dim)
        self.dh = self.value_dim * num_heads

        self.to_q = ConvBN(dim, self.nh_kd)
        self.to_k = ConvBN(dim, self.nh_kd)
        self.to_v = ConvBN(dim, self.dh)

        self.proj = nn.Sequential(nn.ReLU(), ConvBN(self.dh, dim, bn_weight_init=0))
        self.proj_encode_row = nn.Sequential(
            nn.ReLU(), ConvBN(self.dh, self.dh, bn_weight_init=0)
        )
        self.proj_encode_column = nn.Sequential(
            nn.ReLU(), ConvBN(self.dh, self.dh, bn_weight_init=0)
        )
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(self.nh_kd)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(self.nh_kd)
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(self.nh_kd)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(self.nh_kd)

        self.dwconv = ConvBN(
            2 * self.dh,
            2 * self.dh,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=2 * self.dh,
        )
        self.act = nn.ReLU()
        self.pwconv = ConvBN(2 * self.dh, dim)
        self.sigmoid = HardSigmoid()

    def _attend_axis(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        axis: str,
        pool: str,
    ) -> torch.Tensor:
        batch, _, height, width = q.shape
        reduce_dim = -1 if axis == "row" else -2
        length = height if axis == "row" else width

        if pool == "avg":
            q_axis = q.mean(reduce_dim)
            k_axis = k.mean(reduce_dim)
            v_axis = v.mean(reduce_dim)
        elif pool == "max":
            q_axis = q.max(reduce_dim)[0]
            k_axis = k.max(reduce_dim)[0]
            v_axis = v.max(reduce_dim)[0]
        else:
            raise ValueError(f"Unsupported pooling mode: {pool}")

        if axis == "row":
            q_axis = self.pos_emb_rowq(q_axis)
            k_axis = self.pos_emb_rowk(k_axis)
            projection = self.proj_encode_row
            output_shape = (batch, self.dh, height, 1)
        else:
            q_axis = self.pos_emb_columnq(q_axis)
            k_axis = self.pos_emb_columnk(k_axis)
            projection = self.proj_encode_column
            output_shape = (batch, self.dh, 1, width)

        q_axis = q_axis.reshape(batch, self.num_heads, -1, length).permute(0, 1, 3, 2)
        k_axis = k_axis.reshape(batch, self.num_heads, -1, length)
        v_axis = v_axis.reshape(batch, self.num_heads, -1, length).permute(0, 1, 3, 2)
        attention = (q_axis @ k_axis * self.scale).softmax(dim=-1)
        output = attention @ v_axis
        output = output.permute(0, 1, 3, 2).reshape(output_shape)
        return projection(output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        local = torch.cat((q, k, v), dim=1)
        local = self.act(self.dwconv(local))
        local = self.pwconv(local)

        row = self._attend_axis(q, k, v, axis="row", pool="avg")
        row = row + self._attend_axis(q, k, v, axis="row", pool="max")
        column = self._attend_axis(q, k, v, axis="column", pool="avg")
        column = column + self._attend_axis(q, k, v, axis="column", pool="max")

        context = self.proj(v + row + column)
        return self.sigmoid(context) * local


class DPSABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        attention_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.attn = DualPoolingSqueezeAttention(
            dim,
            key_dim=key_dim,
            num_heads=num_heads,
            attention_ratio=attention_ratio,
        )
        self.mlp = ConvolutionalMLP(dim, int(dim * mlp_ratio))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.mlp(x)
