from __future__ import annotations

import torch
from torch import nn

from .common import SmoothForecastHead, multiscale_sequence


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1)
        self.padding = padding
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 1),
        )
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.padding:
            y = y[..., : -self.padding]
        return self.norm(x + y)


class DilatedTCN(nn.Module):
    def __init__(
        self,
        input_dim=9,
        feature_dim=96,
        channels=512,
        blocks=16,
        kernel_size=5,
        dropout=0.1,
        horizon=1500,
        quantiles=3,
        output_rank=192,
    ):
        super().__init__()
        self.input = nn.Conv1d(input_dim, channels, 1)
        self.blocks = nn.Sequential(
            *[
                ResidualTCNBlock(channels, 2 ** (i % 10), kernel_size, dropout)
                for i in range(blocks)
            ]
        )
        self.features = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, channels))
        self.fuse = nn.Sequential(nn.Linear(channels * 2, channels), nn.GELU())
        self.head = SmoothForecastHead(channels, horizon, quantiles, output_rank)

    def forward(self, history: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = multiscale_sequence(history).transpose(1, 2)
        state = self.blocks(self.input(x))[..., -1]
        state = self.fuse(torch.cat((state, self.features(features)), dim=-1))
        return self.head(state)
