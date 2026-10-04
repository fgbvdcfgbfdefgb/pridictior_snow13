from __future__ import annotations

import torch
from torch import nn

from .common import SmoothForecastHead, multiscale_sequence


class GRUAttention(nn.Module):
    def __init__(
        self,
        input_dim=9,
        feature_dim=96,
        hidden_size=768,
        layers=6,
        heads=12,
        dropout=0.1,
        horizon=1500,
        quantiles=3,
        output_rank=192,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.query = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_size))
        self.attention = nn.MultiheadAttention(
            hidden_size, heads, dropout=dropout, batch_first=True
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_size * 2), nn.Linear(hidden_size * 2, hidden_size), nn.GELU()
        )
        self.head = SmoothForecastHead(hidden_size, horizon, quantiles, output_rank)

    def forward(self, history: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        encoded, state = self.gru(multiscale_sequence(history))
        query = self.query(features).unsqueeze(1)
        attended, _ = self.attention(query, encoded, encoded, need_weights=False)
        fused = self.fuse(torch.cat((state[-1], attended[:, 0]), dim=-1))
        return self.head(fused)
