from __future__ import annotations

import torch
from torch import nn

from .common import SmoothForecastHead


class PatchTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 9,
        feature_dim: int = 96,
        d_model: int = 768,
        layers: int = 12,
        heads: int = 12,
        dropout: float = 0.1,
        horizon: int = 1500,
        quantiles: int = 3,
        output_rank: int = 192,
    ):
        super().__init__()
        # 12h branch: 60-second sampling / 6 samples per patch (6 min/token)
        # 3h branch: 10-second sampling / 10 samples per patch (100 sec/token)
        # 30m branch: native seconds / 15 samples per patch (15 sec/token)
        self.patch_long = nn.Conv1d(input_dim, d_model, kernel_size=6, stride=6)
        self.patch_medium = nn.Conv1d(input_dim, d_model, kernel_size=10, stride=10)
        self.patch_recent = nn.Conv1d(input_dim, d_model, kernel_size=15, stride=15)
        self.feature_token = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, d_model)
        )
        self.branch_embedding = nn.Parameter(torch.randn(3, 1, d_model) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = SmoothForecastHead(d_model, horizon, quantiles, output_rank)

    @staticmethod
    def _patch(layer: nn.Conv1d, values: torch.Tensor) -> torch.Tensor:
        return layer(values.transpose(1, 2)).transpose(1, 2)

    def forward(self, history: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        long = self._patch(self.patch_long, history[:, ::60]) + self.branch_embedding[0]
        medium = self._patch(self.patch_medium, history[:, -10_800::10]) + self.branch_embedding[1]
        recent = self._patch(self.patch_recent, history[:, -1_800:]) + self.branch_embedding[2]
        feature = self.feature_token(features).unsqueeze(1)
        cls = self.cls.expand(history.shape[0], -1, -1)
        tokens = torch.cat((cls, feature, long, medium, recent), dim=1)
        encoded = self.encoder(tokens)
        return self.head(self.norm(encoded[:, 0]))
