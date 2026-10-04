from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SmoothForecastHead(nn.Module):
    """Predict sparse knots, then interpolate to a smooth 1,500-point path."""

    def __init__(self, input_dim: int, horizon: int, quantiles: int, knots: int = 192):
        super().__init__()
        self.horizon = horizon
        self.quantiles = quantiles
        self.knots = min(knots, horizon)
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, self.knots * quantiles),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        knots = self.projection(state).view(state.shape[0], self.quantiles, self.knots)
        path = F.interpolate(knots, size=self.horizon, mode="linear", align_corners=True)
        return path.transpose(1, 2)  # [batch, horizon, quantile]


def multiscale_sequence(history: torch.Tensor) -> torch.Tensor:
    """Preserve the last 15m at 1s, prior 3h at 10s, and 12h at 60s."""
    recent = history[:, -900:]
    medium = history[:, -10_800:-900:10]
    long = history[:, :-10_800:60]
    return torch.cat((long, medium, recent), dim=1)
