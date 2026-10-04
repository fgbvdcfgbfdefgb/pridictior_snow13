from __future__ import annotations

import torch
from torch.nn import functional as F


def quantile_loss(
    prediction: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor
) -> torch.Tensor:
    error = target.unsqueeze(-1) - prediction
    return torch.maximum((quantiles - 1.0) * error, quantiles * error).mean()


def forecast_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
    weights: dict[str, float],
    previous_prediction: torch.Tensor | None = None,
    current_log_price: torch.Tensor | None = None,
    previous_log_price: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    median_index = int(torch.argmin(torch.abs(quantiles - 0.5)))
    median = prediction[..., median_index]
    point = F.huber_loss(median[:, -1], target[:, -1])
    path = F.huber_loss(median, target)
    qloss = quantile_loss(prediction, target, quantiles)
    first = F.huber_loss(torch.diff(median, dim=1), torch.diff(target, dim=1))
    second = F.huber_loss(torch.diff(median, n=2, dim=1), torch.diff(target, n=2, dim=1))
    direction = F.soft_margin_loss(
        median[:, -1] * torch.sign(target[:, -1]), torch.ones_like(target[:, -1])
    )
    crossing = F.relu(prediction[..., :-1] - prediction[..., 1:]).mean()
    consistency = prediction.new_zeros(())
    if (
        previous_prediction is not None
        and current_log_price is not None
        and previous_log_price is not None
    ):
        current_abs = current_log_price[:, None] + median
        previous_median = previous_prediction[..., median_index]
        previous_abs = previous_log_price[:, None] + previous_median
        consistency = F.huber_loss(current_abs[:, :-1], previous_abs[:, 1:])
    parts = {
        "point_huber": point,
        "path_huber": path,
        "quantile": qloss,
        "first_difference": first,
        "second_difference": second,
        "temporal_consistency": consistency,
        "direction": direction,
        "quantile_crossing": crossing,
    }
    total = sum(weights.get(name, 0.0) * value for name, value in parts.items()) + 0.05 * crossing
    return total, parts
