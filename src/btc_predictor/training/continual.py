from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from btc_predictor.features.market_analyser import analyse_history, normalize_history


@dataclass(slots=True)
class MaturedSample:
    issued_timestamp_ms: int
    history: np.ndarray
    features: np.ndarray
    target: np.ndarray
    current_log_price: float


class ContinualReplayBuffer:
    """Enforce the 25-minute delayed reward during chronological replay.

    At market time t only the sample issued at t-horizon is released for learning.
    The newest 25 minutes remain label-inaccessible.
    """

    def __init__(self, history_seconds: int = 43_200, horizon_seconds: int = 1_500):
        self.history = history_seconds
        self.horizon = horizon_seconds
        self.capacity = history_seconds + horizon_seconds
        self.rows: deque[np.ndarray] = deque(maxlen=self.capacity)
        self.timestamps: deque[int] = deque(maxlen=self.capacity)

    def append(self, timestamp_ms: int, raw_row: np.ndarray) -> MaturedSample | None:
        if self.timestamps and timestamp_ms <= self.timestamps[-1]:
            raise ValueError("chronological replay requires increasing timestamps")
        self.timestamps.append(int(timestamp_ms))
        self.rows.append(np.asarray(raw_row, dtype=np.float64))
        if len(self.rows) < self.capacity:
            return None
        raw = np.stack(self.rows)
        issue_idx = self.history - 1
        context = raw[: self.history]
        current = max(float(raw[issue_idx, 3]), 1e-12)
        future = raw[issue_idx + 1 :, 3]
        return MaturedSample(
            issued_timestamp_ms=self.timestamps[issue_idx],
            history=normalize_history(context),
            features=analyse_history(context),
            target=np.log(np.maximum(future, 1e-12) / current).astype(np.float32),
            current_log_price=float(np.log(current)),
        )
