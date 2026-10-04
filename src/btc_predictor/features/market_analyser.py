from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

WINDOWS = (5, 15, 30, 60, 300, 900, 3600, 14400, 43200)
WINDOW_FEATURES = (
    "return",
    "realized_vol",
    "range",
    "close_z",
    "volume_z",
    "quote_volume_z",
    "trade_rate",
    "taker_imbalance",
    "trend",
    "max_drawdown",
)
BASE_FEATURES = ("body", "upper_wick", "lower_wick", "log_volume", "log_quote_volume", "log_trades")
FEATURE_NAMES = tuple(f"w{w}_{name}" for w in WINDOWS for name in WINDOW_FEATURES) + BASE_FEATURES
assert len(FEATURE_NAMES) == 96


@dataclass(slots=True)
class AnalyserOutput:
    timestamp_ms: int
    values: np.ndarray
    warm: bool


class MarketAnalyser:
    """A single deterministic CPU analyser shared by every GPU candidate.

    Input order is open, high, low, close, volume, quote_volume, trade_count,
    taker_buy_base_volume, taker_buy_quote_volume. It performs no look-ahead.
    """

    def __init__(self, history_seconds: int = 43_200, clip_zscore: float = 8.0):
        self.history_seconds = history_seconds
        self.clip_zscore = clip_zscore
        self._rows: deque[np.ndarray] = deque(maxlen=history_seconds)
        self._timestamps: deque[int] = deque(maxlen=history_seconds)

    def update(self, timestamp_ms: int, row: np.ndarray) -> AnalyserOutput:
        values = np.asarray(row, dtype=np.float64)
        if values.shape != (9,):
            raise ValueError(f"expected 9 market values, got {values.shape}")
        if self._timestamps and timestamp_ms <= self._timestamps[-1]:
            raise ValueError("timestamps must be strictly increasing")
        self._timestamps.append(int(timestamp_ms))
        self._rows.append(values)
        return AnalyserOutput(timestamp_ms, self.compute(), len(self._rows) == self.history_seconds)

    @staticmethod
    def _safe_z(last: float, values: np.ndarray) -> float:
        std = float(values.std())
        return 0.0 if std < 1e-12 else (last - float(values.mean())) / std

    def compute(self) -> np.ndarray:
        if not self._rows:
            return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        data = np.stack(self._rows)
        output: list[float] = []
        eps = 1e-12
        for window in WINDOWS:
            sample = data[-min(window, len(data)) :]
            close = np.maximum(sample[:, 3], eps)
            returns = np.diff(np.log(close))
            peak = np.maximum.accumulate(close)
            x = np.arange(len(close), dtype=np.float64)
            x -= x.mean()
            denom = float(np.dot(x, x)) or 1.0
            trend = float(np.dot(x, np.log(close) - np.log(close).mean()) / denom)
            imbalance = 2.0 * sample[:, 8].sum() / max(sample[:, 5].sum(), eps) - 1.0
            output.extend(
                (
                    float(np.log(close[-1] / close[0])) if len(close) > 1 else 0.0,
                    float(returns.std() * np.sqrt(max(window, 1))) if len(returns) else 0.0,
                    float(np.log(max(sample[:, 1].max(), eps) / max(sample[:, 2].min(), eps))),
                    self._safe_z(close[-1], close),
                    self._safe_z(sample[-1, 4], sample[:, 4]),
                    self._safe_z(sample[-1, 5], sample[:, 5]),
                    float(sample[:, 6].mean()),
                    float(np.clip(imbalance, -1.0, 1.0)),
                    trend,
                    float(np.min(close / np.maximum(peak, eps) - 1.0)),
                )
            )
        last = data[-1]
        open_, high, low, close, volume, quote_volume, trades = last[:7]
        scale = max(close, eps)
        output.extend(
            (
                (close - open_) / scale,
                (high - max(open_, close)) / scale,
                (min(open_, close) - low) / scale,
                np.log1p(max(volume, 0.0)),
                np.log1p(max(quote_volume, 0.0)),
                np.log1p(max(trades, 0.0)),
            )
        )
        result = np.nan_to_num(np.asarray(output, dtype=np.float32))
        return np.clip(result, -self.clip_zscore, self.clip_zscore)


def analyse_history(rows: np.ndarray, clip_zscore: float = 8.0) -> np.ndarray:
    """Compute the shared analyser vector for an already materialized history."""
    values = np.asarray(rows, dtype=np.float64)
    analyser = MarketAnalyser(len(values), clip_zscore)
    analyser._rows.extend(values)
    analyser._timestamps.extend(range(len(values)))
    return analyser.compute()


def normalize_history(rows: np.ndarray) -> np.ndarray:
    """Convert raw 9-column bars to stationary model channels."""
    x = np.asarray(rows, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 9:
        raise ValueError("history must have shape [time, 9]")
    eps = 1e-12
    anchor = max(float(x[-1, 3]), eps)
    prices = np.log(np.maximum(x[:, :4], eps) / anchor)
    volume = np.log1p(np.maximum(x[:, 4:6], 0.0))
    volume = (volume - volume.mean(axis=0, keepdims=True)) / (
        volume.std(axis=0, keepdims=True) + 1e-6
    )
    trades = np.log1p(np.maximum(x[:, 6:7], 0.0))
    trades = (trades - trades.mean()) / (trades.std() + 1e-6)
    base_imbalance = 2.0 * x[:, 7:8] / np.maximum(x[:, 4:5], eps) - 1.0
    quote_imbalance = 2.0 * x[:, 8:9] / np.maximum(x[:, 5:6], eps) - 1.0
    result = np.concatenate([prices, volume, trades, base_imbalance, quote_imbalance], axis=1)
    return np.clip(np.nan_to_num(result), -10.0, 10.0).astype(np.float32)
