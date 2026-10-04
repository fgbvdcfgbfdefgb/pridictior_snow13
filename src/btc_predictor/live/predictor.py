from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from btc_predictor.features.market_analyser import analyse_history, normalize_history
from btc_predictor.models import build_model


class LivePredictor:
    def __init__(self, checkpoint: str | Path, device: str = "cuda"):
        requested = torch.device(device)
        self.device = (
            requested
            if requested.type != "cuda" or torch.cuda.is_available()
            else torch.device("cpu")
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.config = state["config"]
        model_cfg = dict(self.config["model"])
        model_cfg.update(
            input_dim=9,
            feature_dim=int(self.config["features"]["output_dim"]),
            horizon=int(self.config["data"]["horizon_seconds"]),
            quantiles=len(model_cfg["quantiles"]),
        )
        self.quantiles = tuple(float(q) for q in model_cfg["quantiles"])
        self.model = build_model(model_cfg)
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()
        self.history_size = int(self.config["data"]["history_seconds"])
        self.history: deque[np.ndarray] = deque(maxlen=self.history_size)
        self.last_timestamp_ms: int | None = None

    def bootstrap_from_parquet(self, root: str | Path) -> int:
        files = sorted(Path(root).rglob("*.parquet"), reverse=True)
        remaining = self.history_size
        pieces = []
        columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trade_count",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ]
        for path in files:
            table = pq.read_table(path, columns=columns)
            values = np.column_stack([table.column(c).to_numpy() for c in columns])
            pieces.append(values[-remaining:])
            remaining -= min(remaining, len(values))
            if remaining <= 0:
                break
        if not pieces:
            return 0
        combined = np.concatenate(list(reversed(pieces)), axis=0)[-self.history_size :]
        self.history.extend(combined)
        # Timestamps reset at the first live bar; values only warm the model.
        return len(combined)

    def append(self, timestamp_ms: int, raw_values) -> None:
        values = np.asarray(raw_values, dtype=np.float64)
        if self.last_timestamp_ms is not None and timestamp_ms <= self.last_timestamp_ms:
            return
        self.last_timestamp_ms = timestamp_ms
        self.history.append(values)

    @property
    def warm(self) -> bool:
        return len(self.history) == self.history_size

    @torch.inference_mode()
    def predict(self) -> dict:
        if not self.warm:
            raise RuntimeError(f"history warm-up {len(self.history)}/{self.history_size}")
        raw = np.stack(self.history)
        # This is the single shared CPU analyser definition used during training.
        features = analyse_history(raw)
        history = torch.from_numpy(normalize_history(raw)).unsqueeze(0).to(self.device)
        features_t = torch.from_numpy(features).unsqueeze(0).to(self.device)
        use_amp = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp):
            relative = self.model(history, features_t)[0].float().cpu().numpy()
        relative.sort(axis=-1)  # prevent finite precision quantile crossing at inference
        current = float(raw[-1, 3])
        paths = current * np.exp(relative)
        return {
            "issued_timestamp_ms": self.last_timestamp_ms,
            "current_price": current,
            "quantiles": self.quantiles,
            "paths": paths,
            "warm": True,
        }
