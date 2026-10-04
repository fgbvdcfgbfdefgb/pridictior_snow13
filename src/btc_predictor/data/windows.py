from __future__ import annotations

import random
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from btc_predictor.features.market_analyser import analyse_history, normalize_history

try:
    import torch
    from torch.utils.data import IterableDataset, get_worker_info
except ImportError:  # lets data tooling run without the training extra
    torch = None
    IterableDataset = object
    get_worker_info = lambda: None

RAW_COLUMNS = [
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


class ParquetWindowStream(IterableDataset):
    """Endless random windows, sharded by DDP rank and DataLoader worker.

    Targets are future log returns. The simulator/trainer controls when their reward
    becomes eligible; this class never includes a future value in model inputs.
    """

    def __init__(
        self,
        root: str | Path,
        history: int = 43_200,
        horizon: int = 1_500,
        samples_per_file: int = 2_000,
        seed: int = 1337,
        split: str = "train",
        train_end: str = "2025-01-01",
        validation_end: str = "2026-01-01",
    ):
        super().__init__()
        self.root = Path(root)
        self.history = history
        self.horizon = horizon
        self.samples_per_file = samples_per_file
        self.seed = seed
        self.split = split
        self.train_end = date.fromisoformat(train_end)
        self.validation_end = date.fromisoformat(validation_end)

    @staticmethod
    def _partition_date(path: Path) -> date:
        parts = {
            piece.split("=", 1)[0]: piece.split("=", 1)[1] for piece in path.parts if "=" in piece
        }
        return date(int(parts["year"]), int(parts["month"]), int(parts.get("day", 1)))

    def _files(self) -> list[Path]:
        files = sorted(self.root.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files under {self.root}")
        if self.split == "train":
            selected = [p for p in files if self._partition_date(p) < self.train_end]
        elif self.split == "validation":
            selected = [
                p for p in files if self.train_end <= self._partition_date(p) < self.validation_end
            ]
        elif self.split == "test":
            selected = [p for p in files if self._partition_date(p) >= self.validation_end]
        else:
            raise ValueError(f"unknown split {self.split!r}")
        if not selected:
            raise FileNotFoundError(f"No {self.split} partitions under {self.root}")
        return selected

    @staticmethod
    def _rank() -> tuple[int, int]:
        if (
            torch is not None
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def __iter__(self) -> Iterator[dict]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        rank, world = self._rank()
        shard_id, shard_count = rank * workers + worker_id, world * workers
        files = self._files()[shard_id::shard_count]
        if not files:
            files = self._files()[rank::world]
        rng = random.Random(self.seed + shard_id)
        while True:  # update-count based training; deliberately no epoch abstraction
            rng.shuffle(files)
            for path in files:
                table = pq.read_table(path, columns=["timestamp_ms", *RAW_COLUMNS])
                timestamp = table.column("timestamp_ms").to_numpy()
                raw = np.column_stack([table.column(c).to_numpy() for c in RAW_COLUMNS]).astype(
                    np.float64
                )
                first, last = self.history, len(raw) - self.horizon - 1
                if last <= first:
                    continue
                count = min(self.samples_per_file, last - first)
                for idx in rng.sample(range(first, last), count):
                    history = raw[idx - self.history + 1 : idx + 1]
                    previous_history = raw[idx - self.history : idx]
                    future = raw[idx + 1 : idx + 1 + self.horizon, 3]
                    current = max(raw[idx, 3], 1e-12)
                    previous_current = max(raw[idx - 1, 3], 1e-12)
                    yield {
                        "timestamp_ms": np.int64(timestamp[idx]),
                        "history": normalize_history(history),
                        "features": analyse_history(history),
                        "previous_history": normalize_history(previous_history),
                        "previous_features": analyse_history(previous_history),
                        "target": np.log(np.maximum(future, 1e-12) / current).astype(np.float32),
                        "current_log_price": np.float32(np.log(current)),
                        "previous_log_price": np.float32(np.log(previous_current)),
                    }
