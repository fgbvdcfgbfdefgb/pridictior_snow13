from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow.dataset as ds


@dataclass(frozen=True, slots=True)
class MarketTick:
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trade_count: int
    taker_buy_base_volume: float
    taker_buy_quote_volume: float


class HistoricalMarketSimulator:
    """Strictly ordered feed. This object exposes no future row or target."""

    def __init__(self, root: str | Path, start_ms: int | None = None, end_ms: int | None = None):
        self.root = Path(root)
        self.start_ms = start_ms
        self.end_ms = end_ms

    def _filter(self):
        filt = None
        field = ds.field("timestamp_ms")
        if self.start_ms is not None:
            filt = field >= self.start_ms
        if self.end_ms is not None:
            upper = field <= self.end_ms
            filt = upper if filt is None else filt & upper
        return filt

    def ticks(self, speed: float = 0.0, batch_size: int = 65_536) -> Iterator[MarketTick]:
        dataset = ds.dataset(self.root, format="parquet", partitioning="hive")
        columns = list(MarketTick.__dataclass_fields__)
        previous = None
        wall_start = time.monotonic()
        market_start = None
        for batch in dataset.scanner(
            columns=columns, filter=self._filter(), batch_size=batch_size
        ).to_batches():
            arrays = {name: batch.column(name).to_pylist() for name in columns}
            for idx in range(batch.num_rows):
                tick = MarketTick(**{name: arrays[name][idx] for name in columns})
                if previous is not None and tick.timestamp_ms <= previous:
                    raise RuntimeError("Dataset is not strictly chronological")
                previous = tick.timestamp_ms
                if speed > 0:
                    market_start = market_start or tick.timestamp_ms
                    due = wall_start + (tick.timestamp_ms - market_start) / 1000.0 / speed
                    delay = due - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                yield tick
