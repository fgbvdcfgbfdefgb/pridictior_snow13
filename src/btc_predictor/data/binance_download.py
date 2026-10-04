from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import logging
import os
import tempfile
import zipfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from .schema import KLINE_COLUMNS, MODEL_COLUMNS, SCHEMA_VERSION

LOG = logging.getLogger("btc_download")
BASE = "https://data.binance.vision/data/spot"
NUMERIC_FLOATS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


@dataclass(frozen=True)
class Archive:
    period: str
    url: str
    output: Path


@dataclass(frozen=True)
class ManifestEntry:
    schema_version: str
    source_url: str
    source_sha256: str
    output: str
    output_sha256: str
    rows: int
    min_timestamp_ms: int
    max_timestamp_ms: int
    downloaded_at_utc: str


def _month_starts(start: date, end: date) -> Iterator[date]:
    cursor = start.replace(day=1)
    while cursor <= end:
        yield cursor
        cursor = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)


def plan_archives(start: date, end: date, root: Path, symbol: str) -> list[Archive]:
    """Use monthly archives for closed months and daily files for the current month."""
    if end < start:
        raise ValueError("end must be on or after start")
    result: list[Archive] = []
    for month in _month_starts(start, end):
        last_day = date(month.year, month.month, calendar.monthrange(month.year, month.month)[1])
        if start <= month and last_day <= end:
            period = month.strftime("%Y-%m")
            filename = f"{symbol}-1s-{period}.zip"
            url = f"{BASE}/monthly/klines/{symbol}/1s/{filename}"
            output = (
                root / f"year={month.year:04d}" / f"month={month.month:02d}" / "part-month.parquet"
            )
            result.append(Archive(period, url, output))
        else:
            day = max(start, month)
            while day <= min(end, last_day):
                period = day.isoformat()
                filename = f"{symbol}-1s-{period}.zip"
                url = f"{BASE}/daily/klines/{symbol}/1s/{filename}"
                output = (
                    root
                    / f"year={day.year:04d}"
                    / f"month={day.month:02d}"
                    / f"day={day.day:02d}.parquet"
                )
                result.append(Archive(period, url, output))
                day += timedelta(days=1)
    return result


def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, session: requests.Session) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=(15, 180)) as response:
        if response.status_code == 404:
            raise FileNotFoundError(url)
        response.raise_for_status()
        with target.open("wb") as f:
            for block in response.iter_content(2**20):
                if block:
                    f.write(block)


def _timestamp_ms(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="raise").astype("int64")
    # Binance spot archives use microseconds from 2025-01-01 and milliseconds before it.
    return values.where(values < 10**15, values // 1000)


def convert_zip_to_parquet(
    archive: Path, output: Path, chunk_rows: int = 500_000
) -> tuple[int, int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(".parquet.tmp")
    writer: pq.ParquetWriter | None = None
    total = 0
    min_ts: int | None = None
    max_ts: int | None = None
    try:
        with zipfile.ZipFile(archive) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if len(names) != 1:
                raise ValueError(f"Expected one CSV in {archive}, found {names}")
            with zf.open(names[0]) as raw:
                chunks = pd.read_csv(
                    raw,
                    names=KLINE_COLUMNS,
                    header=None,
                    chunksize=chunk_rows,
                    dtype=str,
                )
                previous_ts: int | None = None
                for frame in chunks:
                    frame["timestamp_ms"] = _timestamp_ms(frame["open_time"])
                    for col in NUMERIC_FLOATS:
                        frame[col] = pd.to_numeric(frame[col], errors="raise").astype("float64")
                    frame["trade_count"] = pd.to_numeric(
                        frame["trade_count"], errors="raise"
                    ).astype("int64")
                    frame = frame[list(MODEL_COLUMNS)]
                    ts = frame["timestamp_ms"].to_numpy()
                    if len(ts) and (ts[1:] <= ts[:-1]).any():
                        raise ValueError(f"Non-increasing timestamps inside {archive}")
                    if previous_ts is not None and len(ts) and int(ts[0]) <= previous_ts:
                        raise ValueError(f"Non-increasing timestamps across chunks in {archive}")
                    if len(ts):
                        previous_ts = int(ts[-1])
                        min_ts = int(ts[0]) if min_ts is None else min_ts
                        max_ts = int(ts[-1])
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            tmp_output,
                            table.schema,
                            compression="zstd",
                            compression_level=9,
                            use_dictionary=["trade_count"],
                            write_statistics=True,
                        )
                    writer.write_table(table, row_group_size=250_000)
                    total += len(frame)
        if writer is None or min_ts is None or max_ts is None:
            raise ValueError(f"No rows in {archive}")
        writer.close()
        writer = None
        os.replace(tmp_output, output)
        return total, min_ts, max_ts
    finally:
        if writer is not None:
            writer.close()
        tmp_output.unlink(missing_ok=True)


def _append_manifest(path: Path, entry: ManifestEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry), sort_keys=True) + "\n")


def ingest_one(
    item: Archive, manifest: Path, session: requests.Session, force: bool = False
) -> None:
    if item.output.exists() and not force:
        LOG.info("skip existing %s", item.output)
        return
    with tempfile.TemporaryDirectory(prefix="btc-binance-") as temp:
        archive = Path(temp) / Path(item.url).name
        LOG.info("download %s", item.url)
        download(item.url, archive, session)
        source_hash = sha256_file(archive)
        rows, min_ts, max_ts = convert_zip_to_parquet(archive, item.output)
    entry = ManifestEntry(
        schema_version=SCHEMA_VERSION,
        source_url=item.url,
        source_sha256=source_hash,
        output=item.output.as_posix(),
        output_sha256=sha256_file(item.output),
        rows=rows,
        min_timestamp_ms=min_ts,
        max_timestamp_ms=max_ts,
        downloaded_at_utc=datetime.now(UTC).isoformat(),
    )
    _append_manifest(manifest, entry)
    LOG.info("wrote %s (%s rows)", item.output, f"{rows:,}")


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    yesterday = datetime.now(UTC).date() - timedelta(days=1)
    parser = argparse.ArgumentParser(description="Download Binance BTCUSDT 1-second klines")
    parser.add_argument("--start", type=parse_date, default=date(2020, 1, 1))
    parser.add_argument(
        "--end",
        type=parse_date,
        default=yesterday,
        help="inclusive UTC day; default is yesterday (last finalizable day)",
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/binance_1s.jsonl"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    items = plan_archives(args.start, args.end, args.output, args.symbol)
    with requests.Session() as session:
        session.headers["User-Agent"] = "pridictior_snow13/0.1"
        for item in items:
            try:
                ingest_one(item, args.manifest, session, args.force)
            except FileNotFoundError:
                LOG.warning("archive not published yet: %s", item.url)


if __name__ == "__main__":
    main()
