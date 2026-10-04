# Dataset contract

Canonical source: Binance spot `BTCUSDT`, one-second klines. Coverage begins 2020-01-01 UTC. Finalized archives end at yesterday UTC; the current UTC day is necessarily partial and is supplied by the Bybit live collector, not mislabeled as finalized history.

Columns:

- `timestamp_ms` (UTC bar open)
- `open`, `high`, `low`, `close`
- `volume`, `quote_volume`, `trade_count`
- `taker_buy_base_volume`, `taker_buy_quote_volume`
- `is_imputed` — true only when the source omitted a wall-clock second; OHLC is the previous close and all activity is zero

Partitions use `data/processed/year=YYYY/month=MM/*.parquet`. Each monthly or daily archive becomes an independent Zstandard Parquet shard and a JSONL manifest record containing source URL, source SHA-256, output SHA-256, row count and timestamp range.

Run `scripts/download_all.sh`. The downloader normalizes Binance's pre-2025 millisecond timestamps and post-2025 microsecond timestamps to milliseconds. Parquet and checkpoint paths are tracked by Git LFS through `.gitattributes`.

A small real-source sample is committed under `data/sample/` for tests. It is not a substitute for the LFS dataset.
