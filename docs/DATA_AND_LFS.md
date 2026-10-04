# Data, provenance and Git LFS

## Why monthly Parquet shards

Normal Git is unsuitable for this series. One row per wall-clock second from 2020-01-01 through 2026-10-03 is 213,235,200 potential rows. GitHub rejects ordinary Git objects above 100 MiB; Git LFS raises the per-file ceiling but still meters storage and bandwidth.

The downloader therefore:

1. uses completed monthly Binance archives where possible;
2. uses daily archives for an incomplete calendar month;
3. processes one ZIP at a time so temporary storage remains bounded;
4. normalizes timestamps to milliseconds;
5. inserts an explicit `is_imputed=true` flat-price/zero-activity row for each missing source second;
6. writes one Zstandard Parquet partition per source archive;
7. records source and output SHA-256 hashes, row/imputation counts and timestamp bounds;
8. removes the temporary archive.

The current UTC day cannot be finalized. “Till today” means finalized archives through yesterday plus a separately labeled live stream for today.

## Reproducibility

```bash
START=2020-01-01 END=2026-10-03 bash scripts/download_all.sh
cat data/manifests/binance_1s.jsonl
```

`data.validate` rejects duplicate/backward timestamps and non-finite core values. It reports gaps rather than silently filling them. If a downstream experiment chooses forward-filled no-trade bars, that transformation must create a new schema/version and an explicit `is_imputed` feature.

## LFS workflow

```bash
git lfs install
git add .gitattributes data/processed data/manifests
git commit -m "data: add BTCUSDT one-second partitions"
git lfs ls-files
git push
```

Before pushing, check the GitHub account's LFS billing/quota. Monthly shards are kept below the documented per-file ceiling. Do not squash large LFS versions repeatedly: old LFS objects can continue consuming storage.

## Offline verification

After `git lfs pull` in the staging environment:

```bash
python -m btc_predictor.data.validate /mnt/data/processed
sha256sum -c offline_bundle/SHA256SUMS
```

A Git LFS pointer is not a dataset. Verify that staged `.parquet` files are actual Parquet binaries and not small text pointers before disabling network access.
