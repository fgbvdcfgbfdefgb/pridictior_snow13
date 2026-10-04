from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def validate_dataset(root: Path) -> dict:
    """Validate shards sequentially with bounded memory."""
    files = sorted(root.rglob("*.parquet"))
    if not files:
        return {
            "rows": 0,
            "first_timestamp_ms": None,
            "last_timestamp_ms": None,
            "duplicates": 0,
            "backwards": 0,
            "gaps_over_one_second": 0,
            "imputed_rows": 0,
            "nonfinite": 0,
            "valid": False,
        }
    rows = duplicates = backwards = gaps = nonfinite = imputed = 0
    previous = None
    first = last = None
    for path in files:
        parquet = pq.ParquetFile(path)
        has_imputed = "is_imputed" in parquet.schema_arrow.names
        columns = ["timestamp_ms", "close", "volume"] + (["is_imputed"] if has_imputed else [])
        for batch in parquet.iter_batches(batch_size=65_536, columns=columns, use_threads=False):
            ts = batch.column(0).to_numpy(zero_copy_only=False)
            if len(ts) == 0:
                continue
            first = int(ts[0]) if first is None else first
            last = int(ts[-1])
            if previous is not None:
                delta = int(ts[0]) - previous
                duplicates += int(delta == 0)
                backwards += int(delta < 0)
                gaps += int(delta > 1000)
            delta = ts[1:] - ts[:-1]
            duplicates += int((delta == 0).sum())
            backwards += int((delta < 0).sum())
            gaps += int((delta > 1000).sum())
            close = batch.column(1).to_numpy(zero_copy_only=False)
            volume = batch.column(2).to_numpy(zero_copy_only=False)
            nonfinite += int((~np.isfinite(close)).sum() + (~np.isfinite(volume)).sum())
            if has_imputed:
                imputed += int(batch.column(3).to_numpy(zero_copy_only=False).sum())
            rows += len(ts)
            previous = int(ts[-1])
    result = {
        "rows": rows,
        "first_timestamp_ms": first,
        "last_timestamp_ms": last,
        "duplicates": duplicates,
        "backwards": backwards,
        "gaps_over_one_second": gaps,
        "imputed_rows": imputed,
        "nonfinite": nonfinite,
        "valid": rows > 0 and duplicates == 0 and backwards == 0 and gaps == 0 and nonfinite == 0,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("data/processed"))
    args = parser.parse_args()
    result = validate_dataset(args.root)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
