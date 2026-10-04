from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds


def validate_dataset(root: Path) -> dict:
    dataset = ds.dataset(root, format="parquet", partitioning="hive")
    scanner = dataset.scanner(columns=["timestamp_ms", "close", "volume"], batch_size=262_144)
    rows = duplicates = backwards = gaps = nonfinite = 0
    previous = None
    first = last = None
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        ts = frame["timestamp_ms"].to_numpy()
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
        nonfinite += int((~np.isfinite(frame[["close", "volume"]].to_numpy(dtype=float))).sum())
        rows += len(frame)
        previous = int(ts[-1])
    result = {
        "rows": rows,
        "first_timestamp_ms": first,
        "last_timestamp_ms": last,
        "duplicates": duplicates,
        "backwards": backwards,
        "gaps_over_one_second": gaps,
        "nonfinite": nonfinite,
        "valid": rows > 0 and duplicates == 0 and backwards == 0 and nonfinite == 0,
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
