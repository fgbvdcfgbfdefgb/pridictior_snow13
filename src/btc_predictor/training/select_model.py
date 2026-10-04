from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEFAULT_WEIGHTS = {
    "mae_bps": 0.35,
    "rmse_bps": 0.20,
    "path_mae_bps": 0.20,
    "direction_error": 0.10,
    "instability_bps": 0.10,
    "calibration_error": 0.05,
}


def score(metrics: dict, weights: dict[str, float] = DEFAULT_WEIGHTS) -> float:
    missing = set(weights) - set(metrics)
    if missing:
        raise ValueError(f"missing selection metrics: {sorted(missing)}")
    return sum(weights[key] * float(metrics[key]) for key in weights)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best immutable walk-forward result")
    parser.add_argument("runs", type=Path, nargs="?", default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/best.pt"))
    args = parser.parse_args()
    candidates = []
    for result in args.runs.glob("*/evaluation.json"):
        metrics = json.loads(result.read_text())
        candidates.append((score(metrics), result.parent, metrics))
    if not candidates:
        raise SystemExit(f"No */evaluation.json under {args.runs}")
    candidates.sort(key=lambda item: item[0])
    value, run, metrics = candidates[0]
    source = run / "latest.pt"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, args.output)
    summary = {"selected_run": run.as_posix(), "score": value, "metrics": metrics}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
