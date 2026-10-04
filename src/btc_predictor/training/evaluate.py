from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from btc_predictor.features.market_analyser import analyse_history, normalize_history
from btc_predictor.models import build_model

COLUMNS = [
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


def load_model(checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    model_cfg = dict(cfg["model"])
    model_cfg.update(
        input_dim=9,
        feature_dim=int(cfg["features"]["output_dim"]),
        horizon=int(cfg["data"]["horizon_seconds"]),
        quantiles=len(model_cfg["quantiles"]),
    )
    model = build_model(model_cfg)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), cfg


def partition_date(path: Path) -> date:
    parts = {
        piece.split("=", 1)[0]: piece.split("=", 1)[1].split(".", 1)[0]
        for piece in path.parts
        if "=" in piece
    }
    return date(int(parts["year"]), int(parts["month"]), int(parts.get("day", 1)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Immutable one-second walk-forward evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model, cfg = load_model(args.checkpoint, device)
    history_len, horizon = int(cfg["data"]["history_seconds"]), int(cfg["data"]["horizon_seconds"])
    quantiles = np.asarray(cfg["model"]["quantiles"])
    median_index = int(np.argmin(np.abs(quantiles - 0.5)))

    files = sorted(args.data.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"No Parquet shards under {args.data}")
    train_end = date.fromisoformat(str(cfg["data"]["train_end"]))
    validation_end = date.fromisoformat(str(cfg["data"]["validation_end"]))
    if args.split == "validation":
        holdout = [p for p in files if train_end <= partition_date(p) < validation_end]
    else:
        holdout = [p for p in files if partition_date(p) >= validation_end]
    if not holdout:
        raise SystemExit(f"No {args.split} partitions under {args.data}")
    selected_file = holdout[-1]
    table = pq.read_table(selected_file, columns=["timestamp_ms", *COLUMNS])
    timestamps = table.column("timestamp_ms").to_numpy()
    raw = np.column_stack([table.column(c).to_numpy() for c in COLUMNS]).astype(np.float64)
    available = len(raw) - history_len - horizon
    count = min(args.samples, available)
    if count <= 1:
        raise SystemExit("Holdout shard is too short for history+horizon")
    indices = np.arange(history_len, history_len + count)
    predictions, targets, anchors = [], [], []

    for offset in range(0, count, args.batch_size):
        batch_idx = indices[offset : offset + args.batch_size]
        histories = [raw[i - history_len + 1 : i + 1] for i in batch_idx]
        features = np.stack([analyse_history(x) for x in histories])
        normalized = np.stack([normalize_history(x) for x in histories])
        target = np.stack(
            [
                np.log(np.maximum(raw[i + 1 : i + 1 + horizon, 3], 1e-12) / max(raw[i, 3], 1e-12))
                for i in batch_idx
            ]
        )
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ),
        ):
            pred = (
                model(
                    torch.from_numpy(normalized).to(device),
                    torch.from_numpy(features).to(device),
                )
                .float()
                .cpu()
                .numpy()
            )
        predictions.append(pred)
        targets.append(target)
        anchors.append(np.log(np.maximum(raw[batch_idx, 3], 1e-12)))

    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    anchor = np.concatenate(anchors)
    median = prediction[..., median_index]
    final_error = median[:, -1] - target[:, -1]
    path_error = median - target
    actual_direction = np.sign(target[:, -1])
    predicted_direction = np.sign(median[:, -1])
    # Shift previous forecast by one horizon step and compare overlapping absolute log-price paths.
    current_abs = anchor[1:, None] + median[1:, :-1]
    previous_abs = anchor[:-1, None] + median[:-1, 1:]
    instability = np.mean(np.abs(current_abs - previous_abs)) * 10_000
    low, high = prediction[..., 0], prediction[..., -1]
    coverage = float(np.mean((target >= low) & (target <= high)))
    intended_coverage = float(quantiles[-1] - quantiles[0])
    metrics = {
        "checkpoint": args.checkpoint.as_posix(),
        "split": args.split,
        "holdout_file": selected_file.as_posix(),
        "first_issue_timestamp_ms": int(timestamps[indices[0]]),
        "last_issue_timestamp_ms": int(timestamps[indices[-1]]),
        "samples": int(count),
        "mae_bps": float(np.mean(np.abs(final_error)) * 10_000),
        "rmse_bps": float(np.sqrt(np.mean(final_error**2)) * 10_000),
        "path_mae_bps": float(np.mean(np.abs(path_error)) * 10_000),
        "direction_error": float(np.mean(predicted_direction != actual_direction)),
        "instability_bps": float(instability),
        "calibration_error": float(abs(coverage - intended_coverage)),
        "interval_coverage": coverage,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
