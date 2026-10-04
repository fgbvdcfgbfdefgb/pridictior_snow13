#!/usr/bin/env python3
"""Snowflake GPU job entry point. No network access is used after package install."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

# Allow this entry point to run directly from a cloned repository without first
# installing the project as a package. Propagate the same path to DDP children.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from btc_predictor.hardware import detect_hardware, recommended_profile  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--candidate",
        choices=["all", "patch_transformer", "dilated_tcn", "gru_attention"],
        default="all",
    )
    parser.add_argument("--max-updates", type=int)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile; disabled by default to avoid a long silent first-step compilation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers per GPU rank (default: 4, or 16 total on four GPUs)",
    )
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    if not any(args.data_root.rglob("*.parquet")):
        raise SystemExit(f"No offline Parquet shards found under {args.data_root}")
    hw = detect_hardware(args.data_root)
    report = {"detected": asdict(hw), "profile": recommended_profile(hw)}
    print(json.dumps(report, indent=2), flush=True)
    gpus = len(hw.gpus)
    if gpus < 1:
        raise SystemExit("This entry point requires at least one visible CUDA GPU")
    candidates = (
        [args.candidate]
        if args.candidate != "all"
        else ["patch_transformer", "dilated_tcn", "gru_attention"]
    )
    default_config = PROJECT_ROOT / "configs/default.yaml"
    config_text = default_config.read_text()
    config_text = config_text.replace("root: data/processed", f"root: {args.data_root}")
    config_text = config_text.replace("num_workers: auto", f"num_workers: {args.num_workers}")
    if not args.compile:
        config_text = config_text.replace("compile: true", "compile: false")
    temp_config = args.output_root / "snowflake-default.yaml"
    temp_config.parent.mkdir(parents=True, exist_ok=True)
    temp_config.write_text(config_text)

    env = os.environ.copy()
    env["BTC_DATA_ROOT"] = str(args.data_root)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SRC_ROOT), env.get("PYTHONPATH", "")) if part
    )
    for model in candidates:
        command = [
            sys.executable,
            "-u",
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={gpus}",
            "-m",
            "btc_predictor.training.train",
            "--config",
            str(temp_config),
            "--model-config",
            str(PROJECT_ROOT / f"configs/models/{model}.yaml"),
            "--output",
            str(args.output_root / model),
        ]
        if args.max_updates:
            command += ["--max-updates", str(args.max_updates)]
        print(
            f"\nLaunching {model} with {gpus} DDP ranks, "
            f"{args.num_workers} data workers/rank, torch.compile={args.compile}",
            flush=True,
        )
        subprocess.run(command, check=True, env=env, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
