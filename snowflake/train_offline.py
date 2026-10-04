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

from btc_predictor.hardware import detect_hardware, recommended_profile


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
    args = parser.parse_args()
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
    for model in candidates:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={gpus}",
            "-m",
            "btc_predictor.training.train",
            "--config",
            "configs/default.yaml",
            "--model-config",
            f"configs/models/{model}.yaml",
            "--output",
            str(args.output_root / model),
        ]
        if args.max_updates:
            command += ["--max-updates", str(args.max_updates)]
        env = os.environ.copy()
        env["BTC_DATA_ROOT"] = str(args.data_root)
        # Override data.root without network by writing a resolved temporary config.
        config_text = Path("configs/default.yaml").read_text()
        config_text = config_text.replace("root: data/processed", f"root: {args.data_root}")
        temp_config = args.output_root / "snowflake-default.yaml"
        temp_config.parent.mkdir(parents=True, exist_ok=True)
        temp_config.write_text(config_text)
        command[command.index("configs/default.yaml")] = str(temp_config)
        subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
