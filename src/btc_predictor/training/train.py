from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from btc_predictor.config import deep_merge, load_config
from btc_predictor.data.windows import ParquetWindowStream
from btc_predictor.hardware import detect_hardware, recommended_profile
from btc_predictor.models import build_model
from btc_predictor.training.losses import forecast_loss


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world, local_rank, device


def resolve_config(cfg: dict) -> dict:
    profile = recommended_profile(detect_hardware())
    cfg = deep_merge(
        cfg,
        {
            "model": {k: v for k, v in profile["model"].items() if cfg["model"].get(k) == "auto"},
            "training": {
                k: v for k, v in profile["training"].items() if cfg["training"].get(k) == "auto"
            },
        },
    )
    return cfg


def atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def cosine_lr(step: int, warmup: int, maximum: int) -> float:
    if step < warmup:
        return max(1e-3, step / max(1, warmup))
    progress = min(1.0, (step - warmup) / max(1, maximum - warmup))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def seed_everything(seed: int, rank: int) -> None:
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update-count-based DDP trainer (no epoch loop)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()

    rank, world, local_rank, device = setup_distributed()
    cfg = resolve_config(load_config(args.config, args.model_config))
    if args.max_updates is not None:
        cfg["training"]["max_updates"] = args.max_updates
    seed_everything(int(cfg["seed"]), rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model_cfg = dict(cfg["model"])
    model_cfg.update(
        input_dim=9,
        feature_dim=int(cfg["features"]["output_dim"]),
        horizon=int(cfg["data"]["horizon_seconds"]),
        quantiles=len(cfg["model"]["quantiles"]),
    )
    model = build_model(model_cfg).to(device)
    if cfg["training"].get("compile") and hasattr(torch, "compile"):
        model = torch.compile(model)
    wrapped = (
        DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
        if world > 1
        else model
    )

    micro = int(cfg["training"]["micro_batch_size"])
    accumulate = int(cfg["training"]["gradient_accumulation"])
    stream = ParquetWindowStream(
        cfg["data"]["root"],
        history=int(cfg["data"]["history_seconds"]),
        horizon=int(cfg["data"]["horizon_seconds"]),
        seed=int(cfg["seed"]),
        split="train",
        train_end=str(cfg["data"]["train_end"]),
        validation_end=str(cfg["data"]["validation_end"]),
    )
    loader = DataLoader(
        stream,
        batch_size=micro,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg["training"]["num_workers"]) > 0,
        prefetch_factor=2 if int(cfg["training"]["num_workers"]) > 0 else None,
    )
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(
        wrapped.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_lr(
            step, int(cfg["training"]["warmup_steps"]), int(cfg["training"]["max_updates"])
        ),
    )
    precision = cfg["training"]["precision"]
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    quantiles = torch.tensor(cfg["model"]["quantiles"], device=device)
    update = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        base = wrapped.module if isinstance(wrapped, DDP) else wrapped
        if hasattr(base, "_orig_mod"):
            base = base._orig_mod
        base.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        update = int(state["update"])

    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "resolved_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        hardware = {"hardware": asdict(detect_hardware()), "world_size": world}
        (args.output / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")
    metrics_path = args.output / "metrics.jsonl"
    wrapped.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()

    while update < int(cfg["training"]["max_updates"]):
        totals: dict[str, float] = {}
        for micro_step in range(accumulate):
            batch = next(iterator)
            history = batch["history"].to(device, non_blocking=True)
            features = batch["features"].to(device, non_blocking=True)
            previous_history = batch["previous_history"].to(device, non_blocking=True)
            previous_features = batch["previous_features"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            current_log = batch["current_log_price"].to(device, non_blocking=True)
            previous_log = batch["previous_log_price"].to(device, non_blocking=True)
            sync = micro_step == accumulate - 1
            sync_context = (
                contextlib.nullcontext()
                if sync or not isinstance(wrapped, DDP)
                else wrapped.no_sync()
            )
            autocast = torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
            )
            with sync_context, autocast:
                # Previous-tick output supplies an overlapping absolute-price path.
                # It is a stop-gradient consistency target, avoiding a second graph.
                with torch.no_grad():
                    previous_prediction = wrapped(previous_history, previous_features)
                prediction = wrapped(history, features)
                loss, parts = forecast_loss(
                    prediction,
                    target,
                    quantiles,
                    cfg["loss"],
                    previous_prediction=previous_prediction,
                    current_log_price=current_log,
                    previous_log_price=previous_log,
                )
                loss = loss / accumulate
            scaler.scale(loss).backward()
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) / accumulate
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            wrapped.parameters(), float(cfg["training"]["gradient_clip"])
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        update += 1

        if rank == 0 and (update % 50 == 0 or update == 1):
            record = {
                "update": update,
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "lr": scheduler.get_last_lr()[0],
                **totals,
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        checkpoint_every = int(cfg["training"]["checkpoint_every"])
        if rank == 0 and (
            update % checkpoint_every == 0 or update == int(cfg["training"]["max_updates"])
        ):
            base = wrapped.module if isinstance(wrapped, DDP) else wrapped
            if hasattr(base, "_orig_mod"):
                base = base._orig_mod
            payload = {
                "model": base.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update": update,
                "config": cfg,
            }
            atomic_checkpoint(args.output / f"checkpoint-{update:08d}.pt", payload)
            atomic_checkpoint(args.output / "latest.pt", payload)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
