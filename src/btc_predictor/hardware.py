from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    total_vram_gib: float
    capability: str


@dataclass(frozen=True)
class HardwareInfo:
    cpu_logical: int
    ram_gib: float
    disk_free_gib: float
    gpus: tuple[GPUInfo, ...]
    torch_version: str | None
    cuda_version: str | None
    bf16_supported: bool
    platform: str


def _ram_gib() -> float:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / 2**30
    except (ValueError, OSError, AttributeError):
        return 0.0


def detect_hardware(path: str | Path = ".") -> HardwareInfo:
    gpus: list[GPUInfo] = []
    torch_version = cuda_version = None
    bf16 = False
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                gpus.append(
                    GPUInfo(
                        index=idx,
                        name=props.name,
                        total_vram_gib=round(props.total_memory / 2**30, 2),
                        capability=f"{props.major}.{props.minor}",
                    )
                )
            bf16 = bool(torch.cuda.is_bf16_supported())
    except ImportError:
        pass
    return HardwareInfo(
        cpu_logical=os.cpu_count() or 1,
        ram_gib=round(_ram_gib(), 2),
        disk_free_gib=round(shutil.disk_usage(path).free / 2**30, 2),
        gpus=tuple(gpus),
        torch_version=torch_version,
        cuda_version=cuda_version,
        bf16_supported=bf16,
        platform=platform.platform(),
    )


def recommended_profile(hw: HardwareInfo) -> dict:
    """Conservative per-rank settings; DDP duplicates a model on each GPU."""
    min_vram = min((gpu.total_vram_gib for gpu in hw.gpus), default=0.0)
    if min_vram >= 40:
        d_model, layers, micro = 1024, 16, 8
    elif min_vram >= 22:  # A10 24 GB profile
        d_model, layers, micro = 768, 12, 4
    elif min_vram >= 14:
        d_model, layers, micro = 512, 10, 2
    elif min_vram >= 8:
        d_model, layers, micro = 384, 8, 1
    else:
        d_model, layers, micro = 256, 6, 1
    world = max(1, len(hw.gpus))
    global_batch = max(32, 16 * world)
    accumulation = max(1, global_batch // (micro * world))
    return {
        "world_size": world,
        "distributed_backend": "nccl" if hw.gpus else "gloo",
        "model": {"d_model": d_model, "layers": layers},
        "training": {
            "micro_batch_size": micro,
            "global_batch_size": global_batch,
            "gradient_accumulation": accumulation,
            "num_workers": min(32, max(2, hw.cpu_logical // world - 1)),
            "precision": "bf16" if hw.bf16_supported else ("fp16" if hw.gpus else "fp32"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect compute and print the auto-sizing profile")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    hw = detect_hardware()
    payload = {"hardware": asdict(hw), "recommended": recommended_profile(hw)}
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
