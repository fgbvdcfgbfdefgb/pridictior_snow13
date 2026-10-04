from __future__ import annotations

import inspect
from typing import Any


def _construct(cls, config: dict[str, Any]):
    allowed = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return cls(**{key: value for key, value in config.items() if key in allowed})


def build_model(config: dict[str, Any]):
    cfg = dict(config)
    architecture = cfg.pop("architecture")
    if architecture == "patch_transformer":
        from .patch_transformer import PatchTransformer

        return _construct(PatchTransformer, cfg)
    if architecture == "dilated_tcn":
        from .dilated_tcn import DilatedTCN

        return _construct(DilatedTCN, cfg)
    if architecture == "gru_attention":
        from .gru_attention import GRUAttention

        return _construct(GRUAttention, cfg)
    raise ValueError(f"unknown architecture {architecture!r}")
