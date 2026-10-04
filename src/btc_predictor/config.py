from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(default_path: str | Path, *overrides: str | Path) -> dict[str, Any]:
    with Path(default_path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for path in overrides:
        with Path(path).open(encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f))
    return cfg
