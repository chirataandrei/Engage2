from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}")
    return data


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load base.yaml and optionally overlay another YAML file."""
    root = project_root()
    base = load_yaml(root / "config" / "base.yaml")
    if path is None:
        return base
    overlay = load_yaml(path)
    return deep_merge(base, overlay)


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def project_root() -> Path:
    """Repo root (Engage2/), four levels above this file."""
    return Path(__file__).resolve().parents[3]
