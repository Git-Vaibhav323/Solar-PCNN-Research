"""Shared paths and configuration helpers for Phase 2 preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "preprocess.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key in ("nsrdb_dir", "ninja_path", "processed_dir", "figures_dir"):
        raw = cfg["paths"][key]
        p = Path(raw)
        cfg["paths"][key] = p if p.is_absolute() else ROOT / p
    return cfg
