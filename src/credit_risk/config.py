"""Configuration loading and project paths."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@lru_cache(maxsize=4)
def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | Path | None = None) -> dict:
    """Load the YAML config. Returns a fresh copy so callers can modify it safely."""
    import copy

    cfg = _read_yaml(str(path or CONFIG_PATH))
    return copy.deepcopy(cfg)


def get_paths(cfg: dict | None = None, create: bool = True) -> dict[str, Path]:
    """Resolve every configured path against the project root.

    The raw-data folder can be overridden with the CREDIT_RISK_RAW_DIR environment variable,
    which is how the synthetic sample data is used for smoke tests.
    """
    cfg = cfg or load_config()
    paths = {key: PROJECT_ROOT / value for key, value in cfg["paths"].items()}
    # Heavy, regenerable files (raw CSVs, database, features, models) can live outside the project,
    # e.g. outside a OneDrive-synced portfolio folder. Code, notebooks and reports stay in the project.
    work = os.environ.get("CREDIT_RISK_WORK_DIR") or cfg.get("work_dir")
    if work:
        w = Path(os.path.expanduser(str(work)))
        for key, sub in {"raw_data": "raw", "interim_data": "interim", "processed_data": "processed",
                         "database": "database/home_credit.db", "models": "models"}.items():
            paths[key] = w / sub
    raw_override = os.environ.get("CREDIT_RISK_RAW_DIR")
    if raw_override:
        p = Path(raw_override)
        paths["raw_data"] = p if p.is_absolute() else PROJECT_ROOT / p
    if create:
        for key, p in paths.items():
            target = p.parent if p.suffix else p
            target.mkdir(parents=True, exist_ok=True)
    return paths
