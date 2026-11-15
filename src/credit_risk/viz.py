"""Shared plotting style and figure saving."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from .config import get_paths

COLORS = {"good": "#2a6f97", "bad": "#e07a5f", "neutral": "#3d405b", "accent": "#81b29a", "warn": "#f2cc8f"}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "figure.autolayout": False,
        }
    )


def savefig(fig, name: str, folder: str | Path | None = None) -> Path:
    """Save a figure to reports/figures (used by the README and the dashboard)."""
    folder = Path(folder) if folder else get_paths()["figures"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path
