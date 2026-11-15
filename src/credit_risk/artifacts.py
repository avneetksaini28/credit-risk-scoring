"""Helpers to read the artifacts that notebooks hand to each other."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .modeling import CreditRiskModel
from .preprocessing import load_feature_lists


def load_features(paths: dict, which: str = "train") -> pd.DataFrame:
    return pd.read_parquet(paths["processed_data"] / f"features_{which}.parquet")


def load_split_frames(paths: dict, features: pd.DataFrame | None = None, id_col: str = "SK_ID_CURR"):
    """Return (train, valid, test) DataFrames using the split saved in notebook 07."""
    features = features if features is not None else load_features(paths, "train")
    ids = pd.read_parquet(paths["processed_data"] / "split_ids.parquet")
    merged = features.merge(ids, on=id_col, how="inner")
    out = tuple(merged[merged["split"] == s].drop(columns="split").reset_index(drop=True) for s in ["train", "valid", "test"])
    return out


def load_feature_contract(paths: dict) -> tuple[list[str], list[str]]:
    return load_feature_lists(paths["models"] / "feature_lists.json")


def load_final_model(paths: dict) -> CreditRiskModel:
    return CreditRiskModel.load(paths["models"] / "final_model.joblib")


def write_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def read_json(path: Path):
    return json.loads(Path(path).read_text())
