"""Splitting, feature selection and reusable sklearn preprocessing pipelines."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .cleaning import categorical_columns

MISSING_CATEGORY = "Missing"


# ----------------------------------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------------------------------
def split_data(df: pd.DataFrame, target: str, test_size: float, valid_size: float, random_state: int = 42):
    """Stratified train / validation / test split.

    The validation set is for early stopping, probability calibration and choosing the threshold.
    The test set is touched once, for the final report.
    Home Credit has no application date, so an out-of-time split isn't possible. A real bank would
    split by origination month instead.
    """
    train_valid, test = train_test_split(df, test_size=test_size, stratify=df[target], random_state=random_state)
    rel_valid = valid_size / (1 - test_size)
    train, valid = train_test_split(train_valid, test_size=rel_valid, stratify=train_valid[target], random_state=random_state)
    return train, valid, test


def save_split_ids(train, valid, test, path: Path, id_col: str = "SK_ID_CURR") -> None:
    ids = pd.concat(
        [
            pd.DataFrame({id_col: train[id_col], "split": "train"}),
            pd.DataFrame({id_col: valid[id_col], "split": "valid"}),
            pd.DataFrame({id_col: test[id_col], "split": "test"}),
        ]
    )
    ids.to_parquet(path, index=False)


# ----------------------------------------------------------------------------------------------
# Feature lists and selection (fitted on TRAIN only)
# ----------------------------------------------------------------------------------------------
def get_feature_lists(df: pd.DataFrame, id_col: str, target: str, exclude: list[str] | None = None) -> tuple[list[str], list[str]]:
    exclude = set(exclude or []) | {id_col, target}
    cols = [c for c in df.columns if c not in exclude]
    cat = [c for c in categorical_columns(df[cols])]
    num = [c for c in cols if c not in cat]
    return num, cat


def select_features(
    train: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    max_missing_rate: float = 0.95,
    max_dominant_share: float = 0.995,
    max_correlation: float = 0.98,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Unsupervised filter: drop near-empty, near-constant and near-duplicate features.

    Uses only the training split and never looks at the target, so it can't leak or overfit.
    The models' own regularisation and SHAP do the supervised part.
    """
    log = []
    keep_num, keep_cat = [], []
    for col in num_cols + cat_cols:
        s = train[col]
        miss = s.isna().mean()
        top_share = s.value_counts(normalize=True, dropna=False).iloc[0] if len(s) else 1.0
        if miss > max_missing_rate:
            log.append((col, "missing", round(miss, 4)))
        elif top_share > max_dominant_share:
            log.append((col, "near_constant", round(top_share, 4)))
        else:
            (keep_num if col in num_cols else keep_cat).append(col)

    # Correlation filter on numeric features (sampled for speed)
    sample = train[keep_num].sample(min(len(train), 50_000), random_state=0).astype("float32")
    corr = sample.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = set()
    for col in upper.columns:
        partners = upper.index[upper[col] > max_correlation].tolist()
        partners = [p for p in partners if p not in to_drop]
        if partners and col not in to_drop:
            to_drop.add(col)
            log.append((col, f"corr>{max_correlation} with {partners[0]}", round(float(upper[col].max()), 4)))
    keep_num = [c for c in keep_num if c not in to_drop]
    return keep_num, keep_cat, pd.DataFrame(log, columns=["feature", "reason", "value"])


def save_feature_lists(num: list[str], cat: list[str], path: Path) -> None:
    Path(path).write_text(json.dumps({"numeric": num, "categorical": cat}, indent=2))


def load_feature_lists(path: Path) -> tuple[list[str], list[str]]:
    d = json.loads(Path(path).read_text())
    return d["numeric"], d["categorical"]


# ----------------------------------------------------------------------------------------------
# Transformers
# ----------------------------------------------------------------------------------------------
def prepare_X(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    """Deterministic, stateless prep applied before the pipeline: column order, dtypes, missing category."""
    X = df.reindex(columns=num_cols + cat_cols).copy()
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), MISSING_CATEGORY).astype(str)
    return X


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each numeric column to quantiles learned on the training data (limits outlier influence on linear models)."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        lower = X.quantile(self.lower).to_numpy(dtype="float64")
        upper = X.quantile(self.upper).to_numpy(dtype="float64")
        # Never clip binary columns (flags, missing indicators): rare 1s would be wiped out.
        binary = (X.nunique() <= 2).to_numpy()
        self.lower_ = np.where(binary, -np.inf, lower)
        self.upper_ = np.where(binary, np.inf, upper)
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        is_df = isinstance(X, pd.DataFrame)
        arr = np.clip(np.asarray(X, dtype="float64"), self.lower_, self.upper_)
        return pd.DataFrame(arr, columns=X.columns, index=X.index) if is_df else arr

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return getattr(self, "feature_names_in_", np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object))


def make_linear_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """For Logistic Regression: impute + missing flags, cap outliers, standardise; one-hot categoricals."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("winsorize", Winsorizer()),
            ("scale", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005, sparse_output=False)
    ct = ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)], verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


def make_tree_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """For tree ensembles: numerics pass through (trees handle NaN and scale), categoricals ordinal-encoded."""
    categorical = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
    ct = ColumnTransformer([("num", "passthrough", num_cols), ("cat", categorical, cat_cols)], verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


def make_rf_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Random Forest in sklearn needs no NaN: impute with an out-of-range value so trees can isolate 'missing'."""
    numeric = SimpleImputer(strategy="constant", fill_value=-999_999, keep_empty_features=True)
    categorical = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
    ct = ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)], verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


def class_balance(y: pd.Series) -> dict:
    pos = float(y.mean())
    return {
        "n": int(len(y)),
        "defaults": int(y.sum()),
        "default_rate": pos,
        "imbalance_ratio": (1 - pos) / pos,  # negatives per positive -> scale_pos_weight
    }
