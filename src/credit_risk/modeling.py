"""Model definitions, training with early stopping, probability calibration and persistence."""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .preprocessing import make_linear_preprocessor, make_rf_preprocessor, make_tree_preprocessor, prepare_X

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


def get_estimator(name: str, pos_weight: float, random_state: int = 42):
    """Model definitions. Imbalance is handled with class weights, not resampling:
    SMOTE-style synthetic borrowers add little on large tabular credit data and distort probabilities."""
    if name == "logistic_regression":
        return LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000, random_state=random_state)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=50, max_features="sqrt", max_samples=0.5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=random_state,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=3000, learning_rate=0.03, max_depth=5, min_child_weight=10, subsample=0.8,
            colsample_bytree=0.5, reg_lambda=2.0, scale_pos_weight=pos_weight, tree_method="hist",
            eval_metric="auc", early_stopping_rounds=150, n_jobs=-1, random_state=random_state,
        )
    if name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=5000, learning_rate=0.02, num_leaves=32, min_child_samples=100, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.4, reg_lambda=2.0, scale_pos_weight=pos_weight,
            metric="auc", n_jobs=-1, random_state=random_state, verbose=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def get_preprocessor(name: str, num_cols, cat_cols):
    if name == "logistic_regression":
        return make_linear_preprocessor(num_cols, cat_cols)
    if name == "random_forest":
        return make_rf_preprocessor(num_cols, cat_cols)
    return make_tree_preprocessor(num_cols, cat_cols)


def train_model(name, X_train, y_train, X_valid, y_valid, num_cols, cat_cols, pos_weight, random_state=42, estimator=None) -> Pipeline:
    """Fit preprocessor on TRAIN only, then the model (with early stopping on VALID for boosting)."""
    prep = get_preprocessor(name, num_cols, cat_cols)
    Xt = prep.fit_transform(prepare_X(X_train, num_cols, cat_cols), y_train)
    Xv = prep.transform(prepare_X(X_valid, num_cols, cat_cols))
    model = estimator if estimator is not None else get_estimator(name, pos_weight, random_state)
    if name == "xgboost":
        model.fit(Xt, y_train, eval_set=[(Xv, y_valid)], verbose=False)
    elif name == "lightgbm":
        # Stop on validation AUC only. With class weights, the (unweighted) validation log-loss rises from the
        # first rounds, and tracking it would halt training after a handful of trees.
        callbacks = [lgb.early_stopping(150, first_metric_only=True, verbose=False), lgb.log_evaluation(0)]
        if "eval_X" in inspect.signature(model.fit).parameters:  # LightGBM >= 4.7
            model.fit(Xt, y_train, eval_X=(Xv,), eval_y=(y_valid,), eval_metric="auc", callbacks=callbacks)
        else:
            model.fit(Xt, y_train, eval_set=[(Xv, y_valid)], eval_metric="auc", callbacks=callbacks)
    else:
        model.fit(Xt, y_train)
    return Pipeline([("prep", prep), ("model", model)])


def best_iteration(pipe: Pipeline) -> int | None:
    m = pipe.named_steps["model"]
    return getattr(m, "best_iteration_", None) or getattr(m, "best_iteration", None)


# ----------------------------------------------------------------------------------------------
# Calibrated scoring wrapper: what gets deployed
# ----------------------------------------------------------------------------------------------
def _logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class CreditRiskModel:
    """Fitted pipeline + Platt calibration + the feature contract. One object to score raw features.

    Class weighting pushes raw scores upward, so they are not probabilities. A PD feeds expected-loss
    and pricing calculations, so it must be calibrated: a predicted 10% should default about 10% of the time.
    Platt scaling is monotonic, so ranking (AUC, Gini) and SHAP reason codes are unchanged.
    """

    name: str
    pipeline: Pipeline
    num_cols: list[str]
    cat_cols: list[str]
    calibrator: LogisticRegression | None = None
    metadata: dict = field(default_factory=dict)

    def raw_score(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(prepare_X(df, self.num_cols, self.cat_cols))[:, 1]

    def calibrate(self, X_valid: pd.DataFrame, y_valid: pd.Series) -> "CreditRiskModel":
        z = _logit(self.raw_score(X_valid)).reshape(-1, 1)
        self.calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(z, y_valid)
        return self

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.raw_score(df)
        if self.calibrator is None:
            return raw
        return self.calibrator.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Model-ready feature matrix (after the fitted preprocessor), used by SHAP."""
        return self.pipeline.named_steps["prep"].transform(prepare_X(df, self.num_cols, self.cat_cols))

    @property
    def estimator(self):
        return self.pipeline.named_steps["model"]

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "CreditRiskModel":
        return joblib.load(path)


# ----------------------------------------------------------------------------------------------
# Optional: small random search for LightGBM (validation-set based, early stopping per trial)
# ----------------------------------------------------------------------------------------------
def tune_lightgbm(X_train, y_train, X_valid, y_valid, num_cols, cat_cols, pos_weight, n_iter=15, random_state=42):
    rng = np.random.default_rng(random_state)
    space = {
        "num_leaves": [16, 24, 32, 48, 64],
        "min_child_samples": [50, 100, 200, 400],
        "colsample_bytree": [0.2, 0.3, 0.4, 0.5, 0.7],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "reg_lambda": [0.0, 1.0, 2.0, 5.0, 10.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "learning_rate": [0.02, 0.03, 0.05],
    }
    results, best = [], (None, -np.inf, None)
    for i in range(n_iter):
        params = {k: v[rng.integers(len(v))] for k, v in space.items()}
        est = clone(get_estimator("lightgbm", pos_weight, random_state)).set_params(**params)
        start = time.time()
        pipe = train_model("lightgbm", X_train, y_train, X_valid, y_valid, num_cols, cat_cols, pos_weight, random_state, estimator=est)
        auc = roc_auc_score(y_valid, pipe.predict_proba(prepare_X(X_valid, num_cols, cat_cols))[:, 1])
        results.append({**params, "valid_auc": auc, "best_iteration": best_iteration(pipe), "seconds": round(time.time() - start, 1)})
        if auc > best[1]:
            best = (params, auc, pipe)
    return best[0], best[2], pd.DataFrame(results).sort_values("valid_auc", ascending=False)
