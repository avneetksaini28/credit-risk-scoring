"""Model evaluation: metrics, decile (gains) tables and standard credit-risk plots."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PALETTE = ["#2a6f97", "#e07a5f", "#3d405b", "#81b29a", "#f2cc8f", "#6d597a"]


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov: max distance between the score CDFs of goods and bads."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, y_score)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "pr_auc": average_precision_score(y_true, y_score),
        "ks": ks_statistic(y_true, y_score),
        "brier": brier_score_loss(y_true, np.clip(y_score, 0, 1)),
        "log_loss": log_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def compare_models(predictions: dict[str, tuple], threshold: float = 0.5) -> pd.DataFrame:
    """predictions: {model_name: (y_true, y_score)}"""
    rows = [{"model": name, **classification_metrics(y, p, threshold)} for name, (y, p) in predictions.items()]
    return pd.DataFrame(rows).set_index("model").sort_values("roc_auc", ascending=False)


def decile_table(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    """Gains / lift table: the standard way to present a scorecard to the business.
    Decile 1 = riskiest 10% of applicants."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_score)})
    df["decile"] = pd.qcut(df["p"].rank(method="first", ascending=False), n_bins, labels=range(1, n_bins + 1))
    g = df.groupby("decile", observed=True).agg(
        applicants=("y", "size"), defaults=("y", "sum"), min_pd=("p", "min"), max_pd=("p", "max"), mean_pd=("p", "mean")
    )
    g["default_rate"] = g["defaults"] / g["applicants"]
    g["cum_defaults_captured"] = g["defaults"].cumsum() / g["defaults"].sum()
    g["cum_population"] = g["applicants"].cumsum() / g["applicants"].sum()
    g["lift"] = g["default_rate"] / df["y"].mean()
    return g.reset_index()


# ----------------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------------
def plot_roc_curves(predictions: dict[str, tuple], ax=None):
    ax = ax or plt.subplots(figsize=(6.5, 5.5))[1]
    for i, (name, (y, p)) in enumerate(predictions.items()):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=2, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set(xlabel="False positive rate (good customers flagged)", ylabel="True positive rate (defaults caught)", title="ROC curve")
    ax.legend(loc="lower right", frameon=False)
    return ax


def plot_pr_curves(predictions: dict[str, tuple], ax=None):
    ax = ax or plt.subplots(figsize=(6.5, 5.5))[1]
    base = None
    for i, (name, (y, p)) in enumerate(predictions.items()):
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, color=PALETTE[i % len(PALETTE)], lw=2, label=f"{name} (AP {average_precision_score(y, p):.3f})")
        base = np.mean(y)
    ax.axhline(base, ls="--", color="grey", lw=1, label=f"Base rate {base:.1%}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    ax.legend(loc="upper right", frameon=False)
    return ax


def plot_confusion(y_true, y_score, threshold: float, ax=None, title: str | None = None):
    ax = ax or plt.subplots(figsize=(4.8, 4.2))[1]
    cm = confusion_matrix(y_true, (np.asarray(y_score) >= threshold).astype(int), labels=[0, 1])
    pct = cm / cm.sum()
    labels = np.array([[f"{v:,}\n({p:.1%})" for v, p in zip(r1, r2)] for r1, r2 in zip(cm, pct)])
    ax.imshow(cm, cmap="Blues")
    for (i, j), lab in np.ndenumerate(labels):
        ax.text(j, i, lab, ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
    ax.set_xticks([0, 1], ["Approve\n(pred. repay)", "Decline\n(pred. default)"])
    ax.set_yticks([0, 1], ["Repaid", "Defaulted"])
    ax.set_xlabel("Model decision")
    ax.set_ylabel("Actual outcome")
    ax.set_title(title or f"Confusion matrix @ threshold {threshold:.2f}")
    return ax


def plot_calibration(predictions: dict[str, tuple], n_bins: int = 10, ax=None):
    ax = ax or plt.subplots(figsize=(6, 5.5))[1]
    lim = 0
    for i, (name, (y, p)) in enumerate(predictions.items()):
        frac, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", color=PALETTE[i % len(PALETTE)], lw=2, label=name)
        lim = max(lim, mean_pred.max(), frac.max())
    ax.plot([0, lim], [0, lim], ls="--", color="grey", lw=1, label="Perfect calibration")
    ax.set(xlabel="Predicted PD", ylabel="Observed default rate", title="Calibration (reliability) curve")
    ax.legend(frameon=False)
    return ax


def plot_score_distributions(y_true, y_score, ax=None):
    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    y = np.asarray(y_true)
    p = np.asarray(y_score)
    bins = np.linspace(0, np.quantile(p, 0.995), 60)
    ax.hist(p[y == 0], bins=bins, alpha=0.6, density=True, color=PALETTE[0], label="Repaid")
    ax.hist(p[y == 1], bins=bins, alpha=0.6, density=True, color=PALETTE[1], label="Defaulted")
    ax.set(xlabel="Predicted probability of default", ylabel="Density", title=f"Score separation (KS = {ks_statistic(y, p):.3f})")
    ax.legend(frameon=False)
    return ax


def plot_gains(decile_df: pd.DataFrame, ax=None):
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    x = np.r_[0, decile_df["cum_population"].to_numpy()]
    y = np.r_[0, decile_df["cum_defaults_captured"].to_numpy()]
    ax.plot(x, y, marker="o", color=PALETTE[0], lw=2, label="Model")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="Random")
    ax.set(xlabel="Share of applicants (riskiest first)", ylabel="Share of defaults captured", title="Cumulative gains")
    ax.legend(frameon=False)
    return ax
