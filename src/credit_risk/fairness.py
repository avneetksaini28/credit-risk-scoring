"""Fairness analysis of lending decisions with Fairlearn.

Convention: the model's 'positive' prediction is DECLINE (predicted default).
Approval rate = 1 - selection rate.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import roc_auc_score


def _approval_rate(y_true, y_pred):
    return 1 - selection_rate(y_true, y_pred)


def _default_rate(y_true, y_pred):
    return float(np.mean(y_true))


def _safe_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan


def group_report(y_true, decline, pd_score, sensitive: pd.Series) -> pd.DataFrame:
    """Per-group outcome and error rates.

    - false_positive_rate = good borrowers wrongly declined (harm to the applicant)
    - false_negative_rate = defaulters wrongly approved     (harm to the lender, and to the borrower who takes on debt they can't repay)
    """
    name = sensitive.name or "group"
    sensitive = sensitive.fillna("Missing").astype(str).reset_index(drop=True).rename(name)
    mf = MetricFrame(
        metrics={
            "applicants": count,
            "actual_default_rate": _default_rate,
            "approval_rate": _approval_rate,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        },
        y_true=np.asarray(y_true),
        y_pred=np.asarray(decline),
        sensitive_features=sensitive,
    )
    out = mf.by_group.copy()
    out.index.name = name
    out["mean_pd"] = pd.Series(np.asarray(pd_score)).groupby(sensitive.to_numpy()).mean()
    out["auc"] = pd.Series(
        {g: _safe_auc(np.asarray(y_true)[sensitive.to_numpy() == g], np.asarray(pd_score)[sensitive.to_numpy() == g]) for g in out.index}
    )
    from .features import AGE_LABELS
    if set(out.index) <= set(AGE_LABELS) | {"Missing", "nan"}:
        out = out.reindex([g for g in AGE_LABELS + ["Missing", "nan"] if g in out.index])
    best = out["approval_rate"].max()
    out["adverse_impact_ratio"] = out["approval_rate"] / best if best > 0 else np.nan
    return out


def summary_metrics(y_true, decline, sensitive: pd.Series, air_threshold: float = 0.8) -> dict:
    sensitive = sensitive.fillna("Missing").astype(str)
    approve = 1 - np.asarray(decline)
    rep = group_report(y_true, decline, np.zeros(len(approve)), sensitive)
    return {
        "demographic_parity_ratio_approval": float(demographic_parity_ratio(np.asarray(y_true), approve, sensitive_features=sensitive)),
        "equalized_odds_difference": float(equalized_odds_difference(np.asarray(y_true), np.asarray(decline), sensitive_features=sensitive)),
        "min_adverse_impact_ratio": float(rep["adverse_impact_ratio"].min()),
        "groups_below_air_threshold": rep.index[rep["adverse_impact_ratio"] < air_threshold].tolist(),
    }


def plot_group_report(report: pd.DataFrame, attribute: str, air_threshold: float = 0.8):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    idx = report.index.astype(str)
    axes[0].bar(idx, report["approval_rate"], color="#81b29a")
    axes[0].set_title(f"Approval rate by {attribute}")
    axes[0].set_ylim(0, 1.18)
    for i, (a, r) in enumerate(zip(report["approval_rate"], report["adverse_impact_ratio"])):
        axes[0].text(i, a + 0.02, f"{a:.0%}\nAIR {r:.2f}", ha="center", fontsize=9, color="#a4161a" if r < air_threshold else "black")
    w = 0.38
    x = np.arange(len(idx))
    axes[1].bar(x - w / 2, report["actual_default_rate"], w, label="Actual default rate", color="#3d405b")
    axes[1].bar(x + w / 2, report["mean_pd"], w, label="Mean predicted PD", color="#f2cc8f")
    axes[1].set_xticks(x, idx)
    axes[1].set_title("Actual vs predicted risk (calibration by group)")
    axes[1].set_ylim(0, 1.3 * max(report["actual_default_rate"].max(), report["mean_pd"].max()))
    axes[1].legend(frameon=False, loc="upper left", fontsize=9)
    axes[2].bar(x - w / 2, report["false_positive_rate"], w, label="Good borrowers declined (FPR)", color="#e07a5f")
    axes[2].bar(x + w / 2, report["false_negative_rate"], w, label="Defaulters approved (FNR)", color="#2a6f97")
    axes[2].set_xticks(x, idx)
    axes[2].set_title("Error rates by group")
    axes[2].set_ylim(0, 1.3 * max(report["false_positive_rate"].max(), report["false_negative_rate"].max()))
    axes[2].legend(frameon=False, loc="upper left", fontsize=9)
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(w_pad=2)
    return fig
