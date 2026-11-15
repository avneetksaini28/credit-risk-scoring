"""Cost-sensitive threshold optimisation and the lending decision policy.

Decision rule: DECLINE if PD >= cutoff, otherwise APPROVE.
  - False approval  (approve a borrower who defaults)    -> credit loss  = LGD x AMT_CREDIT
  - False rejection (decline a borrower who would repay) -> lost profit  = margin x AMT_CREDIT
  - True approval   (approve a borrower who repays)      -> profit       = margin x AMT_CREDIT

With calibrated PDs, the break-even PD at which expected profit = 0 is
    (1 - PD) x margin = PD x LGD   =>   PD* = margin / (margin + LGD)
The empirical profit-maximising cutoff on validation data should land close to this value.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def break_even_pd(lgd: float, margin: float) -> float:
    return margin / (margin + lgd)


def threshold_sweep(y_true, pd_score, exposure, lgd: float, margin: float, thresholds=None) -> pd.DataFrame:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(pd_score, dtype="float64")
    ead = np.asarray(exposure, dtype="float64")
    thresholds = np.round(np.arange(0.01, 0.91, 0.005), 3) if thresholds is None else thresholds
    rows = []
    total_bad_loss = (lgd * ead * y).sum()  # loss if we approved everyone
    for t in thresholds:
        approve = p < t
        tp = int(((~approve) & (y == 1)).sum())   # correctly declined defaulters
        fp = int(((~approve) & (y == 0)).sum())   # false rejections
        tn = int((approve & (y == 0)).sum())      # correct approvals
        fn = int((approve & (y == 1)).sum())      # false approvals
        loss_false_approvals = float((lgd * ead * (approve & (y == 1))).sum())
        lost_profit_false_rejections = float((margin * ead * ((~approve) & (y == 0))).sum())
        profit_good_approvals = float((margin * ead * (approve & (y == 0))).sum())
        rows.append(
            {
                "threshold": t,
                "approval_rate": approve.mean(),
                "bad_rate_approved": y[approve].mean() if approve.any() else np.nan,
                "false_approvals": fn,
                "false_rejections": fp,
                "true_approvals": tn,
                "true_rejections": tp,
                "recall_defaults": tp / max(tp + fn, 1),
                "precision_declines": tp / max(tp + fp, 1),
                "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                "specificity": tn / max(tn + fp, 1),
                "loss_false_approvals": loss_false_approvals,
                "lost_profit_false_rejections": lost_profit_false_rejections,
                "total_cost": loss_false_approvals + lost_profit_false_rejections,
                "net_profit": profit_good_approvals - loss_false_approvals,
                "expected_loss_approved": float((p * lgd * ead * approve).sum()),
                "losses_avoided_pct": 1 - loss_false_approvals / total_bad_loss if total_bad_loss else np.nan,
            }
        )
    return pd.DataFrame(rows)


def candidate_thresholds(sweep: pd.DataFrame, lgd: float, margin: float) -> pd.DataFrame:
    """Compare common ways of choosing a cutoff. Only the profit one reflects business costs."""
    picks = {
        "Default 0.50": sweep.iloc[(sweep["threshold"] - 0.5).abs().argmin()],
        "Max F1": sweep.loc[sweep["f1"].idxmax()],
        "Youden J (recall + specificity)": sweep.loc[(sweep["recall_defaults"] + sweep["specificity"]).idxmax()],
        "Break-even PD (analytic)": sweep.iloc[(sweep["threshold"] - break_even_pd(lgd, margin)).abs().argmin()],
        "Max net profit (empirical)": sweep.loc[sweep["net_profit"].idxmax()],
    }
    cols = ["threshold", "approval_rate", "bad_rate_approved", "false_approvals", "false_rejections",
            "recall_defaults", "precision_declines", "loss_false_approvals", "lost_profit_false_rejections", "net_profit"]
    return pd.DataFrame({k: v[cols] for k, v in picks.items()}).T


def plot_profit_curve(sweep: pd.DataFrame, chosen: float, break_even: float | None = None, ax=None):
    ax = ax or plt.subplots(figsize=(8, 4.8))[1]
    scale = 1e6
    ax.plot(sweep["threshold"], sweep["net_profit"] / scale, color="#2a6f97", lw=2.2, label="Net profit")
    ax.plot(sweep["threshold"], sweep["loss_false_approvals"] / scale, color="#e07a5f", lw=1.6, label="Loss from false approvals")
    ax.plot(sweep["threshold"], sweep["lost_profit_false_rejections"] / scale, color="#3d405b", lw=1.6, ls="--", label="Profit lost to false rejections")
    ax.axvline(chosen, color="#2a6f97", ls=":", lw=1.5)
    ax.annotate(f"Chosen cutoff {chosen:.3f}", (chosen, ax.get_ylim()[1] * 0.92), xytext=(6, 0), textcoords="offset points", color="#2a6f97")
    if break_even is not None:
        ax.axvline(break_even, color="grey", ls="--", lw=1)
    ax.set(xlabel="PD cutoff (decline if PD ≥ cutoff)", ylabel="Millions (currency units)", title="Profit vs PD cutoff (validation set)")
    ax.legend(frameon=True, framealpha=0.9, loc="upper right", fontsize=9)
    return ax


def plot_tradeoff(sweep: pd.DataFrame, chosen: float, ax=None):
    ax = ax or plt.subplots(figsize=(8, 4.8))[1]
    ax.plot(sweep["threshold"], sweep["approval_rate"], color="#81b29a", lw=2, label="Approval rate")
    ax2 = ax.twinx()
    ax2.plot(sweep["threshold"], sweep["bad_rate_approved"], color="#e07a5f", lw=2, label="Bad rate among approved")
    ax.axvline(chosen, color="grey", ls=":")
    ax.set(xlabel="PD cutoff", ylabel="Approval rate")
    ax2.set_ylabel("Bad rate among approved")
    ax.set_title("Approval rate vs portfolio bad rate")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=True, framealpha=0.9, loc="lower right", fontsize=9)
    return ax


# ----------------------------------------------------------------------------------------------
# Decision policy
# ----------------------------------------------------------------------------------------------
def make_policy(decline_cutoff: float, refer_ratio: float, lgd: float, margin: float, grades: dict[str, float]) -> dict:
    return {
        "decline_at_or_above": float(decline_cutoff),
        "auto_approve_below": float(decline_cutoff * refer_ratio),
        "lgd": lgd,
        "profit_margin": margin,
        "break_even_pd": break_even_pd(lgd, margin),
        "risk_grades": grades,
    }


def assign_decision(pd_score, policy: dict) -> np.ndarray:
    p = np.asarray(pd_score)
    return np.where(
        p >= policy["decline_at_or_above"], "Decline",
        np.where(p >= policy["auto_approve_below"], "Refer to underwriter", "Approve"),
    )


def assign_grade(pd_score, grades: dict[str, float]) -> np.ndarray:
    p = np.atleast_1d(np.asarray(pd_score))
    labels = list(grades.keys())
    edges = list(grades.values())
    idx = np.searchsorted(edges, p, side="right")
    idx = np.clip(idx, 0, len(labels) - 1)
    return np.array(labels)[idx]


def save_policy(policy: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(policy, indent=2))


def load_policy(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
