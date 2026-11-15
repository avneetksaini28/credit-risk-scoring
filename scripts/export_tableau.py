"""Export aggregated, Tableau-ready CSVs to reports/tableau/.

Only summary tables are exported (no row-level applicant data), so the files can be
published to Tableau Public without redistributing the Kaggle dataset.
Run after notebooks 01-13:   python scripts/export_tableau.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from credit_risk.config import get_paths, load_config  # noqa: E402

MIN_GROUP = 20  # suppress tiny groups

cfg = load_config()
paths = get_paths(cfg, create=False)
out = paths["reports"] / "tableau"
out.mkdir(parents=True, exist_ok=True)

pf = pd.read_parquet(paths["processed_data"] / "dashboard_portfolio.parquet")
policy = json.loads((paths["models"] / "decision_policy.json").read_text())
metrics = json.loads((paths["reports"] / "test_metrics.json").read_text())
sweep = pd.read_csv(paths["reports"] / "threshold_sweep_validation.csv")
approved = pf["DECISION"] != "Decline"

# 1. Headline KPIs
lgd, margin = policy["lgd"], policy["profit_margin"]
def profit(mask):
    y, c = pf["TARGET"], pf["AMT_CREDIT"]
    return float((margin * c * (mask & (y == 0))).sum() - (lgd * c * (mask & (y == 1))).sum())
all_mask = pd.Series(True, index=pf.index)
loss_all = float((lgd * pf["AMT_CREDIT"] * (pf["TARGET"] == 1)).sum())
loss_policy = float((lgd * pf["AMT_CREDIT"] * ((pf["TARGET"] == 1) & approved)).sum())
kpi = pd.DataFrame([{
    "applicants": len(pf),
    "actual_default_rate": pf["TARGET"].mean(),
    "mean_predicted_pd": pf["PD"].mean(),
    "approval_rate": approved.mean(),
    "bad_rate_of_approved": pf.loc[approved, "TARGET"].mean(),
    "decline_cutoff": policy["decline_at_or_above"],
    "auto_approve_below": policy["auto_approve_below"],
    "break_even_pd": policy["break_even_pd"],
    "roc_auc": metrics["roc_auc"], "gini": metrics["gini"], "ks": metrics["ks"], "pr_auc": metrics["pr_auc"],
    "credit_losses_avoided_pct": 1 - loss_policy / loss_all,
    "profit_uplift_vs_approve_all_pct": profit(approved) / profit(all_mask) - 1,
    "champion_model": metrics["champion"],
}])
kpi.to_csv(out / "kpi_summary.csv", index=False)

# 2. Decision mix
order = {"Approve": 1, "Refer to underwriter": 2, "Decline": 3}
dec = (pf.groupby("DECISION").agg(applicants=("TARGET", "size"), actual_default_rate=("TARGET", "mean"),
                                  mean_pd=("PD", "mean"), total_credit=("AMT_CREDIT", "sum"),
                                  expected_loss=("EXPECTED_LOSS", "sum")).reset_index())
dec["share"] = dec["applicants"] / dec["applicants"].sum()
dec["sort_order"] = dec["DECISION"].map(order)
dec.rename(columns={"DECISION": "decision"}).sort_values("sort_order").to_csv(out / "decision_mix.csv", index=False)

# 3. Risk grades
gr = (pf.groupby("RISK_GRADE").agg(applicants=("TARGET", "size"), actual_default_rate=("TARGET", "mean"),
                                   mean_pd=("PD", "mean"), total_credit=("AMT_CREDIT", "sum")).reset_index())
gr["pd_upper_bound"] = gr["RISK_GRADE"].map(policy["risk_grades"])
gr["share"] = gr["applicants"] / gr["applicants"].sum()
gr.rename(columns={"RISK_GRADE": "risk_grade"}).to_csv(out / "risk_grades.csv", index=False)

# 4. Segments (long format: one row per segment value)
seg_cols = {"NAME_INCOME_TYPE": "Income type", "NAME_EDUCATION_TYPE": "Education", "AGE_GROUP": "Age group",
            "NAME_CONTRACT_TYPE": "Contract type", "OCCUPATION_TYPE": "Occupation", "CODE_GENDER": "Gender"}
rows = []
for col, name in seg_cols.items():
    g = pf.assign(_seg=pf[col].fillna("Unknown").astype(str)).groupby("_seg").agg(
        applicants=("TARGET", "size"), actual_default_rate=("TARGET", "mean"), mean_pd=("PD", "mean"),
        approval_rate=("DECISION", lambda s: (s != "Decline").mean()), avg_credit=("AMT_CREDIT", "mean"))
    g = g[g["applicants"] >= MIN_GROUP].reset_index().rename(columns={"_seg": "segment"})
    g.insert(0, "segment_type", name)
    rows.append(g)
pd.concat(rows).to_csv(out / "segments.csv", index=False)

# 5. PD distribution by outcome (histogram bins)
bins = np.round(np.arange(0, 1.0001, 0.01), 2)
pf["_bin"] = pd.cut(pf["PD"], bins=bins, right=False, labels=bins[:-1]).astype(float)
dist = pf.groupby(["_bin", "TARGET"]).size().reset_index(name="applicants")
dist["outcome"] = dist["TARGET"].map({0: "Repaid", 1: "Defaulted"})
dist = dist[dist["applicants"] > 0].rename(columns={"_bin": "pd_bin"}).drop(columns="TARGET")
dist.to_csv(out / "pd_distribution.csv", index=False)

# 6. Cutoff what-if table (validation set) for a Tableau parameter / slider
sw = sweep[["threshold", "approval_rate", "bad_rate_approved", "false_approvals", "false_rejections",
            "recall_defaults", "precision_declines", "loss_false_approvals", "lost_profit_false_rejections", "net_profit"]].copy()
sw["is_chosen_cutoff"] = np.isclose(sw["threshold"], policy["decline_at_or_above"])
sw.rename(columns={"threshold": "pd_cutoff"}).to_csv(out / "cutoff_whatif.csv", index=False)

# 7-10. Copies of existing summary reports, tidied
dec_tbl = pd.read_csv(paths["reports"] / "decile_table_test.csv")
dec_tbl.to_csv(out / "decile_lift.csv", index=False)
mc = pd.read_csv(paths["reports"] / "model_comparison_test.csv")
mc.to_csv(out / "model_comparison.csv", index=False)
shap = pd.read_csv(paths["reports"] / "shap_global_importance.csv").head(20)
shap["source"] = shap["feature"].str.extract(r"^(BUREAU|PREV|INST|POS|CC)_")[0].map(
    {"BUREAU": "Credit bureau", "PREV": "Previous applications", "INST": "Instalment payments",
     "POS": "POS / cash loans", "CC": "Credit cards"}).fillna("Application")
shap["rank"] = range(1, len(shap) + 1)
shap[["rank", "feature", "label", "source", "mean_abs_shap", "share_pct"]].to_csv(out / "risk_drivers.csv", index=False)
fair = pd.read_csv(paths["reports"] / "fairness_report.csv")
fair = fair[fair["applicants"] >= MIN_GROUP]
fair.to_csv(out / "fairness.csv", index=False)

pf.drop(columns="_bin", inplace=True)
for f in sorted(out.glob("*.csv")):
    print(f"  {f.name:<24} {len(pd.read_csv(f)):>5} rows")
print(f"Tableau files written to {out}")
