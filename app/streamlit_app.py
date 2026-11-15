"""Credit Risk Scoring dashboard (Stage 13).

Run from the project root:   streamlit run app/streamlit_app.py
Requires the artifacts produced by notebooks 08-13.

EDUCATIONAL PROTOTYPE. Not a production credit-decisioning system; never use it for real lending decisions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from credit_risk.config import get_paths, load_config  # noqa: E402
from credit_risk.explainability import get_explainer, reason_codes, shap_values  # noqa: E402
from credit_risk.features import assemble_features  # noqa: E402
from credit_risk.modeling import MODEL_LABELS, CreditRiskModel  # noqa: E402
from credit_risk.thresholds import assign_decision, assign_grade  # noqa: E402

st.set_page_config(page_title="Credit Risk Scoring", page_icon="📊", layout="wide")

CFG = load_config()
PATHS = get_paths(CFG, create=False)
ID, TARGET = CFG["data"]["id_column"], CFG["data"]["target"]
DECISION_COLORS = {"Approve": "#2a9d8f", "Refer to underwriter": "#e9c46a", "Decline": "#e76f51"}
GRADE_COLORS = {"A": "#2a9d8f", "B": "#8ab17d", "C": "#e9c46a", "D": "#f4a261", "E": "#e76f51"}


# ----------------------------------------------------------------------------------------------
# Version-tolerant display helpers (Streamlit renamed use_container_width -> width)
# ----------------------------------------------------------------------------------------------
def show_chart(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_table(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def money(x: float) -> str:
    return f"{x:,.0f}"


# ----------------------------------------------------------------------------------------------
# Artifact loading (cached)
# ----------------------------------------------------------------------------------------------
REQUIRED = {
    "model": PATHS["models"] / "final_model.joblib",
    "policy": PATHS["models"] / "decision_policy.json",
    "portfolio": PATHS["processed_data"] / "dashboard_portfolio.parquet",
    "applicants": PATHS["processed_data"] / "dashboard_applicants.parquet",
    "history": PATHS["processed_data"] / "dashboard_history.parquet",
}


@st.cache_resource(show_spinner="Loading model...")
def load_model() -> CreditRiskModel:
    return CreditRiskModel.load(REQUIRED["model"])


@st.cache_data
def load_policy() -> dict:
    return json.loads(REQUIRED["policy"].read_text())


@st.cache_data
def load_frame(key: str) -> pd.DataFrame:
    return pd.read_parquet(REQUIRED[key])


@st.cache_data
def load_report(name: str):
    p = PATHS["reports"] / name
    if not p.exists():
        return None
    if p.suffix == ".json":
        return json.loads(p.read_text())
    return pd.read_csv(p)


@st.cache_resource(show_spinner="Preparing explainer...")
def load_explainer():
    model = load_model()
    background = assemble_features(load_frame("applicants").head(300), load_frame("history"))
    return get_explainer(model, background)


missing = [k for k, p in REQUIRED.items() if not p.exists()]
if missing:
    st.error("Dashboard artifacts not found: " + ", ".join(missing))
    st.markdown("Run notebooks **01 → 13** first (or `python scripts/run_notebooks.py`), then restart the app.")
    st.stop()

model = load_model()
policy = load_policy()
portfolio = load_frame("portfolio")
applicants = load_frame("applicants")
history = load_frame("history")
test_metrics = load_report("test_metrics.json") or {}

# ----------------------------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------------------------
st.sidebar.title("📊 Credit Risk Scoring")
page = st.sidebar.radio("Navigate", ["Portfolio overview", "Applicant scoring", "Model performance", "Fairness"])
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Model:** {MODEL_LABELS.get(model.name, model.name)}  \n"
    f"**Decline if PD ≥** {policy['decline_at_or_above']:.1%}  \n"
    f"**Auto-approve if PD <** {policy['auto_approve_below']:.1%}  \n"
    f"**LGD** {policy['lgd']:.0%} · **Margin** {policy['profit_margin']:.0%}"
)
st.sidebar.warning(
    "**Experimental prototype.** Trained on the public Home Credit dataset for a data-science portfolio. "
    "Not for real lending decisions."
)


# ----------------------------------------------------------------------------------------------
# Page 1: Portfolio overview
# ----------------------------------------------------------------------------------------------
def page_overview():
    st.title("Credit risk overview")
    st.caption("Hold-out test set: applicants the model never saw during training, scored with the deployed policy.")

    approved = portfolio["DECISION"] != "Decline"
    c = st.columns(6)
    c[0].metric("Applicants", f"{len(portfolio):,}")
    c[1].metric("Actual default rate", f"{portfolio[TARGET].mean():.2%}")
    c[2].metric("Mean predicted PD", f"{portfolio['PD'].mean():.2%}")
    c[3].metric("Approval rate (incl. refer)", f"{approved.mean():.1%}")
    c[4].metric("Bad rate of approved", f"{portfolio.loc[approved, TARGET].mean():.2%}",
                delta=f"{portfolio.loc[approved, TARGET].mean() - portfolio[TARGET].mean():+.2%} vs all", delta_color="inverse")
    c[5].metric("Gini (test)", f"{test_metrics.get('gini', float('nan')):.3f}", help=f"AUC {test_metrics.get('roc_auc', float('nan')):.3f}, KS {test_metrics.get('ks', float('nan')):.3f}")

    left, right = st.columns(2)
    with left:
        dec = (portfolio.groupby("DECISION").agg(applicants=(ID, "size"), default_rate=(TARGET, "mean"), mean_pd=("PD", "mean"),
                                                expected_loss=("EXPECTED_LOSS", "sum"))
               .reindex(["Approve", "Refer to underwriter", "Decline"]).dropna(how="all").reset_index())
        fig = px.bar(dec, x="DECISION", y="applicants", color="DECISION", color_discrete_map=DECISION_COLORS,
                     text=dec["default_rate"].map(lambda v: f"actual default {v:.1%}"), title="Decision mix under the current policy")
        fig.update_layout(showlegend=False, height=380)
        show_chart(fig)
    with right:
        grades = (portfolio.groupby("RISK_GRADE").agg(applicants=(ID, "size"), default_rate=(TARGET, "mean"), mean_pd=("PD", "mean"))
                  .reset_index())
        fig = go.Figure()
        fig.add_bar(x=grades["RISK_GRADE"], y=grades["applicants"], name="Applicants",
                    marker_color=[GRADE_COLORS.get(g, "#888") for g in grades["RISK_GRADE"]])
        fig.add_scatter(x=grades["RISK_GRADE"], y=grades["default_rate"], name="Actual default rate", yaxis="y2", mode="lines+markers")
        fig.add_scatter(x=grades["RISK_GRADE"], y=grades["mean_pd"], name="Mean PD", yaxis="y2", mode="lines+markers", line=dict(dash="dash"))
        fig.update_layout(title="Risk grades: volume and default rate", height=380,
                          yaxis2=dict(overlaying="y", side="right", tickformat=".0%", title="Rate"), yaxis_title="Applicants",
                          legend=dict(orientation="h", y=-0.2))
        show_chart(fig)

    left, right = st.columns(2)
    with left:
        df = portfolio.assign(Outcome=portfolio[TARGET].map({0: "Repaid", 1: "Defaulted"}))
        fig = px.histogram(df, x="PD", color="Outcome", nbins=60, histnorm="probability density", barmode="overlay", opacity=0.6,
                           color_discrete_map={"Repaid": "#2a6f97", "Defaulted": "#e07a5f"}, title="PD distribution by actual outcome")
        fig.add_vline(x=policy["decline_at_or_above"], line_dash="dash", annotation_text="Decline cutoff")
        fig.update_layout(height=380)
        show_chart(fig)
    with right:
        seg_col = st.selectbox("Segment by", ["NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "AGE_GROUP", "NAME_CONTRACT_TYPE", "OCCUPATION_TYPE"])
        seg = (portfolio.groupby(seg_col).agg(applicants=(ID, "size"), actual=(TARGET, "mean"), predicted=("PD", "mean"))
               .query("applicants >= 20").sort_values("actual", ascending=False).reset_index())
        fig = go.Figure([go.Bar(x=seg[seg_col], y=seg["actual"], name="Actual default rate", marker_color="#3d405b"),
                         go.Bar(x=seg[seg_col], y=seg["predicted"], name="Mean predicted PD", marker_color="#f2cc8f")])
        fig.update_layout(title=f"Actual vs predicted risk by {seg_col}", yaxis_tickformat=".0%", height=380, barmode="group",
                          legend=dict(orientation="h", y=-0.35))
        show_chart(fig)

    el = portfolio.loc[approved, "EXPECTED_LOSS"].sum()
    st.info(f"**Expected credit loss on approved applications:** {money(el)} "
            f"({el / portfolio.loc[approved, 'AMT_CREDIT'].sum():.2%} of approved exposure), using PD × LGD × credit amount.")


# ----------------------------------------------------------------------------------------------
# Page 2: Applicant scoring
# ----------------------------------------------------------------------------------------------
def score(app_row: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    feats = assemble_features(app_row, history)
    return feats, float(model.predict_pd(feats)[0])


def gauge(pd_value: float) -> go.Figure:
    cut, auto = policy["decline_at_or_above"], policy["auto_approve_below"]
    top = max(0.5, cut * 2)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=100 * pd_value, number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Probability of default"},
        gauge={"axis": {"range": [0, 100 * top]}, "bar": {"color": "#264653"},
               "steps": [{"range": [0, 100 * auto], "color": "#cde8e5"},
                         {"range": [100 * auto, 100 * cut], "color": "#f7e7b4"},
                         {"range": [100 * cut, 100 * top], "color": "#f6c9bc"}],
               "threshold": {"line": {"color": "#e76f51", "width": 3}, "value": 100 * cut}},
    ))
    fig.update_layout(height=260, margin=dict(t=50, b=10, l=30, r=30))
    return fig


def page_applicant():
    st.title("Applicant scoring")
    st.caption("Pick a hold-out applicant, adjust the application, and see the PD, risk grade, reasons and an experimental recommendation.")

    merged = applicants[[ID]].merge(portfolio[[ID, "DECISION", "PD"]], on=ID, how="left")
    c1, c2 = st.columns([1, 2])
    filt = c1.selectbox("Filter by current decision", ["All", "Approve", "Refer to underwriter", "Decline"])
    pool = merged if filt == "All" else merged[merged["DECISION"] == filt]
    if pool.empty:
        st.warning("No applicants in this group.")
        return
    options = pool.sort_values("PD", ascending=False)[ID].tolist()
    app_id = c2.selectbox("Applicant (SK_ID_CURR), sorted riskiest first", options,
                          format_func=lambda i: f"{i}  ·  stored PD {merged.set_index(ID).loc[i, 'PD']:.1%}")
    base = applicants[applicants[ID] == app_id].reset_index(drop=True)
    r = base.iloc[0]

    with st.expander("Applicant profile", expanded=False):
        prof_cols = ["NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
                     "NAME_HOUSING_TYPE", "CNT_CHILDREN", "REGION_RATING_CLIENT_W_CITY"]
        hist = history[history[ID] == app_id]
        prof = {c: r.get(c) for c in prof_cols if c in base}
        prof["Age (years)"] = round(-r["DAYS_BIRTH"] / 365.25, 1)
        prof["Years employed"] = None if pd.isna(r["DAYS_EMPLOYED"]) else round(-r["DAYS_EMPLOYED"] / 365.25, 1)
        if len(hist):
            h = hist.iloc[0]
            for c, lab in [("BUREAU_LOAN_COUNT", "Bureau credit records"), ("PREV_APP_COUNT", "Previous Home Credit applications"),
                           ("INST_LATE_RATE", "Share of past instalments late"), ("CC_UTILIZATION_MEAN", "Avg card utilisation")]:
                if c in h:
                    prof[lab] = h[c]
        show_table(pd.Series(prof, name="value").astype(str).to_frame())

    st.subheader("What-if: edit the application")
    f = st.columns(4)
    income = f[0].number_input("Annual income", min_value=1000.0, value=float(r["AMT_INCOME_TOTAL"]), step=5000.0)
    credit = f[1].number_input("Loan amount (AMT_CREDIT)", min_value=1000.0, value=float(r["AMT_CREDIT"]), step=5000.0)
    annuity_default = float(r["AMT_ANNUITY"]) if pd.notna(r["AMT_ANNUITY"]) else credit / 20
    annuity = f[2].number_input("Annuity (AMT_ANNUITY)", min_value=100.0, value=annuity_default, step=500.0)
    goods_default = float(r["AMT_GOODS_PRICE"]) if pd.notna(r["AMT_GOODS_PRICE"]) else credit
    goods = f[3].number_input("Goods price", min_value=0.0, value=goods_default, step=5000.0)
    e = st.columns(3)
    ext = {}
    for col, k in zip(e, ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]):
        is_missing = pd.isna(r[k])
        miss = col.checkbox(f"{k} missing", value=bool(is_missing), key=f"miss_{k}_{app_id}")
        ext[k] = np.nan if miss else col.slider(k, 0.0, 1.0, 0.5 if is_missing else float(r[k]), 0.01, key=f"sl_{k}_{app_id}")

    edited = base.copy()
    edited.loc[0, ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]] = [income, credit, annuity, goods]
    for k, v in ext.items():
        edited.loc[0, k] = v
    feats, pd_value = score(edited)
    grade = assign_grade([pd_value], policy["risk_grades"])[0]
    decision = assign_decision([pd_value], policy)[0]
    lgd, margin = policy["lgd"], policy["profit_margin"]
    exp_loss = pd_value * lgd * credit
    exp_profit = (1 - pd_value) * margin * credit - exp_loss

    st.subheader("Result")
    g, m = st.columns([1.2, 2])
    with g:
        show_chart(gauge(pd_value))
    with m:
        k = st.columns(3)
        k[0].metric("Risk grade", grade)
        k[1].metric("Expected loss", money(exp_loss), help="PD × LGD × loan amount")
        k[2].metric("Expected profit", money(exp_profit), help="(1 − PD) × margin × loan − expected loss")
        color = DECISION_COLORS[decision]
        st.markdown(
            f"<div style='padding:14px;border-radius:8px;background:{color}22;border-left:6px solid {color}'>"
            f"<b>Experimental lending recommendation: {decision.upper()}</b><br>"
            f"<span style='font-size:0.9em'>PD {pd_value:.1%} vs decline cutoff {policy['decline_at_or_above']:.1%} and "
            f"auto-approve below {policy['auto_approve_below']:.1%}</span></div>", unsafe_allow_html=True)
        if decision == "Approve":
            st.success("Within risk appetite. Suggested next step: standard verification; price according to risk grade " + grade + ".")
        elif decision == "Refer to underwriter":
            st.warning("Borderline. Suggested next step: manual review, verify income and employment, consider a smaller amount or shorter term.")
        else:
            st.error("Above risk appetite. Principal reasons are listed below (adverse-action notice).")

        if decision != "Approve":
            # Counter-offer search: largest loan amount (same term) that falls into the auto-approve band
            offer = None
            for share in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
                trial = edited.copy()
                trial.loc[0, ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]] = [credit * share, annuity * share, goods * share]
                _, p_trial = score(trial)
                if p_trial < policy["auto_approve_below"]:
                    offer = (share, p_trial)
                    break
            if offer:
                st.info(f"**Possible counter-offer:** a loan of {money(credit * offer[0])} ({offer[0]:.0%} of the request) scores PD {offer[1]:.1%} → auto-approve.")
            else:
                st.caption("No smaller loan amount (down to 30%) brings this applicant into the auto-approve band.")

    st.subheader("Why? Main factors behind this score")
    try:
        explainer = load_explainer()
        exp = shap_values(model, feats, explainer)
        codes = reason_codes(exp[0], top_n=5, raw_values=feats.iloc[0])
        codes["impact"] = codes["shap"]
        fig = px.bar(codes.sort_values("impact"), x="impact", y="label", orientation="h", color="direction",
                     color_discrete_map={"Increases risk": "#e76f51", "Decreases risk": "#2a9d8f"},
                     hover_data={"value": True, "feature": True, "impact": ":.3f"}, title="Top factors (SHAP contribution to model score)")
        fig.update_layout(height=420, yaxis_title="", xaxis_title="← lowers risk · raises risk →", legend_title="")
        show_chart(fig)
        if decision != "Approve":
            ups = codes[codes["shap"] > 0]["label"].head(4).tolist()
            st.markdown("**Principal reasons (adverse-action style):**\n" + "\n".join(f"{i}. {u}" for i, u in enumerate(ups, 1)))
    except Exception as ex:  # explanations are non-critical for the page
        st.caption(f"Explanation unavailable: {ex}")

    actual = portfolio.set_index(ID).loc[app_id, TARGET] if app_id in set(portfolio[ID]) else None
    if actual is not None:
        st.caption(f"Hold-out ground truth for this applicant: {'defaulted' if actual == 1 else 'repaid'} (shown for demonstration only).")


# ----------------------------------------------------------------------------------------------
# Page 3: Model performance
# ----------------------------------------------------------------------------------------------
def page_performance():
    st.title("Model performance")
    comp = load_report("model_comparison_test.csv")
    if comp is not None:
        st.subheader("Model comparison (test set)")
        show_table(comp.rename(columns={comp.columns[0]: "model"}).set_index("model").style.format("{:.4f}"))

    dec = load_report("decile_table_test.csv")
    sweep = load_report("threshold_sweep_validation.csv")
    left, right = st.columns(2)
    if dec is not None:
        with left:
            fig = go.Figure()
            fig.add_bar(x=dec["decile"], y=dec["default_rate"], name="Actual default rate", marker_color="#e07a5f")
            fig.add_scatter(x=dec["decile"], y=dec["mean_pd"], name="Mean predicted PD", mode="lines+markers", line_color="#3d405b")
            fig.update_layout(title="Default rate by risk decile (1 = riskiest)", yaxis_tickformat=".0%", height=400,
                              xaxis_title="Decile", legend=dict(orientation="h", y=-0.2))
            show_chart(fig)
    if sweep is not None:
        with right:
            fig = go.Figure()
            fig.add_scatter(x=sweep["threshold"], y=sweep["net_profit"], name="Net profit", line_color="#2a6f97")
            fig.add_scatter(x=sweep["threshold"], y=sweep["loss_false_approvals"], name="Loss: false approvals", line_color="#e07a5f")
            fig.add_scatter(x=sweep["threshold"], y=sweep["lost_profit_false_rejections"], name="Lost profit: false rejections",
                            line=dict(color="#3d405b", dash="dash"))
            fig.add_vline(x=policy["decline_at_or_above"], line_dash="dot", annotation_text="chosen cutoff")
            fig.update_layout(title="Profit vs PD cutoff (validation)", xaxis_title="Decline if PD ≥", height=400,
                              legend=dict(orientation="h", y=-0.2))
            show_chart(fig)

    imp = load_report("shap_global_importance.csv")
    if imp is not None:
        top = imp.head(15).iloc[::-1]
        fig = px.bar(top, x="mean_abs_shap", y="label", orientation="h", title="Global risk drivers (mean |SHAP|)")
        fig.update_traces(marker_color="#2a6f97")
        fig.update_layout(height=480, yaxis_title="", xaxis_title="Average impact")
        show_chart(fig)

    figs = [("eval_roc_pr_test.png", "ROC and precision-recall"), ("eval_calibration_ks.png", "Calibration and score separation")]
    cols = st.columns(len(figs))
    for col, (fname, cap) in zip(cols, figs):
        p = PATHS["figures"] / fname
        if p.exists():
            col.image(str(p), caption=cap)


# ----------------------------------------------------------------------------------------------
# Page 4: Fairness
# ----------------------------------------------------------------------------------------------
def page_fairness():
    st.title("Fairness monitoring")
    st.caption("Outcomes across groups on the test set. Gender and marital status are NOT model inputs. They are used here only to test outcomes.")
    rep = load_report("fairness_report.csv")
    if rep is None:
        st.warning("Run notebook 12 to generate the fairness report.")
        return
    attr = st.selectbox("Attribute", rep["attribute"].unique().tolist())
    d = rep[rep["attribute"] == attr].copy()
    threshold = CFG["fairness"]["air_threshold"]
    c = st.columns(3)
    c[0].metric("Lowest adverse impact ratio", f"{d['adverse_impact_ratio'].min():.2f}",
                delta="below 0.80 → review" if d["adverse_impact_ratio"].min() < threshold else "≥ 0.80", delta_color="off")
    c[1].metric("Approval-rate gap", f"{d['approval_rate'].max() - d['approval_rate'].min():.1%}")
    c[2].metric("FPR gap (good borrowers declined)", f"{d['false_positive_rate'].max() - d['false_positive_rate'].min():.1%}")

    left, right = st.columns(2)
    with left:
        fig = px.bar(d, x="group", y="approval_rate", text=d["adverse_impact_ratio"].map(lambda v: f"AIR {v:.2f}"),
                     title="Approval rate by group", color=d["adverse_impact_ratio"] < threshold,
                     color_discrete_map={False: "#2a9d8f", True: "#e76f51"})
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%", height=400)
        show_chart(fig)
    with right:
        long = d.melt(id_vars="group", value_vars=["actual_default_rate", "mean_pd", "false_positive_rate"], var_name="metric")
        fig = px.bar(long, x="group", y="value", color="metric", barmode="group", title="Calibration and error rates by group")
        fig.update_layout(yaxis_tickformat=".0%", height=400, legend=dict(orientation="h", y=-0.25))
        show_chart(fig)
    show_table(d.drop(columns=["attribute"]).set_index("group").style.format("{:.3f}"))
    st.markdown(
        "- **AIR** = group approval rate ÷ highest group approval rate; below 0.80 triggers a documented review.\n"
        "- Approval gaps can reflect genuine risk differences. Check that **mean PD tracks the actual default rate** in every group.\n"
        "- Group-specific cutoffs would use protected attributes at decision time, so they are **not** used as a fix."
    )


PAGES = {"Portfolio overview": page_overview, "Applicant scoring": page_applicant, "Model performance": page_performance, "Fairness": page_fairness}
PAGES[page]()
