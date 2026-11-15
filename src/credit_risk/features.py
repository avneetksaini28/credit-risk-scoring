"""Feature engineering.

Design rules that prevent data leakage:
1. Every historical table is filtered to events that happened before the application date
   (DAYS_* <= 0, MONTHS_BALANCE <= 0). Home Credit already measures time relative to the application.
2. No aggregation ever touches TARGET, so features are computed the same way for train and test rows.
3. Aggregations are per client (SK_ID_CURR) and don't depend on other clients, so they can be
   computed once for all rows before splitting. Anything learned across rows (imputation, scaling,
   encoding, outlier caps, feature selection) is fitted on the training split only (see preprocessing.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cleaning import (
    clean_bureau,
    clean_installments,
    clean_monthly_balance,
    clean_previous_application,
)

AGG_PREFIXES = ("BUREAU_", "PREV_", "INST_", "POS_", "CC_")

# Raw DAYS_* columns replaced by easier-to-read YEARS_* columns.
REPLACED_RAW_COLUMNS = ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"]

AGE_BINS = [0, 25, 35, 45, 55, 65, 120]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]


def safe_div(a, b):
    """Division that returns NaN instead of inf when the denominator is 0 or missing."""
    a = pd.Series(a) if not isinstance(a, pd.Series) else a
    b = pd.Series(b, index=a.index) if not isinstance(b, pd.Series) else b
    out = a.astype("float64") / b.replace(0, np.nan).astype("float64")
    return out.replace([np.inf, -np.inf], np.nan)


def _flatten(agg: pd.DataFrame, prefix: str) -> pd.DataFrame:
    agg.columns = [f"{prefix}{c[0]}_{c[1]}".upper() if isinstance(c, tuple) else f"{prefix}{c}".upper() for c in agg.columns]
    return agg


# ----------------------------------------------------------------------------------------------
# 1. Application-level features (row-wise, no aggregation)
# ----------------------------------------------------------------------------------------------
def build_application_features(app: pd.DataFrame) -> pd.DataFrame:
    """Affordability, stability and external-score features from the current application."""
    df = app.copy()

    # Life-stage and stability (in years, easier to explain than DAYS_*)
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["AGE_GROUP"] = pd.cut(df["AGE_YEARS"], bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365.25
    df["YEARS_REGISTRATION"] = -df["DAYS_REGISTRATION"] / 365.25
    df["YEARS_ID_PUBLISH"] = -df["DAYS_ID_PUBLISH"] / 365.25
    df["YEARS_SINCE_PHONE_CHANGE"] = -df["DAYS_LAST_PHONE_CHANGE"] / 365.25
    df["EMPLOYED_TO_AGE_RATIO"] = safe_div(df["YEARS_EMPLOYED"], df["AGE_YEARS"])

    # Affordability ratios: the core of any underwriting decision
    df["CREDIT_TO_INCOME_RATIO"] = safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_TO_INCOME_RATIO"] = safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])  # payment-to-income
    df["CREDIT_TO_ANNUITY_RATIO"] = safe_div(df["AMT_CREDIT"], df["AMT_ANNUITY"])  # proxy for loan term
    df["CREDIT_TO_GOODS_RATIO"] = safe_div(df["AMT_CREDIT"], df["AMT_GOODS_PRICE"])  # loan-to-value proxy
    df["DOWN_PAYMENT_PROXY"] = df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]
    df["INCOME_PER_FAMILY_MEMBER"] = safe_div(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
    df["INCOME_AFTER_ANNUITY"] = df["AMT_INCOME_TOTAL"] - df["AMT_ANNUITY"]

    # External credit scores (the strongest predictors in this dataset)
    ext = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df]
    if ext:
        e = df[ext]
        df["EXT_SOURCE_MEAN"] = e.mean(axis=1)
        df["EXT_SOURCE_MIN"] = e.min(axis=1)
        df["EXT_SOURCE_MAX"] = e.max(axis=1)
        df["EXT_SOURCE_STD"] = e.std(axis=1)
        df["EXT_SOURCE_PROD"] = e.prod(axis=1, min_count=len(ext))
        df["EXT_SOURCE_NAN_COUNT"] = e.isna().sum(axis=1)

    # Documentation and contactability
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)
    contact_cols = [c for c in ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"] if c in df]
    if contact_cols:
        df["CONTACT_FLAG_COUNT"] = df[contact_cols].sum(axis=1)
    req_cols = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    if req_cols:
        df["BUREAU_ENQUIRIES_TOTAL"] = df[req_cols].sum(axis=1, min_count=1)
    if {"DEF_30_CNT_SOCIAL_CIRCLE", "OBS_30_CNT_SOCIAL_CIRCLE"} <= set(df.columns):
        df["SOCIAL_CIRCLE_DEFAULT_RATE_30"] = safe_div(df["DEF_30_CNT_SOCIAL_CIRCLE"], df["OBS_30_CNT_SOCIAL_CIRCLE"])

    return df.drop(columns=[c for c in REPLACED_RAW_COLUMNS if c in df.columns])


# ----------------------------------------------------------------------------------------------
# 2. Credit Bureau history (other lenders)
# ----------------------------------------------------------------------------------------------
BUREAU_STATUS_DPD = {"0": 0, "1": 15, "2": 45, "3": 75, "4": 105, "5": 135}  # midpoint DPD of each bucket


def bureau_balance_features(bb: pd.DataFrame) -> pd.DataFrame:
    """Per bureau credit: months of history and how often it was delinquent."""
    bb = clean_monthly_balance(bb).copy()
    # Keep STATUS categorical: converting 27M values to Python strings would cost ~1.5 GB of RAM.
    status = bb["STATUS"].astype("category")
    bucket = status.map({"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5})  # 'C' (closed) and 'X' (unknown) -> NaN
    bb["DPD_BUCKET"] = pd.to_numeric(pd.Series(bucket, index=bb.index), errors="coerce").astype("float32")
    bb["IS_DPD"] = (bb["DPD_BUCKET"] >= 1).astype("int8")
    bb["IS_DPD_60_PLUS"] = (bb["DPD_BUCKET"] >= 2).astype("int8")
    bb["IS_RECENT"] = (bb["MONTHS_BALANCE"] >= -12).astype("int8")
    bb["DPD_RECENT"] = bb["IS_DPD"] * bb["IS_RECENT"]
    agg = bb.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS=("MONTHS_BALANCE", "count"),
        BB_DPD_RATE=("IS_DPD", "mean"),
        BB_DPD60_COUNT=("IS_DPD_60_PLUS", "sum"),
        BB_MAX_DPD_BUCKET=("DPD_BUCKET", "max"),
        BB_RECENT_MONTHS=("IS_RECENT", "sum"),
        BB_RECENT_DPD=("DPD_RECENT", "sum"),
    )
    return agg.reset_index()


def bureau_features(bureau: pd.DataFrame, bureau_balance: pd.DataFrame | None = None) -> pd.DataFrame:
    b = clean_bureau(bureau)
    if bureau_balance is not None and len(bureau_balance):
        b = b.merge(bureau_balance_features(bureau_balance), on="SK_ID_BUREAU", how="left")

    b["IS_ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype("int8")
    b["IS_CLOSED"] = (b["CREDIT_ACTIVE"] == "Closed").astype("int8")
    b["IS_BAD_DEBT"] = b["CREDIT_ACTIVE"].isin(["Bad debt", "Sold"]).astype("int8")
    b["IS_CREDIT_CARD"] = (b["CREDIT_TYPE"] == "Credit card").astype("int8")
    b["IS_OVERDUE"] = (b["CREDIT_DAY_OVERDUE"] > 0).astype("int8")
    b["OPENED_LAST_YEAR"] = (b["DAYS_CREDIT"] >= -365).astype("int8")
    b["ACTIVE_DEBT"] = b["AMT_CREDIT_SUM_DEBT"].fillna(0) * b["IS_ACTIVE"]
    b["ACTIVE_CREDIT"] = b["AMT_CREDIT_SUM"].fillna(0) * b["IS_ACTIVE"]
    b["CARD_DEBT"] = b["AMT_CREDIT_SUM_DEBT"].fillna(0) * b["IS_CREDIT_CARD"] * b["IS_ACTIVE"]
    b["CARD_LIMIT"] = b["AMT_CREDIT_SUM"].fillna(0) * b["IS_CREDIT_CARD"] * b["IS_ACTIVE"]
    b["ENDDATE_GAP"] = b["DAYS_ENDDATE_FACT"] - b["DAYS_CREDIT_ENDDATE"]  # >0 = closed later than planned

    spec = {
        "SK_ID_BUREAU": ["count"],
        "IS_ACTIVE": ["sum", "mean"],
        "IS_CLOSED": ["sum"],
        "IS_BAD_DEBT": ["sum"],
        "IS_OVERDUE": ["sum"],
        "OPENED_LAST_YEAR": ["sum"],
        "DAYS_CREDIT": ["min", "max", "mean"],
        "DAYS_CREDIT_ENDDATE": ["max"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["max"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "AMT_CREDIT_SUM": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_DEBT": ["sum", "mean"],
        "AMT_CREDIT_SUM_OVERDUE": ["sum"],
        "ACTIVE_DEBT": ["sum"],
        "ACTIVE_CREDIT": ["sum"],
        "CARD_DEBT": ["sum"],
        "CARD_LIMIT": ["sum"],
        "ENDDATE_GAP": ["mean", "max"],
        "CREDIT_TYPE": ["nunique"],
    }
    for col in ["BB_MONTHS", "BB_DPD_RATE", "BB_DPD60_COUNT", "BB_MAX_DPD_BUCKET", "BB_RECENT_DPD"]:
        if col in b:
            spec[col] = ["mean", "max"] if col != "BB_DPD60_COUNT" else ["sum"]
    agg = _flatten(b.groupby("SK_ID_CURR").agg(spec), "BUREAU_")
    agg = agg.rename(columns={"BUREAU_SK_ID_BUREAU_COUNT": "BUREAU_LOAN_COUNT"})

    # Ratios a credit analyst would read off a bureau report
    agg["BUREAU_CREDIT_HISTORY_YEARS"] = -agg["BUREAU_DAYS_CREDIT_MIN"] / 365.25
    agg["BUREAU_YEARS_SINCE_LAST_CREDIT"] = -agg["BUREAU_DAYS_CREDIT_MAX"] / 365.25
    agg["BUREAU_ACTIVE_DEBT_TO_CREDIT"] = safe_div(agg["BUREAU_ACTIVE_DEBT_SUM"], agg["BUREAU_ACTIVE_CREDIT_SUM"])
    agg["BUREAU_CARD_UTILIZATION"] = safe_div(agg["BUREAU_CARD_DEBT_SUM"], agg["BUREAU_CARD_LIMIT_SUM"])
    agg["BUREAU_OVERDUE_TO_DEBT"] = safe_div(agg["BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM"], agg["BUREAU_AMT_CREDIT_SUM_DEBT_SUM"])
    agg = agg.drop(columns=["BUREAU_DAYS_CREDIT_MIN", "BUREAU_DAYS_CREDIT_MAX"])
    return agg.reset_index()


# ----------------------------------------------------------------------------------------------
# 3. Previous applications at Home Credit
# ----------------------------------------------------------------------------------------------
def previous_application_features(prev: pd.DataFrame) -> pd.DataFrame:
    p = clean_previous_application(prev)
    p = p[~(p["DAYS_DECISION"] > 0)].copy()
    status = p["NAME_CONTRACT_STATUS"]
    p["IS_APPROVED"] = (status == "Approved").astype("int8")
    p["IS_REFUSED"] = (status == "Refused").astype("int8")
    p["IS_CANCELED"] = (status == "Canceled").astype("int8")
    p["IS_RECENT"] = (p["DAYS_DECISION"] >= -365).astype("int8")
    p["REFUSED_RECENT"] = p["IS_REFUSED"] * p["IS_RECENT"]
    p["CREDIT_TO_APPLICATION"] = safe_div(p["AMT_CREDIT"], p["AMT_APPLICATION"])  # < 1 = got less than asked
    p["REFUSED_DAYS"] = p["DAYS_DECISION"].where(p["IS_REFUSED"] == 1)

    agg = p.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_COUNT=("IS_APPROVED", "sum"),
        PREV_REFUSED_COUNT=("IS_REFUSED", "sum"),
        PREV_CANCELED_COUNT=("IS_CANCELED", "sum"),
        PREV_APPROVAL_RATE=("IS_APPROVED", "mean"),
        PREV_REFUSAL_RATE=("IS_REFUSED", "mean"),
        PREV_RECENT_APP_COUNT=("IS_RECENT", "sum"),
        PREV_RECENT_REFUSED=("REFUSED_RECENT", "sum"),
        PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_AMT_CREDIT_MAX=("AMT_CREDIT", "max"),
        PREV_AMT_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
        PREV_CREDIT_TO_APPLICATION_MEAN=("CREDIT_TO_APPLICATION", "mean"),
        PREV_CREDIT_TO_APPLICATION_MIN=("CREDIT_TO_APPLICATION", "min"),
        PREV_AMT_DOWN_PAYMENT_MEAN=("AMT_DOWN_PAYMENT", "mean"),
        PREV_RATE_DOWN_PAYMENT_MEAN=("RATE_DOWN_PAYMENT", "mean"),
        PREV_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
        PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),
        PREV_DAYS_DECISION_MEAN=("DAYS_DECISION", "mean"),
        PREV_LAST_REFUSAL_DAYS=("REFUSED_DAYS", "max"),
        PREV_PRODUCT_TYPES=("NAME_CONTRACT_TYPE", "nunique"),
    )
    agg["PREV_YEARS_SINCE_LAST_APP"] = -agg.pop("PREV_DAYS_DECISION_MAX") / 365.25
    agg["PREV_YEARS_SINCE_LAST_REFUSAL"] = -agg.pop("PREV_LAST_REFUSAL_DAYS") / 365.25
    return agg.reset_index()


# ----------------------------------------------------------------------------------------------
# 4. Installment payments: payment delays and shortfalls
# ----------------------------------------------------------------------------------------------
def installments_features(inst: pd.DataFrame) -> pd.DataFrame:
    i = clean_installments(inst)
    # Several partial payments can cover one instalment: collapse to one row per instalment first.
    i = (
        i.groupby(["SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER"], observed=True)
        .agg(
            DAYS_INSTALMENT=("DAYS_INSTALMENT", "first"),
            AMT_INSTALMENT=("AMT_INSTALMENT", "first"),
            DAYS_ENTRY_PAYMENT=("DAYS_ENTRY_PAYMENT", "max"),
            AMT_PAYMENT=("AMT_PAYMENT", "sum"),
        )
        .reset_index()
    )
    i["DAYS_LATE"] = (i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]).clip(lower=0)
    i["DAYS_EARLY"] = (i["DAYS_INSTALMENT"] - i["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    i["IS_LATE"] = (i["DAYS_LATE"] > 0).astype("int8")
    i["IS_LATE_30"] = (i["DAYS_LATE"] > 30).astype("int8")
    i["PAYMENT_RATIO"] = safe_div(i["AMT_PAYMENT"], i["AMT_INSTALMENT"])
    i["SHORTFALL"] = (i["AMT_INSTALMENT"] - i["AMT_PAYMENT"]).clip(lower=0)
    i["IS_UNDERPAID"] = (i["SHORTFALL"] > 1).astype("int8")
    recent = i["DAYS_INSTALMENT"] >= -365
    i["LATE_RECENT"] = i["IS_LATE"].where(recent)
    i["DAYS_LATE_RECENT"] = i["DAYS_LATE"].where(recent)

    agg = i.groupby("SK_ID_CURR").agg(
        INST_COUNT=("NUM_INSTALMENT_NUMBER", "count"),
        INST_LATE_COUNT=("IS_LATE", "sum"),
        INST_LATE_RATE=("IS_LATE", "mean"),
        INST_LATE_30_COUNT=("IS_LATE_30", "sum"),
        INST_DAYS_LATE_MEAN=("DAYS_LATE", "mean"),
        INST_DAYS_LATE_MAX=("DAYS_LATE", "max"),
        INST_DAYS_EARLY_MEAN=("DAYS_EARLY", "mean"),
        INST_PAYMENT_RATIO_MEAN=("PAYMENT_RATIO", "mean"),
        INST_PAYMENT_RATIO_MIN=("PAYMENT_RATIO", "min"),
        INST_UNDERPAID_RATE=("IS_UNDERPAID", "mean"),
        INST_SHORTFALL_SUM=("SHORTFALL", "sum"),
        INST_LATE_RATE_RECENT=("LATE_RECENT", "mean"),
        INST_DAYS_LATE_RECENT_MAX=("DAYS_LATE_RECENT", "max"),
        INST_AMT_PAYMENT_SUM=("AMT_PAYMENT", "sum"),
    )
    return agg.reset_index()


# ----------------------------------------------------------------------------------------------
# 5. POS / cash loan monthly balances
# ----------------------------------------------------------------------------------------------
def pos_cash_features(pos: pd.DataFrame) -> pd.DataFrame:
    p = clean_monthly_balance(pos).copy()
    p["HAS_DPD"] = (p["SK_DPD"] > 0).astype("int8")
    p["HAS_DPD_DEF"] = (p["SK_DPD_DEF"] > 0).astype("int8")
    p["IS_COMPLETED"] = (p["NAME_CONTRACT_STATUS"] == "Completed").astype("int8")
    # Latest snapshot per previous loan: remaining instalments at application time
    latest = p.sort_values("MONTHS_BALANCE").groupby("SK_ID_PREV").tail(1)
    latest_agg = latest.groupby("SK_ID_CURR").agg(
        POS_REMAINING_INSTALMENTS_SUM=("CNT_INSTALMENT_FUTURE", "sum"),
        POS_ACTIVE_LOANS=("IS_COMPLETED", lambda s: int((s == 0).sum())),
    )
    agg = p.groupby("SK_ID_CURR").agg(
        POS_MONTHS=("MONTHS_BALANCE", "count"),
        POS_LOANS=("SK_ID_PREV", "nunique"),
        POS_DPD_MAX=("SK_DPD", "max"),
        POS_DPD_MEAN=("SK_DPD", "mean"),
        POS_DPD_RATE=("HAS_DPD", "mean"),
        POS_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
        POS_DPD_DEF_RATE=("HAS_DPD_DEF", "mean"),
        POS_COMPLETED_COUNT=("IS_COMPLETED", "sum"),
        POS_INSTALMENT_TERM_MEAN=("CNT_INSTALMENT", "mean"),
    )
    return agg.join(latest_agg).reset_index()


# ----------------------------------------------------------------------------------------------
# 6. Credit card monthly balances: utilisation and payment behaviour
# ----------------------------------------------------------------------------------------------
def credit_card_features(cc: pd.DataFrame) -> pd.DataFrame:
    c = clean_monthly_balance(cc).copy()
    c["UTILIZATION"] = safe_div(c["AMT_BALANCE"], c["AMT_CREDIT_LIMIT_ACTUAL"]).clip(lower=0, upper=5)
    c["PAYMENT_TO_MIN"] = safe_div(c["AMT_PAYMENT_CURRENT"], c["AMT_INST_MIN_REGULARITY"]).clip(upper=50)
    c["MISSED_MIN_PAYMENT"] = ((c["AMT_INST_MIN_REGULARITY"] > 0) & (c["AMT_PAYMENT_CURRENT"].fillna(0) < c["AMT_INST_MIN_REGULARITY"])).astype("int8")
    c["HAS_DPD"] = (c["SK_DPD"] > 0).astype("int8")
    recent = c["MONTHS_BALANCE"] >= -6
    c["UTIL_RECENT"] = c["UTILIZATION"].where(recent)
    c["ATM_SHARE"] = safe_div(c["AMT_DRAWINGS_ATM_CURRENT"], c["AMT_DRAWINGS_CURRENT"])

    agg = c.groupby("SK_ID_CURR").agg(
        CC_MONTHS=("MONTHS_BALANCE", "count"),
        CC_CARDS=("SK_ID_PREV", "nunique"),
        CC_UTILIZATION_MEAN=("UTILIZATION", "mean"),
        CC_UTILIZATION_MAX=("UTILIZATION", "max"),
        CC_UTILIZATION_RECENT=("UTIL_RECENT", "mean"),
        CC_BALANCE_MEAN=("AMT_BALANCE", "mean"),
        CC_LIMIT_MEAN=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        CC_PAYMENT_TO_MIN_MEAN=("PAYMENT_TO_MIN", "mean"),
        CC_MISSED_MIN_RATE=("MISSED_MIN_PAYMENT", "mean"),
        CC_DPD_MAX=("SK_DPD", "max"),
        CC_DPD_RATE=("HAS_DPD", "mean"),
        CC_ATM_SHARE_MEAN=("ATM_SHARE", "mean"),
        CC_DRAWINGS_COUNT_MEAN=("CNT_DRAWINGS_CURRENT", "mean"),
    )
    return agg.reset_index()


# ----------------------------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------------------------
def build_aggregated_features(tables: dict[str, pd.DataFrame], verbose: bool = True) -> pd.DataFrame:
    """All historical-behaviour features, one row per SK_ID_CURR."""
    parts = []
    builders = [
        ("bureau", lambda: bureau_features(tables["bureau"], tables.get("bureau_balance"))),
        ("previous_application", lambda: previous_application_features(tables["previous_application"])),
        ("installments_payments", lambda: installments_features(tables["installments_payments"])),
        ("pos_cash_balance", lambda: pos_cash_features(tables["pos_cash_balance"])),
        ("credit_card_balance", lambda: credit_card_features(tables["credit_card_balance"])),
    ]
    for name, fn in builders:
        if name not in tables:
            continue
        f = fn()
        if verbose:
            print(f"  {name:<24} -> {f.shape[1] - 1:>3} features for {len(f):>9,} clients")
        parts.append(f.set_index("SK_ID_CURR"))
    agg = pd.concat(parts, axis=1, join="outer")
    agg.index.name = "SK_ID_CURR"
    return agg.reset_index()


def add_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features combining the current application with history."""
    df = df.copy()
    if "BUREAU_ACTIVE_DEBT_SUM" in df:
        df["TOTAL_DEBT_TO_INCOME"] = safe_div(df["BUREAU_ACTIVE_DEBT_SUM"].fillna(0) + df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    if "PREV_AMT_ANNUITY_MEAN" in df:
        df["ANNUITY_VS_PREVIOUS"] = safe_div(df["AMT_ANNUITY"], df["PREV_AMT_ANNUITY_MEAN"])
    # Clients with no history at all are a distinct ("thin file") risk segment
    hist_cols = [c for c in ["BUREAU_LOAN_COUNT", "PREV_APP_COUNT"] if c in df]
    if hist_cols:
        df["THIN_FILE"] = df[hist_cols].fillna(0).sum(axis=1).eq(0).astype("int8")
    for col in ["BUREAU_LOAN_COUNT", "PREV_APP_COUNT", "INST_COUNT", "POS_MONTHS", "CC_MONTHS"]:
        if col in df:
            df[col] = df[col].fillna(0)  # no record = zero loans, not unknown
    return df


def assemble_features(app_clean: pd.DataFrame, aggregated: pd.DataFrame) -> pd.DataFrame:
    """Application features + historical aggregates + cross features. Used by notebooks AND the app."""
    app_feats = build_application_features(app_clean)
    merged = app_feats.merge(aggregated, on="SK_ID_CURR", how="left")
    return add_cross_features(merged)
