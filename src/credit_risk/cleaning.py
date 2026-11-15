"""Data cleaning and data-quality checks.

Only deterministic, row-level fixes live here (anomaly codes, impossible values, types).
Anything learned from the data (imputation values, outlier caps, encoders) is fitted inside the
sklearn pipeline on the training split only, so the test split can't leak into it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_ANOMALY = 365243

PREVIOUS_DAYS_COLS = [
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    """Non-numeric, non-boolean columns (works for object, pandas 'str' and category dtypes)."""
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]


# ----------------------------------------------------------------------------------------------
# Table-specific cleaning
# ----------------------------------------------------------------------------------------------
def clean_application(df: pd.DataFrame, anomaly: int = DAYS_ANOMALY) -> pd.DataFrame:
    """Clean application_train / application_test."""
    df = df.copy()

    # 1. DAYS_EMPLOYED = 365243 is a placeholder (~18% of rows, almost all pensioners / unemployed).
    #    Keep the information as a flag, then set the value to missing.
    if "DAYS_EMPLOYED" in df:
        df["FLAG_DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == anomaly).astype("int8")
        df.loc[df["DAYS_EMPLOYED"] == anomaly, "DAYS_EMPLOYED"] = np.nan

    # 2. Placeholder categories that really mean "unknown".
    if "CODE_GENDER" in df:
        df["CODE_GENDER"] = df["CODE_GENDER"].mask(df["CODE_GENDER"] == "XNA")
    if "NAME_FAMILY_STATUS" in df:
        df["NAME_FAMILY_STATUS"] = df["NAME_FAMILY_STATUS"].mask(df["NAME_FAMILY_STATUS"] == "Unknown")
    if "ORGANIZATION_TYPE" in df:
        # 'XNA' here = no employer (pensioners); keep it as an explicit, readable category.
        df["ORGANIZATION_TYPE"] = df["ORGANIZATION_TYPE"].astype(object).where(df["ORGANIZATION_TYPE"] != "XNA", "Not employed")

    # 3. Binary Y/N flags -> 0/1 integers.
    for col in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        if col in df and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map({"Y": 1, "N": 0}).astype("float32")

    # 4. Impossible values -> missing.
    for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
        if col in df:
            df.loc[df[col] <= 0, col] = np.nan
    if "DAYS_LAST_PHONE_CHANGE" in df:
        df.loc[df["DAYS_LAST_PHONE_CHANGE"] > 0, "DAYS_LAST_PHONE_CHANGE"] = np.nan

    # 5. Types.
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype("int64")
    if "TARGET" in df:
        df["TARGET"] = df["TARGET"].astype("int8")
    return df


def clean_bureau(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Event dates must be in the past relative to the application (DAYS <= 0).
    for col in ["DAYS_CREDIT", "DAYS_ENDDATE_FACT", "DAYS_CREDIT_UPDATE"]:
        if col in df:
            df.loc[df[col] > 0, col] = np.nan
    # Planned end dates more than ~50 years away are data errors.
    if "DAYS_CREDIT_ENDDATE" in df:
        df.loc[df["DAYS_CREDIT_ENDDATE"].abs() > 365 * 50, "DAYS_CREDIT_ENDDATE"] = np.nan
    for col in ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE"]:
        if col in df:
            df.loc[df[col] < 0, col] = 0
    return df


def clean_previous_application(df: pd.DataFrame, anomaly: int = DAYS_ANOMALY) -> pd.DataFrame:
    df = df.copy()
    for col in PREVIOUS_DAYS_COLS:
        if col in df:
            df.loc[df[col] == anomaly, col] = np.nan
    # XNA / XAP are Home Credit's "not available" / "not applicable" codes.
    for col in categorical_columns(df):
        df[col] = df[col].mask(df[col] == "XNA")  # mask() works for both object and category dtypes
    for col in ["AMT_APPLICATION", "AMT_CREDIT"]:
        if col in df:
            df.loc[df[col] < 0, col] = np.nan
    return df


def clean_installments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # A payment recorded after the application date would be future information: drop it.
    if "DAYS_ENTRY_PAYMENT" in df:
        df = df[~(df["DAYS_ENTRY_PAYMENT"] > 0)]
    return df


def clean_monthly_balance(df: pd.DataFrame) -> pd.DataFrame:
    """POS_CASH_balance / credit_card_balance / bureau_balance: keep only months before application."""
    if "MONTHS_BALANCE" in df:
        df = df[df["MONTHS_BALANCE"] <= 0]
    return df


# ----------------------------------------------------------------------------------------------
# Data-quality reporting
# ----------------------------------------------------------------------------------------------
def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    out = pd.DataFrame(
        {
            "missing_count": miss,
            "missing_pct": (100 * miss / len(df)).round(2),
            "dtype": df.dtypes.astype(str),
        }
    )
    return out[out["missing_count"] > 0].sort_values("missing_pct", ascending=False)


def missing_vs_target(df: pd.DataFrame, target: str = "TARGET", min_missing_pct: float = 1.0) -> pd.DataFrame:
    """Is missingness itself informative? Default rate when a value is missing vs present."""
    rows = []
    for col in df.columns:
        if col == target:
            continue
        m = df[col].isna()
        pct = 100 * m.mean()
        if pct < min_missing_pct or m.all():
            continue
        rows.append(
            {
                "column": col,
                "missing_pct": round(pct, 1),
                "default_rate_missing": round(100 * df.loc[m, target].mean(), 2),
                "default_rate_present": round(100 * df.loc[~m, target].mean(), 2),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out["lift_when_missing"] = (out["default_rate_missing"] / out["default_rate_present"]).round(2)
        out = out.sort_values("lift_when_missing", ascending=False)
    return out


def duplicate_report(tables: dict[str, pd.DataFrame], keys: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for name, df in tables.items():
        key = keys.get(name)
        rows.append(
            {
                "table": name,
                "rows": len(df),
                "exact_duplicate_rows": int(df.duplicated().sum()),
                "primary_key": ", ".join(key) if key else "-",
                "duplicate_keys": int(df.duplicated(subset=key).sum()) if key else None,
            }
        )
    return pd.DataFrame(rows)


def outlier_report(df: pd.DataFrame, cols: list[str], iqr_mult: float = 3.0) -> pd.DataFrame:
    """IQR-based outlier counts (3 x IQR = 'extreme' outliers) plus percentile context."""
    rows = []
    for col in cols:
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append(
            {
                "column": col,
                "p1": s.quantile(0.01),
                "median": s.median(),
                "p99": s.quantile(0.99),
                "max": s.max(),
                "lower_fence": lo,
                "upper_fence": hi,
                "outliers": n_out,
                "outlier_pct": round(100 * n_out / len(s), 2),
            }
        )
    return pd.DataFrame(rows)


def validate_application(df: pd.DataFrame) -> pd.DataFrame:
    """Business-rule validation checks. Every check should pass after cleaning."""
    checks = []

    def add(name, passed, detail=""):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    add("SK_ID_CURR is unique", df["SK_ID_CURR"].is_unique, f"{df['SK_ID_CURR'].duplicated().sum()} duplicates")
    if "TARGET" in df:
        add("TARGET is binary 0/1", set(df["TARGET"].dropna().unique()) <= {0, 1}, str(sorted(df["TARGET"].unique())))
    for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
        if col in df:
            n_bad = int((df[col] <= 0).sum())
            add(f"{col} > 0", n_bad == 0, f"{n_bad} non-positive")
    if "DAYS_BIRTH" in df:
        age = -df["DAYS_BIRTH"] / 365.25
        add("Applicant age between 18 and 75", age.between(18, 75).all(), f"min {age.min():.1f}, max {age.max():.1f}")
    if "DAYS_EMPLOYED" in df:
        n_bad = int((df["DAYS_EMPLOYED"] > 0).sum())
        add("DAYS_EMPLOYED <= 0 (anomaly removed)", n_bad == 0, f"{n_bad} positive values")
    if {"AMT_CREDIT", "AMT_ANNUITY"} <= set(df.columns):
        n_bad = int((df["AMT_ANNUITY"] > df["AMT_CREDIT"]).sum())
        add("Annuity does not exceed credit amount", n_bad == 0, f"{n_bad} rows")
    for col in [c for c in df.columns if c.startswith("EXT_SOURCE_")]:
        s = df[col].dropna()
        add(f"{col} within [0, 1]", s.between(0, 1).all(), f"min {s.min():.3f}, max {s.max():.3f}")
    return pd.DataFrame(checks)
