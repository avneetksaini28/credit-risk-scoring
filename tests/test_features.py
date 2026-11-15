import numpy as np
import pandas as pd

from credit_risk.cleaning import clean_application
from credit_risk.features import build_application_features, installments_features, safe_div


def _app(**over):
    row = dict(SK_ID_CURR=1, TARGET=0, DAYS_BIRTH=-365.25 * 40, DAYS_EMPLOYED=-365.25 * 5, DAYS_REGISTRATION=-1000,
               DAYS_ID_PUBLISH=-2000, DAYS_LAST_PHONE_CHANGE=-100, AMT_INCOME_TOTAL=100_000.0, AMT_CREDIT=300_000.0,
               AMT_ANNUITY=15_000.0, AMT_GOODS_PRICE=250_000.0, CNT_FAM_MEMBERS=2.0, EXT_SOURCE_1=0.2, EXT_SOURCE_2=0.4,
               EXT_SOURCE_3=np.nan, CODE_GENDER="F", FLAG_OWN_CAR="N", FLAG_OWN_REALTY="Y", NAME_FAMILY_STATUS="Married",
               ORGANIZATION_TYPE="School")
    row.update(over)
    return pd.DataFrame([row])


def test_safe_div_handles_zero_and_nan():
    out = safe_div(pd.Series([1.0, 1.0, np.nan]), pd.Series([0.0, 2.0, 1.0]))
    assert np.isnan(out[0]) and out[1] == 0.5 and np.isnan(out[2])


def test_days_employed_anomaly_is_flagged_and_removed():
    df = clean_application(_app(DAYS_EMPLOYED=365243, CODE_GENDER="XNA"))
    assert df.loc[0, "FLAG_DAYS_EMPLOYED_ANOM"] == 1
    assert np.isnan(df.loc[0, "DAYS_EMPLOYED"])
    assert pd.isna(df.loc[0, "CODE_GENDER"])


def test_application_ratios_and_ext_sources():
    f = build_application_features(clean_application(_app()))
    r = f.iloc[0]
    assert r["CREDIT_TO_INCOME_RATIO"] == 3.0
    assert r["ANNUITY_TO_INCOME_RATIO"] == 0.15
    assert r["CREDIT_TO_ANNUITY_RATIO"] == 20.0
    assert abs(r["AGE_YEARS"] - 40) < 1e-9 and r["AGE_GROUP"] == "35-44"
    assert abs(r["EXT_SOURCE_MEAN"] - 0.3) < 1e-9 and r["EXT_SOURCE_NAN_COUNT"] == 1
    assert "DAYS_BIRTH" not in f.columns  # replaced by AGE_YEARS


def test_installments_partial_payments_collapse_to_one_instalment():
    inst = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 1], "SK_ID_PREV": [10, 10, 10], "NUM_INSTALMENT_VERSION": [1, 1, 1],
        "NUM_INSTALMENT_NUMBER": [1, 1, 2], "DAYS_INSTALMENT": [-60, -60, -30],
        "DAYS_ENTRY_PAYMENT": [-62, -50, -35], "AMT_INSTALMENT": [100.0, 100.0, 100.0], "AMT_PAYMENT": [40.0, 60.0, 100.0],
    })
    out = installments_features(inst).iloc[0]
    assert out["INST_COUNT"] == 2                  # two instalments, not three payments
    assert out["INST_LATE_COUNT"] == 1             # instalment 1 fully paid 10 days late
    assert out["INST_DAYS_LATE_MAX"] == 10
    assert out["INST_UNDERPAID_RATE"] == 0


def test_future_payments_are_dropped():
    inst = pd.DataFrame({
        "SK_ID_CURR": [1, 1], "SK_ID_PREV": [10, 10], "NUM_INSTALMENT_VERSION": [1, 1], "NUM_INSTALMENT_NUMBER": [1, 2],
        "DAYS_INSTALMENT": [-30, -1], "DAYS_ENTRY_PAYMENT": [-30, 5], "AMT_INSTALMENT": [100.0, 100.0], "AMT_PAYMENT": [100.0, 100.0],
    })
    assert installments_features(inst).iloc[0]["INST_COUNT"] == 1
