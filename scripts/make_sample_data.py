"""Generate a small SYNTHETIC dataset with the same schema as the Kaggle Home Credit data.

Purpose: smoke-test the pipeline end to end (CI, a laptop without the 2.5 GB download, or a quick demo).
The numbers mean nothing. Never report results computed on this data.

    python scripts/make_sample_data.py --n-train 6000 --n-test 1500 --out data/sample
    CREDIT_RISK_RAW_DIR=data/sample jupyter lab        # notebooks now read the sample data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ANOM = 365243


def choice(rng, values, n, p=None):
    return rng.choice(np.array(values, dtype=object), size=n, p=p)


def make_applications(rng, n, start_id, with_target):
    ids = np.arange(start_id, start_id + n)
    age_days = rng.uniform(21, 68, n) * 365.25
    income_type = choice(rng, ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"], n, [0.52, 0.23, 0.18, 0.068, 0.002])
    pensioner = np.isin(income_type, ["Pensioner", "Unemployed"])
    days_emp = -np.minimum(rng.gamma(2.0, 900, n), age_days - 18 * 365.25)
    days_emp = np.where(pensioner, ANOM, days_emp)
    income = np.round(np.exp(rng.normal(11.9, 0.5, n)), -2)
    credit = np.round(np.exp(rng.normal(13.1, 0.65, n)), -2)
    goods = np.round(credit * rng.uniform(0.75, 1.0, n), -2)
    annuity = np.round(credit / rng.uniform(10, 40, n), 1)
    ext1 = np.where(rng.random(n) < 0.56, np.nan, rng.beta(4, 4, n))
    ext2 = np.where(rng.random(n) < 0.002, np.nan, rng.beta(5, 3, n))
    ext3 = np.where(rng.random(n) < 0.2, np.nan, rng.beta(4, 3, n))
    edu = choice(rng, ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"], n, [0.71, 0.243, 0.033, 0.0125, 0.0015])
    gender = choice(rng, ["F", "M", "XNA"], n, [0.6579, 0.342, 0.0001])
    df = pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "NAME_CONTRACT_TYPE": choice(rng, ["Cash loans", "Revolving loans"], n, [0.9, 0.1]),
            "CODE_GENDER": gender,
            "FLAG_OWN_CAR": choice(rng, ["N", "Y"], n, [0.66, 0.34]),
            "FLAG_OWN_REALTY": choice(rng, ["Y", "N"], n, [0.69, 0.31]),
            "CNT_CHILDREN": rng.poisson(0.42, n),
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": np.where(rng.random(n) < 0.0001, np.nan, annuity),
            "AMT_GOODS_PRICE": np.where(rng.random(n) < 0.001, np.nan, goods),
            "NAME_TYPE_SUITE": choice(rng, ["Unaccompanied", "Family", "Spouse, partner", "Children", None], n, [0.8, 0.13, 0.037, 0.011, 0.022]),
            "NAME_INCOME_TYPE": income_type,
            "NAME_EDUCATION_TYPE": edu,
            "NAME_FAMILY_STATUS": choice(rng, ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"], n, [0.64, 0.15, 0.1, 0.06, 0.05]),
            "NAME_HOUSING_TYPE": choice(rng, ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment"], n, [0.89, 0.05, 0.036, 0.016, 0.008]),
            "REGION_POPULATION_RELATIVE": rng.uniform(0.001, 0.072, n).round(6),
            "DAYS_BIRTH": -age_days.astype(int),
            "DAYS_EMPLOYED": days_emp.astype(int),
            "DAYS_REGISTRATION": -rng.uniform(0, 15000, n).round(0),
            "DAYS_ID_PUBLISH": -rng.uniform(0, 6500, n).astype(int),
            "OWN_CAR_AGE": np.nan,
            "FLAG_MOBIL": 1,
            "FLAG_EMP_PHONE": (~pensioner).astype(int),
            "FLAG_WORK_PHONE": (rng.random(n) < 0.2).astype(int),
            "FLAG_CONT_MOBILE": (rng.random(n) < 0.998).astype(int),
            "FLAG_PHONE": (rng.random(n) < 0.28).astype(int),
            "FLAG_EMAIL": (rng.random(n) < 0.057).astype(int),
            "OCCUPATION_TYPE": np.where(pensioner, None, choice(rng, ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers", "High skill tech staff", "Accountants", "Medicine staff", None], n)),
            "CNT_FAM_MEMBERS": np.clip(rng.poisson(1.2, n) + 1, 1, 10).astype(float),
            "REGION_RATING_CLIENT": choice(rng, [1, 2, 3], n, [0.1, 0.74, 0.16]),
            "WEEKDAY_APPR_PROCESS_START": choice(rng, ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"], n),
            "HOUR_APPR_PROCESS_START": rng.integers(6, 20, n),
            "REG_CITY_NOT_LIVE_CITY": (rng.random(n) < 0.08).astype(int),
            "REG_CITY_NOT_WORK_CITY": (rng.random(n) < 0.23).astype(int),
            "ORGANIZATION_TYPE": np.where(pensioner, "XNA", choice(rng, ["Business Entity Type 3", "Self-employed", "Other", "Medicine", "Government", "School", "Trade: type 7", "Construction", "Transport: type 4"], n)),
            "EXT_SOURCE_1": ext1,
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3,
            "APARTMENTS_AVG": np.where(rng.random(n) < 0.5, np.nan, rng.beta(2, 15, n)),
            "FLOORSMAX_AVG": np.where(rng.random(n) < 0.5, np.nan, rng.beta(2, 8, n)),
            "TOTALAREA_MODE": np.where(rng.random(n) < 0.48, np.nan, rng.beta(2, 15, n)),
            "OBS_30_CNT_SOCIAL_CIRCLE": rng.poisson(1.4, n).astype(float),
            "DEF_30_CNT_SOCIAL_CIRCLE": rng.poisson(0.14, n).astype(float),
            "OBS_60_CNT_SOCIAL_CIRCLE": rng.poisson(1.4, n).astype(float),
            "DEF_60_CNT_SOCIAL_CIRCLE": rng.poisson(0.1, n).astype(float),
            "DAYS_LAST_PHONE_CHANGE": -rng.gamma(1.5, 650, n).round(0),
        }
    )
    df["REGION_RATING_CLIENT_W_CITY"] = np.clip(df["REGION_RATING_CLIENT"] + rng.choice([-1, 0, 0, 0, 1], n), 1, 3)
    df.loc[df["FLAG_OWN_CAR"] == "Y", "OWN_CAR_AGE"] = rng.integers(0, 30, int((df["FLAG_OWN_CAR"] == "Y").sum()))
    for k in range(2, 22):
        p = 0.71 if k == 3 else (0.08 if k in (6, 8) else 0.015)
        df[f"FLAG_DOCUMENT_{k}"] = (rng.random(n) < p).astype(int)
    for k, lam in [("HOUR", 0.006), ("DAY", 0.007), ("WEEK", 0.034), ("MON", 0.27), ("QRT", 0.26), ("YEAR", 1.9)]:
        df[f"AMT_REQ_CREDIT_BUREAU_{k}"] = np.where(rng.random(n) < 0.135, np.nan, rng.poisson(lam, n))
    # Data-quality issues similar to the real data
    df.loc[df.sample(frac=0.0003, random_state=1).index, "AMT_INCOME_TOTAL"] *= 80  # extreme income outliers
    df.loc[df.sample(frac=0.0002, random_state=2).index, "NAME_FAMILY_STATUS"] = "Unknown"

    # Latent risk drives TARGET and (later) repayment behaviour in the history tables
    e = np.c_[ext1, ext2, ext3]
    ext_mean = np.where(np.isnan(e).all(axis=1), 0.5, np.nanmean(np.where(np.isnan(e), 0, e), axis=1) * 3 / np.maximum((~np.isnan(e)).sum(axis=1), 1))
    z = (
        -2.9
        - 9.0 * (ext_mean - 0.55)
        + 0.6 * (credit / income > 5)
        - 0.012 * (age_days / 365.25 - 40)
        + 0.35 * (edu == "Lower secondary")
        - 0.3 * (edu == "Higher education")
        + 0.25 * (gender == "M")
        + 0.6 * (df["REGION_RATING_CLIENT_W_CITY"].to_numpy() == 3)
        + rng.normal(0, 0.3, n)
    )
    risk = 1 / (1 + np.exp(-z))
    if with_target:
        df.insert(1, "TARGET", (rng.random(n) < risk).astype(int))
    return df, pd.Series(risk, index=ids)


def make_bureau(rng, risk: pd.Series, start_bureau_id: int):
    counts = rng.poisson(4.5, len(risk)) * (rng.random(len(risk)) > 0.14)
    cur = np.repeat(risk.index.to_numpy(), counts)
    r = np.repeat(risk.to_numpy(), counts)
    n = len(cur)
    active = rng.random(n) < 0.37
    days_credit = -rng.integers(1, 2922, n)
    end = days_credit + rng.integers(60, 3000, n)
    credit_sum = np.round(np.exp(rng.normal(12, 1.1, n)), -2)
    debt = np.where(active, credit_sum * rng.uniform(0, 1, n), 0)
    overdue_days = np.where(rng.random(n) < r * 0.2, rng.integers(1, 120, n), 0)
    b = pd.DataFrame(
        {
            "SK_ID_CURR": cur,
            "SK_ID_BUREAU": np.arange(start_bureau_id, start_bureau_id + n),
            "CREDIT_ACTIVE": np.where(active, "Active", np.where(rng.random(n) < 0.004, "Sold", np.where(rng.random(n) < 0.002, "Bad debt", "Closed"))),
            "CREDIT_CURRENCY": "currency 1",
            "DAYS_CREDIT": days_credit,
            "CREDIT_DAY_OVERDUE": overdue_days,
            "DAYS_CREDIT_ENDDATE": np.where(rng.random(n) < 0.06, np.nan, end),
            "DAYS_ENDDATE_FACT": np.where(active, np.nan, np.minimum(end + rng.integers(-200, 200, n), -1)),
            "AMT_CREDIT_MAX_OVERDUE": np.where(rng.random(n) < 0.65, np.nan, rng.exponential(2000, n) * (1 + 10 * r)),
            "CNT_CREDIT_PROLONG": (rng.random(n) < 0.005).astype(int),
            "AMT_CREDIT_SUM": credit_sum,
            "AMT_CREDIT_SUM_DEBT": np.where(rng.random(n) < 0.15, np.nan, debt.round(2)),
            "AMT_CREDIT_SUM_LIMIT": np.where(rng.random(n) < 0.34, np.nan, 0.0),
            "AMT_CREDIT_SUM_OVERDUE": np.where(overdue_days > 0, rng.exponential(5000, n), 0.0).round(2),
            "CREDIT_TYPE": choice(rng, ["Consumer credit", "Credit card", "Car loan", "Mortgage", "Microloan"], n, [0.73, 0.23, 0.017, 0.011, 0.012]),
            "DAYS_CREDIT_UPDATE": np.minimum(days_credit + rng.integers(0, 3000, n), 0),
            "AMT_ANNUITY": np.where(rng.random(n) < 0.7, np.nan, credit_sum / 24),
        }
    )
    # bureau_balance: monthly status history for ~70% of bureau credits
    has_bb = rng.random(n) < 0.7
    months = rng.integers(3, 30, n) * has_bb
    bid = np.repeat(b["SK_ID_BUREAU"].to_numpy(), months)
    rr = np.repeat(r, months)
    mb = -np.concatenate([np.arange(m) for m in months]) if months.sum() else np.array([], dtype=int)
    u = rng.random(len(bid))
    status = np.where(u < 0.25, "C", np.where(u < 0.4, "X", "0"))
    dpd = rng.random(len(bid)) < rr * 0.35
    status = np.where(dpd, choice(rng, ["1", "2", "3", "4", "5"], len(bid), [0.8, 0.1, 0.04, 0.03, 0.03]), status)
    bb = pd.DataFrame({"SK_ID_BUREAU": bid, "MONTHS_BALANCE": mb - 1, "STATUS": status})
    return b, bb


def make_previous(rng, risk: pd.Series, start_prev_id: int):
    counts = rng.poisson(4.0, len(risk)) * (rng.random(len(risk)) > 0.05)
    cur = np.repeat(risk.index.to_numpy(), counts)
    r = np.repeat(risk.to_numpy(), counts)
    n = len(cur)
    status = np.where(rng.random(n) < 0.12 + r, "Refused", choice(rng, ["Approved", "Canceled", "Unused offer"], n, [0.8, 0.18, 0.02]))
    amt_app = np.round(np.exp(rng.normal(11.5, 1.0, n)), -2)
    amt_credit = np.where(status == "Refused", amt_app, amt_app * rng.uniform(0.8, 1.2, n)).round(-2)
    cnt_payment = choice(rng, [6, 10, 12, 18, 24, 36, 48, 60], n).astype(float)
    days_dec = -rng.integers(1, 2922, n)
    first_due = np.where(status == "Approved", days_dec + 30, ANOM)
    p = pd.DataFrame(
        {
            "SK_ID_PREV": np.arange(start_prev_id, start_prev_id + n),
            "SK_ID_CURR": cur,
            "NAME_CONTRACT_TYPE": choice(rng, ["Cash loans", "Consumer loans", "Revolving loans", "XNA"], n, [0.45, 0.43, 0.115, 0.005]),
            "AMT_ANNUITY": np.where(rng.random(n) < 0.22, np.nan, (amt_credit / cnt_payment).round(2)),
            "AMT_APPLICATION": amt_app,
            "AMT_CREDIT": amt_credit,
            "AMT_DOWN_PAYMENT": np.where(rng.random(n) < 0.53, np.nan, (amt_app * rng.uniform(0, 0.3, n)).round(2)),
            "AMT_GOODS_PRICE": np.where(rng.random(n) < 0.23, np.nan, amt_app),
            "WEEKDAY_APPR_PROCESS_START": choice(rng, ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"], n),
            "HOUR_APPR_PROCESS_START": rng.integers(6, 20, n),
            "FLAG_LAST_APPL_PER_CONTRACT": "Y",
            "NFLAG_LAST_APPL_IN_DAY": 1,
            "RATE_DOWN_PAYMENT": np.where(rng.random(n) < 0.53, np.nan, rng.uniform(0, 0.3, n)),
            "NAME_CASH_LOAN_PURPOSE": choice(rng, ["XAP", "XNA", "Repairs", "Other", "Urgent needs"], n, [0.55, 0.4, 0.02, 0.02, 0.01]),
            "NAME_CONTRACT_STATUS": status,
            "DAYS_DECISION": days_dec,
            "NAME_PAYMENT_TYPE": choice(rng, ["Cash through the bank", "XNA", "Non-cash from your account"], n, [0.62, 0.37, 0.01]),
            "CODE_REJECT_REASON": np.where(status == "Refused", choice(rng, ["HC", "LIMIT", "SCO", "SCOFR"], n), "XAP"),
            "NAME_CLIENT_TYPE": choice(rng, ["Repeater", "New", "Refreshed"], n, [0.74, 0.18, 0.08]),
            "NAME_GOODS_CATEGORY": choice(rng, ["XNA", "Mobile", "Consumer Electronics", "Computers", "Audio/Video"], n, [0.57, 0.14, 0.13, 0.09, 0.07]),
            "NAME_PORTFOLIO": choice(rng, ["POS", "Cash", "XNA", "Cards"], n, [0.41, 0.28, 0.22, 0.09]),
            "NAME_PRODUCT_TYPE": choice(rng, ["XNA", "x-sell", "walk-in"], n, [0.64, 0.27, 0.09]),
            "CHANNEL_TYPE": choice(rng, ["Credit and cash offices", "Country-wide", "Stone", "Regional / Local", "Contact center"], n),
            "SELLERPLACE_AREA": rng.integers(-1, 200, n),
            "NAME_SELLER_INDUSTRY": choice(rng, ["XNA", "Consumer electronics", "Connectivity", "Furniture"], n, [0.51, 0.24, 0.17, 0.08]),
            "CNT_PAYMENT": np.where(status == "Approved", cnt_payment, np.nan),
            "NAME_YIELD_GROUP": choice(rng, ["XNA", "middle", "high", "low_normal", "low_action"], n),
            "PRODUCT_COMBINATION": choice(rng, ["Cash", "POS household with interest", "POS mobile with interest", "Card Street"], n),
            "DAYS_FIRST_DRAWING": np.where(status == "Approved", ANOM, np.nan),
            "DAYS_FIRST_DUE": np.where(status == "Approved", first_due, np.nan),
            "DAYS_LAST_DUE_1ST_VERSION": np.where(status == "Approved", first_due + cnt_payment * 30, np.nan),
            "DAYS_LAST_DUE": np.where(status == "Approved", np.where(rng.random(n) < 0.5, ANOM, first_due + cnt_payment * 30), np.nan),
            "DAYS_TERMINATION": np.where(status == "Approved", np.where(rng.random(n) < 0.55, ANOM, first_due + cnt_payment * 30 + 5), np.nan),
            "NFLAG_INSURED_ON_APPROVAL": np.where(status == "Approved", (rng.random(n) < 0.33).astype(float), np.nan),
        }
    )
    return p, pd.Series(r, index=p["SK_ID_PREV"])


def make_behaviour(rng, prev: pd.DataFrame, prev_risk: pd.Series):
    """installments_payments, POS_CASH_balance and credit_card_balance for approved previous loans."""
    appr = prev[prev["NAME_CONTRACT_STATUS"] == "Approved"].copy()
    appr["risk"] = prev_risk.reindex(appr["SK_ID_PREV"]).to_numpy()
    # Instalments: up to 12 most recent instalments of each approved loan
    n_inst = np.minimum(appr["CNT_PAYMENT"].fillna(12).astype(int).to_numpy(), 12)
    idx = np.repeat(np.arange(len(appr)), n_inst)
    num = np.concatenate([np.arange(1, k + 1) for k in n_inst])
    a = appr.iloc[idx]
    days_inst = np.minimum(a["DAYS_DECISION"].to_numpy() + 30 * num, -1)
    late = rng.random(len(idx)) < (0.03 + a["risk"].to_numpy() * 0.8)
    delay = np.where(late, rng.exponential(12, len(idx)) + 1, -rng.exponential(8, len(idx)))
    amt_inst = (a["AMT_CREDIT"].to_numpy() / np.maximum(a["CNT_PAYMENT"].fillna(12).to_numpy(), 1)).round(2)
    underpay = rng.random(len(idx)) < a["risk"].to_numpy() * 0.3
    inst = pd.DataFrame(
        {
            "SK_ID_PREV": a["SK_ID_PREV"].to_numpy(),
            "SK_ID_CURR": a["SK_ID_CURR"].to_numpy(),
            "NUM_INSTALMENT_VERSION": 1.0,
            "NUM_INSTALMENT_NUMBER": num,
            "DAYS_INSTALMENT": days_inst.astype(float),
            "DAYS_ENTRY_PAYMENT": np.where(rng.random(len(idx)) < 0.0002, np.nan, np.minimum(days_inst + np.round(delay), -1)),
            "AMT_INSTALMENT": amt_inst,
            "AMT_PAYMENT": np.where(underpay, amt_inst * rng.uniform(0.2, 0.9, len(idx)), amt_inst).round(2),
        }
    )
    # POS_CASH_balance: monthly snapshots for POS / cash loans
    pos_loans = appr[appr["NAME_CONTRACT_TYPE"] != "Revolving loans"]
    months = np.minimum(pos_loans["CNT_PAYMENT"].fillna(12).astype(int).to_numpy(), 18)
    pidx = np.repeat(np.arange(len(pos_loans)), months)
    mnum = np.concatenate([np.arange(k) for k in months])
    pl = pos_loans.iloc[pidx]
    dpd_flag = rng.random(len(pidx)) < pl["risk"].to_numpy() * 0.25
    pos = pd.DataFrame(
        {
            "SK_ID_PREV": pl["SK_ID_PREV"].to_numpy(),
            "SK_ID_CURR": pl["SK_ID_CURR"].to_numpy(),
            "MONTHS_BALANCE": -(mnum + 1),
            "CNT_INSTALMENT": pl["CNT_PAYMENT"].to_numpy(),
            "CNT_INSTALMENT_FUTURE": np.maximum(pl["CNT_PAYMENT"].to_numpy() - (months[pidx] - mnum), 0),
            "NAME_CONTRACT_STATUS": np.where(mnum == 0, choice(rng, ["Active", "Completed"], len(pidx), [0.6, 0.4]), "Active"),
            "SK_DPD": np.where(dpd_flag, rng.integers(1, 60, len(pidx)), 0),
            "SK_DPD_DEF": np.where(dpd_flag & (rng.random(len(pidx)) < 0.3), rng.integers(1, 30, len(pidx)), 0),
        }
    )
    # credit_card_balance for revolving loans
    cards = appr[appr["NAME_CONTRACT_TYPE"] == "Revolving loans"]
    cm = rng.integers(6, 30, len(cards))
    cidx = np.repeat(np.arange(len(cards)), cm)
    cmn = np.concatenate([np.arange(k) for k in cm]) if len(cm) else np.array([], dtype=int)
    cl = cards.iloc[cidx]
    limit = np.round(cl["AMT_CREDIT"].to_numpy().clip(20000) * 1.0, -3)
    util = np.clip(rng.beta(1.5, 3, len(cidx)) + cl["risk"].to_numpy() * 0.8, 0, 1.2)
    bal = (limit * util).round(2)
    minpay = (bal * 0.05).round(2)
    pay = np.where(rng.random(len(cidx)) < cl["risk"].to_numpy() * 0.5, minpay * rng.uniform(0, 0.9, len(cidx)), minpay * rng.uniform(1, 4, len(cidx))).round(2)
    draw = (bal * rng.uniform(0, 0.3, len(cidx))).round(2)
    cc = pd.DataFrame(
        {
            "SK_ID_PREV": cl["SK_ID_PREV"].to_numpy(),
            "SK_ID_CURR": cl["SK_ID_CURR"].to_numpy(),
            "MONTHS_BALANCE": -(cmn + 1),
            "AMT_BALANCE": bal,
            "AMT_CREDIT_LIMIT_ACTUAL": limit,
            "AMT_DRAWINGS_ATM_CURRENT": (draw * rng.uniform(0, 1, len(cidx))).round(2),
            "AMT_DRAWINGS_CURRENT": draw,
            "AMT_DRAWINGS_OTHER_CURRENT": 0.0,
            "AMT_DRAWINGS_POS_CURRENT": (draw * 0.5).round(2),
            "AMT_INST_MIN_REGULARITY": minpay,
            "AMT_PAYMENT_CURRENT": pay,
            "AMT_PAYMENT_TOTAL_CURRENT": pay,
            "AMT_RECEIVABLE_PRINCIPAL": (bal * 0.95).round(2),
            "AMT_RECIVABLE": bal,
            "AMT_TOTAL_RECEIVABLE": bal,
            "CNT_DRAWINGS_ATM_CURRENT": rng.poisson(0.5, len(cidx)).astype(float),
            "CNT_DRAWINGS_CURRENT": rng.poisson(2, len(cidx)),
            "CNT_DRAWINGS_OTHER_CURRENT": 0.0,
            "CNT_DRAWINGS_POS_CURRENT": rng.poisson(1.5, len(cidx)).astype(float),
            "CNT_INSTALMENT_MATURE_CUM": cmn.astype(float),
            "NAME_CONTRACT_STATUS": "Active",
            "SK_DPD": np.where(rng.random(len(cidx)) < cl["risk"].to_numpy() * 0.2, rng.integers(1, 90, len(cidx)), 0),
            "SK_DPD_DEF": 0,
        }
    )
    return inst, pos, cc


def column_descriptions(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    name_map = {"application_train": "application_{train|test}.csv", "POS_CASH_balance": "POS_CASH_balance.csv"}
    for t, df in tables.items():
        if t == "application_test":
            continue
        for c in df.columns:
            rows.append({"Table": name_map.get(t, f"{t}.csv"), "Row": c, "Description": f"[synthetic] {c.replace('_', ' ').lower()}", "Special": ""})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--out", default="data/sample")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train, risk_tr = make_applications(rng, args.n_train, 100002, with_target=True)
    test, risk_te = make_applications(rng, args.n_test, 100002 + args.n_train + 10, with_target=False)
    risk = pd.concat([risk_tr, risk_te])
    bureau, bb = make_bureau(rng, risk, 5000000)
    prev, prev_risk = make_previous(rng, risk, 1000000)
    inst, pos, cc = make_behaviour(rng, prev, prev_risk)

    tables = {
        "application_train": train,
        "application_test": test,
        "bureau": bureau,
        "bureau_balance": bb,
        "previous_application": prev,
        "installments_payments": inst,
        "POS_CASH_balance": pos,
        "credit_card_balance": cc,
    }
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name:<24} {df.shape}")
    column_descriptions(tables).to_csv(out / "HomeCredit_columns_description.csv", encoding="latin1")
    print(f"Synthetic default rate: {train['TARGET'].mean():.2%}  ->  written to {out.resolve()}")


if __name__ == "__main__":
    main()
