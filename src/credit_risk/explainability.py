"""SHAP explanations and adverse-action style reason codes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .modeling import CreditRiskModel

# Plain-English labels for the features most likely to appear in reason codes.
FEATURE_LABELS = {
    "EXT_SOURCE_MEAN": "External credit score (average)",
    "EXT_SOURCE_MIN": "External credit score (lowest)",
    "EXT_SOURCE_MAX": "External credit score (highest)",
    "EXT_SOURCE_PROD": "External credit scores (combined)",
    "EXT_SOURCE_1": "External credit score 1",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    "EXT_SOURCE_STD": "Disagreement between external scores",
    "CREDIT_TO_ANNUITY_RATIO": "Loan term (credit / annuity)",
    "CREDIT_TO_GOODS_RATIO": "Loan amount relative to goods price",
    "ANNUITY_TO_INCOME_RATIO": "Repayment burden relative to income",
    "CREDIT_TO_INCOME_RATIO": "Loan amount relative to income",
    "TOTAL_DEBT_TO_INCOME": "Total debt relative to income",
    "AGE_YEARS": "Applicant age",
    "YEARS_EMPLOYED": "Time in current employment",
    "EMPLOYED_TO_AGE_RATIO": "Share of adult life employed",
    "YEARS_ID_PUBLISH": "Time since ID document issued",
    "YEARS_REGISTRATION": "Time since registration change",
    "YEARS_SINCE_PHONE_CHANGE": "Time since phone number change",
    "AMT_CREDIT": "Requested loan amount",
    "AMT_ANNUITY": "Loan annuity (payment)",
    "AMT_GOODS_PRICE": "Price of goods financed",
    "AMT_INCOME_TOTAL": "Reported income",
    "INST_LATE_RATE": "Share of past instalments paid late",
    "INST_DAYS_LATE_MAX": "Worst past payment delay (days)",
    "INST_DAYS_LATE_MEAN": "Average past payment delay (days)",
    "INST_PAYMENT_RATIO_MIN": "Lowest share of an instalment paid",
    "INST_UNDERPAID_RATE": "Share of instalments underpaid",
    "INST_LATE_RATE_RECENT": "Late payments in the last 12 months",
    "INST_DAYS_EARLY_MEAN": "Average days paid early",
    "BUREAU_ACTIVE_DEBT_TO_CREDIT": "Outstanding debt on active credit lines",
    "BUREAU_CARD_UTILIZATION": "Credit card utilisation (bureau)",
    "BUREAU_CREDIT_HISTORY_YEARS": "Length of credit history",
    "BUREAU_YEARS_SINCE_LAST_CREDIT": "Time since most recent new credit",
    "BUREAU_IS_ACTIVE_SUM": "Number of active credit lines",
    "BUREAU_IS_ACTIVE_MEAN": "Share of credit lines still active",
    "BUREAU_OPENED_LAST_YEAR_SUM": "Credit lines opened in the last year",
    "BUREAU_LOAN_COUNT": "Number of bureau credit records",
    "BUREAU_CREDIT_DAY_OVERDUE_MAX": "Current days overdue on bureau credit",
    "BUREAU_AMT_CREDIT_MAX_OVERDUE_MAX": "Largest overdue amount on bureau credit",
    "BUREAU_BB_DPD_RATE_MEAN": "Delinquent months on bureau credit",
    "BUREAU_BB_DPD_RATE_MAX": "Worst delinquency rate on a bureau credit",
    "BUREAU_BB_RECENT_DPD_MEAN": "Recent bureau delinquencies (12 months)",
    "BUREAU_BB_RECENT_DPD_MAX": "Recent bureau delinquencies (12 months, worst line)",
    "BUREAU_BB_MAX_DPD_BUCKET_MEAN": "Typical worst bureau arrears bucket",
    "BUREAU_BB_MAX_DPD_BUCKET_MAX": "Worst bureau arrears bucket ever",
    "BUREAU_BB_DPD60_COUNT_SUM": "Months 60+ days past due (bureau)",
    "PREV_REFUSAL_RATE": "Share of previous applications refused",
    "PREV_REFUSED_COUNT": "Number of previous refusals",
    "PREV_RECENT_REFUSED": "Refusals in the last 12 months",
    "PREV_APPROVAL_RATE": "Share of previous applications approved",
    "PREV_CREDIT_TO_APPLICATION_MEAN": "Previously granted vs requested amount",
    "PREV_CNT_PAYMENT_MEAN": "Average term of previous loans",
    "POS_DPD_RATE": "Delinquent months on previous POS loans",
    "POS_REMAINING_INSTALMENTS_SUM": "Remaining instalments on existing loans",
    "CC_UTILIZATION_MEAN": "Average credit card utilisation",
    "CC_UTILIZATION_RECENT": "Recent credit card utilisation",
    "CC_MISSED_MIN_RATE": "Missed minimum card payments",
    "CC_DPD_RATE": "Delinquent months on credit cards",
    "REGION_RATING_CLIENT_W_CITY": "Region risk rating",
    "REGION_RATING_CLIENT": "Region risk rating",
    "NAME_EDUCATION_TYPE": "Education level",
    "NAME_INCOME_TYPE": "Income type",
    "OCCUPATION_TYPE": "Occupation",
    "ORGANIZATION_TYPE": "Employer type",
    "NAME_CONTRACT_TYPE": "Loan product type",
    "FLAG_OWN_CAR": "Car ownership",
    "OWN_CAR_AGE": "Age of owned car",
    "DOCUMENT_COUNT": "Number of documents provided",
    "THIN_FILE": "No prior credit history",
    "DOWN_PAYMENT_PROXY": "Down payment (goods price minus loan)",
    "ANNUITY_VS_PREVIOUS": "Annuity vs previous loans",
    "POS_MONTHS": "Months of POS loan history",
    "POS_LOANS": "Number of previous POS loans",
    "INST_COUNT": "Number of past instalments",
    "INST_AMT_PAYMENT_SUM": "Total past instalment payments",
    "PREV_AMT_ANNUITY_MEAN": "Average annuity of previous loans",
    "BUREAU_AMT_CREDIT_SUM_MEAN": "Average bureau credit amount",
    "BUREAU_DAYS_CREDIT_MEAN": "Average age of bureau credits",
    "BUREAU_DAYS_CREDIT_ENDDATE_MAX": "Latest planned bureau credit end date",
    "INCOME_PER_FAMILY_MEMBER": "Income per family member",
    "REGION_POPULATION_RELATIVE": "Region population density",
    "DAYS_ID_PUBLISH": "Time since ID document issued",
    "FLAG_DAYS_EMPLOYED_ANOM": "Not in employment (pensioner / unemployed)",
}


def label(feature: str) -> str:
    base = feature.replace("missingindicator_", "")
    text = FEATURE_LABELS.get(base, base.replace("_", " ").title())
    return f"{text} (missing)" if feature.startswith("missingindicator_") else text


def get_explainer(model: CreditRiskModel, background: pd.DataFrame | None = None):
    """TreeExplainer for tree models (exact, fast); LinearExplainer for logistic regression."""
    est = model.estimator
    if model.name == "logistic_regression":
        bg = model.transform(background) if background is not None else None
        return shap.LinearExplainer(est, bg)
    return shap.TreeExplainer(est)


def shap_values(model: CreditRiskModel, X: pd.DataFrame, explainer=None) -> shap.Explanation:
    """SHAP values in log-odds of the (uncalibrated) model. Platt calibration is monotonic,
    so the direction and ranking of each feature's effect is the same for the calibrated PD."""
    explainer = explainer or get_explainer(model, X)
    Xt = model.transform(X)
    exp = explainer(Xt)
    if exp.values.ndim == 3:  # some model types return one set per class
        exp = exp[:, :, 1]
    return exp


def global_importance(exp: shap.Explanation) -> pd.DataFrame:
    vals = np.abs(exp.values).mean(axis=0)
    signed = exp.values.mean(axis=0)
    df = pd.DataFrame({"feature": exp.feature_names, "mean_abs_shap": vals, "mean_shap": signed})
    df["label"] = df["feature"].map(label)
    df["share_pct"] = 100 * df["mean_abs_shap"] / df["mean_abs_shap"].sum()
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def reason_codes(exp_row: shap.Explanation, top_n: int = 4, raw_values: pd.Series | None = None) -> pd.DataFrame:
    """Top factors pushing ONE applicant's risk up (adverse-action reasons) and down."""
    df = pd.DataFrame({"feature": exp_row.feature_names, "shap": exp_row.values, "model_value": exp_row.data})
    df["label"] = df["feature"].map(label)
    if raw_values is not None:
        df["value"] = df["feature"].map(lambda f: raw_values.get(f.replace("missingindicator_", ""), np.nan))
    df["direction"] = np.where(df["shap"] > 0, "Increases risk", "Decreases risk")
    up = df[df["shap"] > 0].nlargest(top_n, "shap")
    down = df[df["shap"] < 0].nsmallest(top_n, "shap")
    return pd.concat([up, down]).reset_index(drop=True)
