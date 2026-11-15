import numpy as np
import pandas as pd

from credit_risk.preprocessing import Winsorizer, split_data
from credit_risk.thresholds import assign_decision, assign_grade, break_even_pd, make_policy, threshold_sweep

GRADES = {"A": 0.03, "B": 0.06, "C": 0.10, "D": 0.20, "E": 1.00}


def test_break_even():
    assert abs(break_even_pd(0.6, 0.15) - 0.2) < 1e-12


def test_sweep_accounting():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.30, 0.10, 0.50])
    ead = np.array([100.0, 100.0, 100.0, 100.0])
    s = threshold_sweep(y, p, ead, lgd=0.5, margin=0.1, thresholds=np.array([0.2])).iloc[0]
    # approve p < 0.2 -> rows 0 (good) and 2 (bad)
    assert s["approval_rate"] == 0.5
    assert s["false_approvals"] == 1 and s["false_rejections"] == 1
    assert s["loss_false_approvals"] == 50.0 and s["lost_profit_false_rejections"] == 10.0
    assert s["net_profit"] == 10.0 - 50.0


def test_policy_bands_and_grades():
    pol = make_policy(0.2, 0.7, 0.6, 0.15, GRADES)
    assert list(assign_decision([0.05, 0.15, 0.25], pol)) == ["Approve", "Refer to underwriter", "Decline"]
    assert list(assign_grade([0.01, 0.03, 0.07, 0.5, 0.99], GRADES)) == ["A", "B", "C", "E", "E"]


def test_split_is_stratified_and_disjoint():
    df = pd.DataFrame({"SK_ID_CURR": range(2000), "TARGET": [1 if i % 10 == 0 else 0 for i in range(2000)]})
    tr, va, te = split_data(df, "TARGET", 0.15, 0.15, 0)
    assert len(tr) + len(va) + len(te) == 2000
    assert not set(tr.SK_ID_CURR) & set(te.SK_ID_CURR)
    for part in (tr, va, te):
        assert abs(part.TARGET.mean() - 0.1) < 0.01


def test_winsorizer_leaves_binary_columns_alone():
    X = pd.DataFrame({"x": np.r_[np.arange(999.0), 1e9], "flag": np.r_[np.zeros(998), [1.0, 1.0]]})
    out = Winsorizer().fit(X).transform(X)
    assert out["x"].max() < 1e9
    assert out["flag"].sum() == 2
