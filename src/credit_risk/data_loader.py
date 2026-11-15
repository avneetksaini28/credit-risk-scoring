"""Loading the Home Credit tables, memory optimisation, and data dictionaries."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import get_paths, load_config

# Logical table key -> description of grain, used in notebooks and the README.
TABLE_INFO = {
    "application_train": "One row per current loan application (with TARGET). Grain: SK_ID_CURR.",
    "application_test": "One row per current loan application (no TARGET, Kaggle scoring set).",
    "bureau": "Client's previous credits at OTHER institutions reported to the Credit Bureau. Grain: SK_ID_BUREAU.",
    "bureau_balance": "Monthly status of each bureau credit. Grain: SK_ID_BUREAU x MONTHS_BALANCE.",
    "previous_application": "Previous applications to Home Credit. Grain: SK_ID_PREV.",
    "pos_cash_balance": "Monthly snapshots of previous POS / cash loans at Home Credit. Grain: SK_ID_PREV x MONTHS_BALANCE.",
    "installments_payments": "Repayment history of previous Home Credit loans. Grain: one row per payment.",
    "credit_card_balance": "Monthly snapshots of previous Home Credit credit cards. Grain: SK_ID_PREV x MONTHS_BALANCE.",
}

# Parent -> child relationships (parent table, child table, join key, cardinality).
RELATIONSHIPS = [
    ("application_train", "bureau", "SK_ID_CURR", "1 : 0..N"),
    ("bureau", "bureau_balance", "SK_ID_BUREAU", "1 : 0..N"),
    ("application_train", "previous_application", "SK_ID_CURR", "1 : 0..N"),
    ("previous_application", "pos_cash_balance", "SK_ID_PREV", "1 : 0..N"),
    ("previous_application", "installments_payments", "SK_ID_PREV", "1 : 0..N"),
    ("previous_application", "credit_card_balance", "SK_ID_PREV", "1 : 0..N"),
]

# Name used for each table inside HomeCredit_columns_description.csv
_DESCRIPTION_TABLE_NAMES = {
    "application_train": "application_{train|test}.csv",
    "application_test": "application_{train|test}.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "installments_payments": "installments_payments.csv",
    "credit_card_balance": "credit_card_balance.csv",
}


# Low-cardinality text columns in the big tables: reading them as 'category' saves GBs of RAM.
CATEGORY_HINTS = {
    "bureau": ["CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE"],
    "bureau_balance": ["STATUS"],
    "pos_cash_balance": ["NAME_CONTRACT_STATUS"],
    "credit_card_balance": ["NAME_CONTRACT_STATUS"],
    "previous_application": ["NAME_CONTRACT_TYPE", "NAME_CONTRACT_STATUS", "NAME_CASH_LOAN_PURPOSE", "NAME_PAYMENT_TYPE",
                             "CODE_REJECT_REASON", "NAME_CLIENT_TYPE", "NAME_GOODS_CATEGORY", "NAME_PORTFOLIO",
                             "NAME_PRODUCT_TYPE", "CHANNEL_TYPE", "NAME_SELLER_INDUSTRY", "NAME_YIELD_GROUP",
                             "PRODUCT_COMBINATION", "WEEKDAY_APPR_PROCESS_START", "FLAG_LAST_APPL_PER_CONTRACT", "NAME_TYPE_SUITE"],
}


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Downcast numeric columns. Typically cuts memory by 50-70% on these tables."""
    start = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_integer_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            df[col] = s.astype(np.float32)
    end = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Memory: {start:,.1f} MB -> {end:,.1f} MB ({100 * (start - end) / max(start, 1e-9):.0f}% saved)")
    return df


def table_path(name: str, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return get_paths(cfg)["raw_data"] / cfg["data"]["files"][name]


def load_table(
    name: str,
    cfg: dict | None = None,
    nrows: int | None = None,
    usecols: list[str] | None = None,
    reduce_memory: bool = True,
) -> pd.DataFrame:
    """Load one table by its logical key (see config.data.files)."""
    path = table_path(name, cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Kaggle data into data/raw/ (see README) "
            f"or generate the synthetic sample with scripts/make_sample_data.py."
        )
    hints = {c: "category" for c in CATEGORY_HINTS.get(name, []) if usecols is None or c in usecols}
    df = pd.read_csv(path, nrows=nrows, usecols=usecols, dtype=hints or None)
    return reduce_memory_usage(df) if reduce_memory else df


def load_all_tables(cfg: dict | None = None, nrows: int | None = None, include_test: bool = True) -> dict[str, pd.DataFrame]:
    keys = [k for k in TABLE_INFO if include_test or k != "application_test"]
    return {k: load_table(k, cfg, nrows=nrows) for k in keys}


def load_column_descriptions(cfg: dict | None = None) -> pd.DataFrame:
    path = table_path("columns_description", cfg)
    if not path.exists():
        return pd.DataFrame(columns=["Table", "Row", "Description", "Special"])
    df = pd.read_csv(path, encoding="latin1")
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])


def table_overview(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Dimensions, memory and missingness for each table."""
    rows = []
    for name, df in tables.items():
        miss = df.isna().mean()
        rows.append(
            {
                "table": name,
                "rows": len(df),
                "columns": df.shape[1],
                "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 1),
                "cols_with_missing": int((miss > 0).sum()),
                "avg_missing_pct": round(100 * miss.mean(), 1),
                "grain": TABLE_INFO.get(name, ""),
            }
        )
    return pd.DataFrame(rows)


def build_data_dictionary(tables: dict[str, pd.DataFrame], descriptions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Column-level data dictionary: type, missingness, cardinality, example and official description."""
    desc_lookup = {}
    if descriptions is not None and len(descriptions):
        for _, r in descriptions.iterrows():
            desc_lookup[(str(r["Table"]), str(r["Row"]))] = (r.get("Description"), r.get("Special"))

    records = []
    for name, df in tables.items():
        desc_table = _DESCRIPTION_TABLE_NAMES.get(name, f"{name}.csv")
        for col in df.columns:
            s = df[col]
            non_null = s.dropna()
            desc, special = desc_lookup.get((desc_table, col), (None, None))
            records.append(
                {
                    "table": name,
                    "column": col,
                    "dtype": str(s.dtype),
                    "kind": "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical",
                    "missing_pct": round(100 * s.isna().mean(), 2),
                    "n_unique": int(s.nunique(dropna=True)),
                    "example": non_null.iloc[0] if len(non_null) else None,
                    "min": non_null.min() if pd.api.types.is_numeric_dtype(s) and len(non_null) else None,
                    "max": non_null.max() if pd.api.types.is_numeric_dtype(s) and len(non_null) else None,
                    "description": desc,
                    "notes": special,
                }
            )
    return pd.DataFrame(records)


def check_relationships(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Verify join keys: how many child keys match a parent, and how many parents have children."""
    rows = []
    for parent, child, key, card in RELATIONSHIPS:
        if parent not in tables or child not in tables:
            continue
        p_keys = pd.Index(tables[parent][key].unique())
        c_keys = pd.Index(tables[child][key].unique())
        rows.append(
            {
                "parent": parent,
                "child": child,
                "key": key,
                "cardinality": card,
                "child_rows": len(tables[child]),
                "child_keys_matching_parent_pct": round(100 * c_keys.isin(p_keys).mean(), 1),
                "parents_with_children_pct": round(100 * p_keys.isin(c_keys).mean(), 1),
                "avg_children_per_parent": round(len(tables[child]) / max(len(c_keys), 1), 1),
            }
        )
    return pd.DataFrame(rows)
