"""SQLite helpers: build the database from the raw CSVs and run named SQL queries."""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

import pandas as pd

from .config import get_paths, load_config
from .data_loader import table_path

# Indexes that make the joins in sql/02_analysis_queries.sql fast on the full data.
INDEXES = {
    "application_train": ["SK_ID_CURR"],
    "application_test": ["SK_ID_CURR"],
    "bureau": ["SK_ID_CURR", "SK_ID_BUREAU"],
    "bureau_balance": ["SK_ID_BUREAU"],
    "previous_application": ["SK_ID_CURR", "SK_ID_PREV"],
    "pos_cash_balance": ["SK_ID_CURR", "SK_ID_PREV"],
    "installments_payments": ["SK_ID_CURR", "SK_ID_PREV"],
    "credit_card_balance": ["SK_ID_CURR", "SK_ID_PREV"],
}


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    db_path = Path(db_path) if db_path else get_paths()["database"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA cache_size = -200000")  # ~200 MB page cache
    return conn


def load_csv_to_sqlite(conn: sqlite3.Connection, table: str, csv_path: Path, chunksize: int = 250_000) -> int:
    """Stream a CSV into SQLite in chunks so even the 27M-row tables fit in memory."""
    total = 0
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
        chunk.to_sql(table, conn, if_exists="replace" if i == 0 else "append", index=False)
        total += len(chunk)
    for col in INDEXES.get(table, []):
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{col} ON {table} ({col})")
    conn.commit()
    return total


def build_database(cfg: dict | None = None, tables: list[str] | None = None, verbose: bool = True) -> pd.DataFrame:
    """Create (or refresh) every table in the SQLite database. Returns a load summary."""
    cfg = cfg or load_config()
    skip = set(cfg["sql"].get("skip_tables") or [])
    tables = tables or [t for t in INDEXES if t not in skip]
    conn = get_connection(get_paths(cfg)["database"])
    summary = []
    try:
        for t in tables:
            path = table_path(t, cfg)
            if not path.exists():
                if verbose:
                    print(f"  skipped {t}: {path.name} not found")
                continue
            start = time.time()
            n = load_csv_to_sqlite(conn, t, path, cfg["sql"]["chunksize"])
            secs = time.time() - start
            summary.append({"table": t, "rows_loaded": n, "seconds": round(secs, 1)})
            if verbose:
                print(f"  loaded {t:<24} {n:>12,} rows in {secs:6.1f}s")
    finally:
        conn.close()
    return pd.DataFrame(summary)


def list_tables(conn: sqlite3.Connection) -> list[str]:
    q = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    return [r[0] for r in conn.execute(q).fetchall()]


def run_query(sql: str, conn: sqlite3.Connection, params: tuple | dict | None = None) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def run_sql_script(path: str | Path, conn: sqlite3.Connection) -> None:
    conn.executescript(Path(path).read_text(encoding="utf-8"))
    conn.commit()


def load_named_queries(path: str | Path) -> dict[str, dict]:
    """Parse a .sql file where each query starts with `-- name: <id>` and an optional `-- question: <text>`."""
    text = Path(path).read_text(encoding="utf-8")
    blocks = re.split(r"^--\s*name:\s*", text, flags=re.MULTILINE)[1:]
    queries = {}
    for block in blocks:
        name, _, body = block.partition("\n")
        question = ""
        m = re.search(r"^--\s*question:\s*(.+)$", body, flags=re.MULTILINE)
        if m:
            question = m.group(1).strip()
        sql_lines = [ln for ln in body.splitlines() if not ln.strip().startswith("--")]
        sql = "\n".join(sql_lines).strip().rstrip(";")
        queries[name.strip()] = {"question": question, "sql": sql}
    return queries
