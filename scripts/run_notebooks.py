"""Execute all stage notebooks in order (headless), stopping at the first failure.

    python scripts/run_notebooks.py                 # run everything
    python scripts/run_notebooks.py --from 8        # resume from stage 8
    python scripts/run_notebooks.py --inplace       # save outputs into the notebooks
    CREDIT_RISK_RAW_DIR=data/sample python scripts/run_notebooks.py   # smoke test on synthetic data
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--to", dest="end", type=int, default=99)
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--timeout", type=int, default=7200, help="seconds per cell")
    ap.add_argument("--kernel", default="python3")
    args = ap.parse_args()

    notebooks = sorted(p for p in NB_DIR.glob("[0-9][0-9]_*.ipynb") if args.start <= int(p.name[:2]) <= args.end)
    for nb_path in notebooks:
        start = time.time()
        print(f"> {nb_path.name} ...", end=" ", flush=True)
        nb = nbformat.read(nb_path, as_version=4)
        try:
            NotebookClient(nb, timeout=args.timeout, kernel_name=args.kernel, resources={"metadata": {"path": str(NB_DIR)}}).execute()
        except Exception:
            nbformat.write(nb, nb_path)  # keep the partial run so the failing cell's error is visible in the notebook
            print(f"FAILED after {time.time() - start:.0f}s. Open {nb_path.name} to see the error, then resume with --from {nb_path.name[:2]}")
            raise SystemExit(1)
        if args.inplace:
            nbformat.write(nb, nb_path)
        print(f"ok ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
