"""Download and extract the Home Credit Default Risk data from Kaggle into data/raw/.

Prerequisites:
  1. pip install kaggle   (included in environment.yml)
  2. Kaggle API token at ~/.kaggle/kaggle.json (Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json)
  3. Accept the competition rules: https://www.kaggle.com/competitions/home-credit-default-risk/rules
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
COMPETITION = "home-credit-default-risk"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    zip_path = RAW / f"{COMPETITION}.zip"
    if not zip_path.exists():
        print("Downloading (~690 MB zipped, ~2.5 GB extracted)...")
        subprocess.run([sys.executable, "-m", "kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(RAW)], check=True)
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW)
    for f in sorted(RAW.glob("*.csv")):
        print(f"  {f.name:<40} {f.stat().st_size / 1024**2:8.1f} MB")
    print("Done. You can delete the zip to save space.")


if __name__ == "__main__":
    main()
