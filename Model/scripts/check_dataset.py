#!/usr/bin/env python3
"""Phase 3 helper: run this after every recording session.

Shows how many samples each sign has per participant and warns about
problems BEFORE you waste time training on bad data.

    python Model/scripts/check_dataset.py
"""
import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

FEATURE_DIM = 129
MIN_PER_CLASS = 300


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(root / "ml" / "data" / "processed" / "static"))
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.data) / "*.csv")))
    if not files:
        sys.exit(f"No CSV files in {args.data}\nDownload them from the Collect data page and put them there.")

    df = pd.concat([pd.read_csv(f, dtype={"participant_id": str, "label": str}) for f in files], ignore_index=True)
    feat_cols = [f"f{i}" for i in range(FEATURE_DIM)]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        sys.exit(f"CSV is missing feature columns (expected f0..f{FEATURE_DIM - 1}).")

    print(f"{len(files)} files, {len(df)} samples, {df['participant_id'].nunique()} participants, {df['label'].nunique()} signs\n")
    table = pd.crosstab(df["label"], df["participant_id"], margins=True, margins_name="TOTAL")
    print(table.to_string(), "\n")

    warnings = []
    if df[feat_cols].isna().any().any():
        warnings.append("Some feature values are empty (NaN).")
    zero_rows = int((df[feat_cols].abs().sum(axis=1) == 0).sum())
    if zero_rows:
        warnings.append(f"{zero_rows} rows are all zeros (no hand detected).")
    if df.duplicated(subset=["participant_id", "label"] + feat_cols).any():
        warnings.append("Duplicate rows found (same file added twice?).")
    low = table.loc[table.index != "TOTAL", "TOTAL"]
    low = low[low < MIN_PER_CLASS]
    if len(low):
        warnings.append(f"Signs with fewer than {MIN_PER_CLASS} samples: {', '.join(low.index)}")
    if df["participant_id"].nunique() < 3:
        warnings.append("Fewer than 3 participants: a participant-independent split is not possible yet.")

    print("Warnings:" if warnings else "Looks good.")
    for w in warnings:
        print(f"  - {w}")


if __name__ == "__main__":
    main()
