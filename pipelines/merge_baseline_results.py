#!/usr/bin/env python3
"""Merge multiple baseline result CSV/JSON files into one table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge multiple baseline result files")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input baseline_results.csv files")
    ap.add_argument("--output-csv", required=True, help="Merged CSV output path")
    ap.add_argument("--output-json", default="", help="Optional merged JSON output path")
    ap.add_argument(
        "--dedupe-by",
        nargs="+",
        default=["method", "target_fpr"],
        help="Columns used to de-duplicate rows; later files win",
    )
    ap.add_argument(
        "--sort-by",
        nargs="+",
        default=["target_fpr", "method"],
        help="Columns used to sort merged rows",
    )
    args = ap.parse_args()

    frames: list[pd.DataFrame] = []
    for item in args.inputs:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f"input file not found: {path}")
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=args.dedupe_by, keep="last")
    merged = merged.sort_values(args.sort_by).reset_index(drop=True)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"merged csv saved to: {out_csv}")

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(merged.to_dict(orient="records"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"merged json saved to: {out_json}")

    view_cols = [c for c in ["method", "target_fpr", "auc", "ap", "calib_f1", "calib_recall", "test_benign_fpr"] if c in merged.columns]
    if view_cols:
        print()
        print(merged[view_cols])


if __name__ == "__main__":
    main()

