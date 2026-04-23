#!/usr/bin/env python3
"""Export the repository's SWaT formal split into TimesNet's CSV layout.

TimesNet's built-in SWAT loader expects:
  - swat_train2.csv
  - swat2.csv

and assumes the last column is the label. This exporter keeps the same
chronological split used in the paper, while packaging it into the loader's
expected file names.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_loaders import _frame_to_numeric_and_labels, _read_csv


def _with_label(frame: pd.DataFrame, label_value: int) -> pd.DataFrame:
    out = frame.copy()
    out["label"] = int(label_value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Export SWaT formal split for TimesNet")
    ap.add_argument("--data-dir", default="dataset/SWaT", help="SWaT directory")
    ap.add_argument(
        "--out-dir",
        default="external/Time-Series-Library/dataset/SWaT_FORMAL",
        help="TimesNet-formatted output dir",
    )
    ap.add_argument("--benign-file", default="benign_data/benign_samples_5sec.csv")
    ap.add_argument("--attack-file", default="attack_data/attack_samples_5sec.csv")
    ap.add_argument("--train-fraction", type=float, default=0.6)
    ap.add_argument("--calib-fraction", type=float, default=0.2)
    args = ap.parse_args()

    root = Path(args.data_dir)
    benign_path = root / args.benign_file
    attack_path = root / args.attack_file
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benign_frame, _ = _frame_to_numeric_and_labels(_read_csv(benign_path), "swat")
    attack_frame, _ = _frame_to_numeric_and_labels(_read_csv(attack_path), "swat")

    common_cols = [c for c in benign_frame.columns if c in set(attack_frame.columns)]
    if not common_cols:
        raise ValueError("No common numeric feature columns between benign and attack SWaT files.")

    benign_frame = benign_frame.loc[:, common_cols].reset_index(drop=True)
    attack_frame = attack_frame.loc[:, common_cols].reset_index(drop=True)

    n_benign = len(benign_frame)
    train_end = int(n_benign * float(args.train_fraction))
    calib_end = int(n_benign * (float(args.train_fraction) + float(args.calib_fraction)))
    if train_end <= 0 or calib_end <= train_end or calib_end >= n_benign:
        raise ValueError("Invalid benign split fractions for SWaT formal export.")

    benign_train = benign_frame.iloc[:train_end].reset_index(drop=True)
    benign_calib = benign_frame.iloc[train_end:calib_end].reset_index(drop=True)
    benign_test = benign_frame.iloc[calib_end:].reset_index(drop=True)
    attack_test = attack_frame.reset_index(drop=True)

    # TimesNet's built-in SWAT loader derives its own validation split from the
    # tail of swat_train2.csv, so we merge train+calib here to preserve the
    # chronological order while keeping a dedicated benign-only training file.
    train_frame = pd.concat(
        [_with_label(benign_train, 0), _with_label(benign_calib, 0)],
        axis=0,
        ignore_index=True,
    )
    test_frame = pd.concat(
        [_with_label(benign_test, 0), _with_label(attack_test, 1)],
        axis=0,
        ignore_index=True,
    )

    train_path = out_dir / "swat_train2.csv"
    test_path = out_dir / "swat2.csv"
    train_frame.to_csv(train_path, index=False)
    test_frame.to_csv(test_path, index=False)

    metadata = {
        "source_data_dir": str(root.resolve()),
        "train_fraction": float(args.train_fraction),
        "calib_fraction": float(args.calib_fraction),
        "feature_count": len(common_cols),
        "feature_names": [str(c) for c in common_cols],
        "benign_rows": int(len(benign_frame)),
        "attack_rows": int(len(attack_frame)),
        "train_rows_total": int(len(train_frame)),
        "test_rows_total": int(len(test_frame)),
        "test_positive_rows": int(len(attack_test)),
        "notes": "swat_train2.csv contains benign train+calibration rows; swat2.csv contains benign_test+attack_test rows.",
    }
    (out_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Exported TimesNet SWAT split to: {out_dir}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
