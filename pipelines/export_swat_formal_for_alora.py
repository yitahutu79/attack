#!/usr/bin/env python3
"""Export the repository's SWaT formal split into ALoRa's CSV layout.

This keeps the split consistent with the paper:
  - benign train
  - benign calibration
  - benign test + attack test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_loaders import _frame_to_numeric_and_labels, _read_csv


def _write_feature_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", *frame.columns.tolist()])
        for idx, row in enumerate(frame.itertuples(index=False, name=None)):
            writer.writerow([idx, *row])


def _write_label_csv(path: Path, labels: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "label"])
        for idx, value in enumerate(labels):
            writer.writerow([idx, int(value)])


def main() -> None:
    ap = argparse.ArgumentParser(description="Export SWaT formal split for ALoRa")
    ap.add_argument("--data-dir", default="dataset/SWaT", help="SWaT directory")
    ap.add_argument("--out-dir", default="external/ALoRa/Datasets/SWAT_FORMAL", help="ALoRa-formatted output dir")
    ap.add_argument("--benign-file", default="benign_data/benign_samples_5sec.csv")
    ap.add_argument("--attack-file", default="attack_data/attack_samples_5sec.csv")
    ap.add_argument("--train-fraction", type=float, default=0.6)
    ap.add_argument("--calib-fraction", type=float, default=0.2)
    args = ap.parse_args()

    root = Path(args.data_dir)
    benign_path = root / args.benign_file
    attack_path = root / args.attack_file
    out_dir = Path(args.out_dir)

    benign_frame, benign_y = _frame_to_numeric_and_labels(_read_csv(benign_path), "swat")
    attack_frame, attack_y = _frame_to_numeric_and_labels(_read_csv(attack_path), "swat")

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

    test_frame = pd.concat([benign_test, attack_test], axis=0, ignore_index=True)
    val_labels = [0] * len(benign_calib)
    test_labels = [0] * len(benign_test) + [1] * len(attack_test)

    _write_feature_csv(out_dir / "train_minute.csv", benign_train)
    _write_feature_csv(out_dir / "val_minute.csv", benign_calib)
    _write_label_csv(out_dir / "val_minute_labels.csv", val_labels)
    _write_feature_csv(out_dir / "test_minute.csv", test_frame)
    _write_label_csv(out_dir / "test_minute_labels.csv", test_labels)

    metadata = {
        "source_data_dir": str(root.resolve()),
        "benign_file": str(benign_path),
        "attack_file": str(attack_path),
        "train_fraction": float(args.train_fraction),
        "calib_fraction": float(args.calib_fraction),
        "feature_count": len(common_cols),
        "feature_names": [str(c) for c in common_cols],
        "benign_rows": int(len(benign_frame)),
        "attack_rows": int(len(attack_frame)),
        "train_rows": int(len(benign_train)),
        "calib_rows": int(len(benign_calib)),
        "benign_test_rows": int(len(benign_test)),
        "test_rows_total": int(len(test_frame)),
        "attack_labels_positive": int(sum(test_labels)),
        "benign_labels_positive": int(sum(benign_y)),
        "attack_labels_from_source_positive": int(sum(attack_y)),
    }
    (out_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Exported SWaT formal split to: {out_dir}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
