#!/usr/bin/env python3
"""Export local datasets into the CSV layout expected by TimesNet's SWAT loader.

The output directory will contain:
  - swat_train2.csv
  - swat2.csv

Both files store labels in the last column named ``label``. We reuse the SWAT
loader because it already supports multivariate anomaly detection on CSV files
with this structure.
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

from dataset_loaders import _frame_to_numeric_and_labels, _read_csv  # noqa: E402


def _load_numeric(path: Path, dataset: str) -> tuple[pd.DataFrame, pd.Series]:
    frame, labels = _frame_to_numeric_and_labels(_read_csv(path), dataset)
    return frame.reset_index(drop=True), pd.Series(labels.astype(int))


def _concat_with_common_cols(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    if not frames:
        raise ValueError("No frames to align.")
    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    cols = [c for c in frames[0].columns if c in common]
    if not cols:
        raise ValueError("No common numeric columns across input frames.")
    return [frame.loc[:, cols].reset_index(drop=True) for frame in frames]


def _append_label(frame: pd.DataFrame, labels: pd.Series | list[int]) -> pd.DataFrame:
    out = frame.copy()
    out["label"] = pd.Series(labels, dtype=int).reset_index(drop=True)
    return out


def export_cicids(data_dir: Path, out_dir: Path, train_files: list[str], test_files: list[str], train_fraction: float) -> dict:
    train_frames_raw: list[pd.DataFrame] = []
    train_labels_raw: list[pd.Series] = []
    test_frames_raw: list[pd.DataFrame] = []
    test_labels_raw: list[pd.Series] = []

    for name in train_files:
        frame, labels = _load_numeric(data_dir / name, "cicids2017")
        train_frames_raw.append(frame)
        train_labels_raw.append(labels)
    for name in test_files:
        frame, labels = _load_numeric(data_dir / name, "cicids2017")
        test_frames_raw.append(frame)
        test_labels_raw.append(labels)

    aligned = _concat_with_common_cols(train_frames_raw + test_frames_raw)
    train_frames = aligned[: len(train_frames_raw)]
    test_frames = aligned[len(train_frames_raw):]

    train_benign = pd.concat(
        [frame.loc[labels.values == 0] for frame, labels in zip(train_frames, train_labels_raw, strict=True)],
        ignore_index=True,
    )
    n_train = len(train_benign)
    split = max(int(n_train * train_fraction), 1)
    split = min(split, n_train - 1)
    train_frame = train_benign.iloc[:split].reset_index(drop=True)
    val_frame = train_benign.iloc[split:].reset_index(drop=True)
    merged_train = pd.concat([train_frame, val_frame], ignore_index=True)
    test_frame = pd.concat(test_frames, ignore_index=True)
    test_labels = pd.concat(test_labels_raw, ignore_index=True).astype(int)

    _append_label(merged_train, [0] * len(merged_train)).to_csv(out_dir / "swat_train2.csv", index=False)
    _append_label(test_frame, test_labels).to_csv(out_dir / "swat2.csv", index=False)

    return {
        "protocol": "cicids",
        "feature_count": int(merged_train.shape[1]),
        "train_rows_total": int(len(merged_train)),
        "test_rows_total": int(len(test_frame)),
        "test_positive_rows": int(test_labels.sum()),
    }


def export_unsw(data_dir: Path, out_dir: Path, train_file: str, test_file: str, train_fraction: float) -> dict:
    train_frame_raw, train_labels_raw = _load_numeric(data_dir / train_file, "unsw_nb15")
    test_frame_raw, test_labels_raw = _load_numeric(data_dir / test_file, "unsw_nb15")
    train_frame, test_frame = _concat_with_common_cols([train_frame_raw, test_frame_raw])

    benign_train = train_frame.loc[train_labels_raw.values == 0].reset_index(drop=True)
    n_train = len(benign_train)
    split = max(int(n_train * train_fraction), 1)
    split = min(split, n_train - 1)
    train_part = benign_train.iloc[:split].reset_index(drop=True)
    val_part = benign_train.iloc[split:].reset_index(drop=True)
    merged_train = pd.concat([train_part, val_part], ignore_index=True)

    _append_label(merged_train, [0] * len(merged_train)).to_csv(out_dir / "swat_train2.csv", index=False)
    _append_label(test_frame, test_labels_raw).to_csv(out_dir / "swat2.csv", index=False)

    return {
        "protocol": "unsw",
        "feature_count": int(merged_train.shape[1]),
        "train_rows_total": int(len(merged_train)),
        "test_rows_total": int(len(test_frame)),
        "test_positive_rows": int(test_labels_raw.sum()),
    }


def export_chrono(data_dir: Path, out_dir: Path, mixed_file: str, dataset_name: str, train_fraction: float, calib_fraction: float) -> dict:
    frame, labels = _load_numeric(data_dir / mixed_file, dataset_name)
    n = len(frame)
    train_end = max(int(n * train_fraction), 1)
    calib_end = max(int(n * (train_fraction + calib_fraction)), train_end + 1)
    calib_end = min(calib_end, n - 1)

    train_frame = frame.iloc[:train_end].loc[labels.iloc[:train_end].values == 0].reset_index(drop=True)
    val_frame = frame.iloc[train_end:calib_end].loc[labels.iloc[train_end:calib_end].values == 0].reset_index(drop=True)
    test_frame = frame.iloc[calib_end:].reset_index(drop=True)
    test_labels = labels.iloc[calib_end:].astype(int).reset_index(drop=True)

    if len(train_frame) == 0 or len(val_frame) == 0:
        raise ValueError("Chrono export produced empty benign train/val split.")

    merged_train = pd.concat([train_frame, val_frame], ignore_index=True)
    _append_label(merged_train, [0] * len(merged_train)).to_csv(out_dir / "swat_train2.csv", index=False)
    _append_label(test_frame, test_labels).to_csv(out_dir / "swat2.csv", index=False)

    return {
        "protocol": "chrono",
        "dataset": dataset_name,
        "feature_count": int(merged_train.shape[1]),
        "train_rows_total": int(len(merged_train)),
        "test_rows_total": int(len(test_frame)),
        "test_positive_rows": int(test_labels.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export datasets into TimesNet-compatible CSV layout")
    sub = ap.add_subparsers(dest="mode", required=True)

    cic = sub.add_parser("cicids")
    cic.add_argument("--data-dir", default="dataset/CICIDS2017")
    cic.add_argument("--out-dir", required=True)
    cic.add_argument("--train-fraction", type=float, default=0.8)
    cic.add_argument("--train-files", nargs="+", required=True)
    cic.add_argument("--test-files", nargs="+", required=True)

    unsw = sub.add_parser("unsw")
    unsw.add_argument("--data-dir", default="dataset/UNSW-NB15")
    unsw.add_argument("--out-dir", required=True)
    unsw.add_argument("--train-file", default="Training and Testing Sets/UNSW_NB15_training-set.csv")
    unsw.add_argument("--test-file", default="Training and Testing Sets/UNSW_NB15_testing-set.csv")
    unsw.add_argument("--train-fraction", type=float, default=0.8)

    chrono = sub.add_parser("chrono")
    chrono.add_argument("--data-dir", required=True)
    chrono.add_argument("--out-dir", required=True)
    chrono.add_argument("--mixed-file", required=True)
    chrono.add_argument("--dataset", required=True, choices=["ton_iot", "swat", "unsw_nb15", "cicids2017"])
    chrono.add_argument("--train-fraction", type=float, default=0.6)
    chrono.add_argument("--calib-fraction", type=float, default=0.1)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "cicids":
        meta = export_cicids(Path(args.data_dir), out_dir, args.train_files, args.test_files, args.train_fraction)
    elif args.mode == "unsw":
        meta = export_unsw(Path(args.data_dir), out_dir, args.train_file, args.test_file, args.train_fraction)
    else:
        meta = export_chrono(
            Path(args.data_dir),
            out_dir,
            args.mixed_file,
            args.dataset,
            args.train_fraction,
            args.calib_fraction,
        )

    (out_dir / "split_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
