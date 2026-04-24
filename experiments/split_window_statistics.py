#!/usr/bin/env python3
"""Build split/window statistics tables for CICIDS2017, SWaT, and UNSW-NB15."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_loaders import (  # noqa: E402
    BENIGN_TOKENS,
    NON_FEATURE_COLS,
    _labels_from_frame,
    _read_csv,
)


FIELDNAMES = [
    "Dataset",
    "Split",
    "#Records",
    "#Windows",
    "#Benign Windows",
    "#Anomalous Windows",
    "Attack Types",
    "Window Size",
    "Stride",
    "Anomaly Ratio Threshold",
]


CICIDS_TRAIN_FILES = [
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
]
CICIDS_TEST_FILES = [
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

UNSW_TRAIN_FILE = "Training and Testing Sets/UNSW_NB15_training-set.csv"
UNSW_TEST_FILE = "Training and Testing Sets/UNSW_NB15_testing-set.csv"

SWAT_BENIGN_FILE = "benign_data/benign_samples_5sec.csv"
SWAT_ATTACK_FILE = "attack_data/attack_samples_5sec.csv"

SWAT_TRAIN_FRACTION = 0.6
SWAT_CALIB_FRACTION = 0.2

BENIGN_ATTACK_TYPE_TOKENS = {
    *BENIGN_TOKENS,
    "nan",
    "none",
    "null",
    "",
}


def count_csv_records(path: Path) -> int:
    """Count data rows in a CSV without loading it into memory."""
    newline_count = 0
    last_byte = b""
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            newline_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if not last_byte:
        return 0
    line_count = newline_count if last_byte == b"\n" else newline_count + 1
    return max(int(line_count) - 1, 0)


def select_attack_type_column(df: pd.DataFrame, dataset: str) -> str | None:
    by_lower = {str(c).strip().lower(): c for c in df.columns}
    preferred: list[str]
    if dataset == "cicids2017":
        preferred = ["label", "attack_cat", "type"]
    elif dataset == "swat":
        # SWaT label_full is extremely fine-grained (hundreds of classes) and not table-friendly.
        # Prefer high-level attack taxonomy first.
        preferred = ["label2", "label3", "label4", "label_full", "label1", "type", "label"]
    elif dataset == "unsw_nb15":
        preferred = ["attack_cat", "label", "type"]
    else:
        preferred = ["attack_cat", "label_full", "label", "type"]
    for key in preferred:
        if key in by_lower:
            return str(by_lower[key])
    return None


def normalize_attack_type_token(value: Any) -> str | None:
    token = re.sub(r"\s+", " ", str(value).strip())
    if token.lower() in BENIGN_ATTACK_TYPE_TOKENS:
        return None
    return token


def extract_attack_types(df: pd.DataFrame, dataset: str) -> np.ndarray | None:
    col = select_attack_type_column(df, dataset)
    if col is None:
        return None
    values = df[col].astype(str).to_numpy()
    return values


def preprocess_frame_with_attack_types(
    df: pd.DataFrame,
    dataset: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    labels = _labels_from_frame(df, dataset)
    attack_types = extract_attack_types(df, dataset)

    drop_cols = [c for c in df.columns if str(c).strip().lower() in NON_FEATURE_COLS]
    numeric = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError("No numeric features available after preprocessing.")

    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    if len(numeric) != len(labels):
        keep_idx = numeric.index.to_numpy()
        labels = labels[keep_idx]
        if attack_types is not None:
            attack_types = attack_types[keep_idx]
    numeric = numeric.reset_index(drop=True)
    return numeric, labels.astype(np.uint8), attack_types


def load_payload(data_dir: Path, dataset: str, rel_path: str) -> dict[str, Any]:
    path = (data_dir / rel_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    raw_records = count_csv_records(path)
    df = _read_csv(path)
    numeric, labels, attack_types = preprocess_frame_with_attack_types(df, dataset)
    return {
        "path": path,
        "rel_path": rel_path,
        "raw_records": int(raw_records),
        "numeric": numeric,
        "labels": labels,
        "attack_types": attack_types,
    }


def align_payload_features(payloads: list[dict[str, Any]]) -> None:
    common = set(payloads[0]["numeric"].columns)
    for payload in payloads[1:]:
        common &= set(payload["numeric"].columns)
    feature_names = [c for c in payloads[0]["numeric"].columns if c in common]
    if not feature_names:
        raise ValueError("No common numeric feature columns across selected files.")
    for payload in payloads:
        payload["numeric"] = payload["numeric"].loc[:, feature_names].reset_index(drop=True)


def count_windows_for_segment(
    labels: np.ndarray,
    attack_types: np.ndarray | None,
    *,
    window_size: int,
    stride: int,
    anomaly_ratio_threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    n = int(labels.size)
    if n < int(window_size):
        return {
            "windows": 0,
            "benign_windows": 0,
            "anomalous_windows": 0,
            "attack_type_windows": Counter(),
        }

    total = n - int(window_size) + 1
    starts = np.arange(0, total, int(stride), dtype=np.int64)
    prefix = np.concatenate(([0], np.cumsum(labels, dtype=np.int64)))
    attack_counts = prefix[starts + int(window_size)] - prefix[starts]
    anomalous_mask = (attack_counts.astype(np.float64) / float(window_size)) >= float(anomaly_ratio_threshold)

    windows = int(starts.size)
    anomalous_windows = int(np.sum(anomalous_mask))
    benign_windows = windows - anomalous_windows

    type_counter: Counter[str] = Counter()
    if attack_types is not None and anomalous_windows > 0:
        attack_types = np.asarray(attack_types).reshape(-1)
        for start in starts[anomalous_mask]:
            end = int(start + int(window_size))
            win_labels = labels[start:end]
            win_types = attack_types[start:end]
            win_attack_types: list[str] = []
            for t, y in zip(win_types, win_labels, strict=True):
                if int(y) != 1:
                    continue
                token = normalize_attack_type_token(t)
                if token is not None:
                    win_attack_types.append(token)
            if not win_attack_types:
                continue
            dominant_type = Counter(win_attack_types).most_common(1)[0][0]
            type_counter[dominant_type] += 1

    return {
        "windows": windows,
        "benign_windows": benign_windows,
        "anomalous_windows": anomalous_windows,
        "attack_type_windows": type_counter,
    }


def aggregate_split_window_stats(
    segments: list[tuple[np.ndarray, np.ndarray | None]],
    *,
    window_size: int,
    stride: int,
    anomaly_ratio_threshold: float,
    keep_benign_only: bool,
) -> dict[str, Any]:
    stats = {
        "records": 0,
        "windows": 0,
        "benign_windows": 0,
        "anomalous_windows": 0,
        "attack_type_windows": Counter(),
    }
    for labels, attack_types in segments:
        seg_stats = count_windows_for_segment(
            labels,
            attack_types,
            window_size=window_size,
            stride=stride,
            anomaly_ratio_threshold=anomaly_ratio_threshold,
        )
        stats["records"] += int(len(labels))
        stats["windows"] += int(seg_stats["windows"])
        stats["benign_windows"] += int(seg_stats["benign_windows"])
        stats["anomalous_windows"] += int(seg_stats["anomalous_windows"])
        stats["attack_type_windows"].update(seg_stats["attack_type_windows"])

    if keep_benign_only:
        stats["windows"] = int(stats["benign_windows"])
        stats["anomalous_windows"] = 0
        stats["attack_type_windows"] = Counter()

    return stats


def format_attack_types(counter: Counter[str]) -> str:
    if not counter:
        return "-"
    pairs = sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return "; ".join(f"{k}:{v}" for k, v in pairs)


def make_row(
    *,
    dataset: str,
    split: str,
    records: int | str,
    windows: int | str,
    benign_windows: int | str,
    anomalous_windows: int | str,
    attack_types: str,
    window_size: int,
    stride: int,
    anomaly_ratio_threshold: float,
) -> dict[str, Any]:
    return {
        "Dataset": dataset,
        "Split": split,
        "#Records": records,
        "#Windows": windows,
        "#Benign Windows": benign_windows,
        "#Anomalous Windows": anomalous_windows,
        "Attack Types": attack_types,
        "Window Size": int(window_size),
        "Stride": int(stride),
        "Anomaly Ratio Threshold": float(anomaly_ratio_threshold),
    }


def build_cicids_rows(
    *,
    data_dir: Path,
    window_size: int,
    stride: int,
    anomaly_ratio_threshold: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    payloads = [
        load_payload(data_dir, "cicids2017", rel_path)
        for rel_path in CICIDS_TRAIN_FILES + CICIDS_TEST_FILES
    ]
    align_payload_features(payloads)

    train_payloads = payloads[: len(CICIDS_TRAIN_FILES)]
    test_payloads = payloads[len(CICIDS_TRAIN_FILES) :]
    train_segments = [(p["labels"], p["attack_types"]) for p in train_payloads]
    test_segments = [(p["labels"], p["attack_types"]) for p in test_payloads]

    train_stats = aggregate_split_window_stats(
        train_segments,
        window_size=window_size,
        stride=stride,
        anomaly_ratio_threshold=anomaly_ratio_threshold,
        keep_benign_only=True,
    )
    test_stats = aggregate_split_window_stats(
        test_segments,
        window_size=window_size,
        stride=stride,
        anomaly_ratio_threshold=anomaly_ratio_threshold,
        keep_benign_only=False,
    )
    benign_train_records = int(sum(np.sum(p["labels"] == 0) for p in train_payloads))
    raw_records_total = int(sum(int(p["raw_records"]) for p in payloads))

    rows = [
        make_row(
            dataset="CICIDS2017",
            split="raw_total",
            records=raw_records_total,
            windows="-",
            benign_windows="-",
            anomalous_windows="-",
            attack_types="-",
            window_size=window_size,
            stride=stride,
            anomaly_ratio_threshold=anomaly_ratio_threshold,
        ),
        make_row(
            dataset="CICIDS2017",
            split="train",
            records=int(train_stats["records"]),
            windows=int(train_stats["windows"]),
            benign_windows=int(train_stats["benign_windows"]),
            anomalous_windows=int(train_stats["anomalous_windows"]),
            attack_types="-",
            window_size=window_size,
            stride=stride,
            anomaly_ratio_threshold=anomaly_ratio_threshold,
        ),
        make_row(
            dataset="CICIDS2017",
            split="calibration(train_benign_reuse)",
            records=benign_train_records,
            windows=int(train_stats["windows"]),
            benign_windows=int(train_stats["windows"]),
            anomalous_windows=0,
            attack_types="-",
            window_size=window_size,
            stride=stride,
            anomaly_ratio_threshold=anomaly_ratio_threshold,
        ),
        make_row(
            dataset="CICIDS2017",
            split="test",
            records=int(test_stats["records"]),
            windows=int(test_stats["windows"]),
            benign_windows=int(test_stats["benign_windows"]),
            anomalous_windows=int(test_stats["anomalous_windows"]),
            attack_types=format_attack_types(test_stats["attack_type_windows"]),
            window_size=window_size,
            stride=stride,
            anomaly_ratio_threshold=anomaly_ratio_threshold,
        ),
    ]
    notes = [
        "CICIDS2017 calibration windows are reused from benign train windows (no separate calibration file split).",
    ]
    return rows, notes


def build_swat_rows(
    *,
    data_dir: Path,
    window_size: int,
    stride: int,
    anomaly_ratio_threshold: float,
    train_fraction: float,
    calib_fraction: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    benign_payload = load_payload(data_dir, "swat", SWAT_BENIGN_FILE)
    attack_payload = load_payload(data_dir, "swat", SWAT_ATTACK_FILE)
    align_payload_features([benign_payload, attack_payload])

    benign_labels = benign_payload["labels"]
    benign_types = benign_payload["attack_types"]
    attack_labels = attack_payload["labels"]
    attack_types = attack_payload["attack_types"]

    n_benign = int(len(benign_labels))
    train_end = int(n_benign * float(train_fraction))
    calib_end = int(n_benign * float(train_fraction + calib_fraction))
    min_rows = int(window_size)
    train_end = max(train_end, min_rows)
    calib_end = max(calib_end, train_end + min_rows)
    calib_end = min(calib_end, n_benign - min_rows)
    if train_end <= 0 or calib_end <= train_end or calib_end >= n_benign:
        raise ValueError("SWaT split failed: benign file is too short for train/calib/test windowing.")

    train_segments = [(benign_labels[:train_end], None if benign_types is None else benign_types[:train_end])]
    calib_segments = [
        (benign_labels[train_end:calib_end], None if benign_types is None else benign_types[train_end:calib_end])
    ]
    test_segments = [
        (benign_labels[calib_end:], None if benign_types is None else benign_types[calib_end:]),
        (attack_labels, attack_types),
    ]

    train_stats = aggregate_split_window_stats(
        train_segments,
        window_size=window_size,
        stride=stride,
        anomaly_ratio_threshold=anomaly_ratio_threshold,
        keep_benign_only=True,
    )
    calib_stats = aggregate_split_window_stats(
        calib_segments,
        window_size=window_size,
        stride=stride,
        anomaly_ratio_threshold=anomaly_ratio_threshold,
        keep_benign_only=True,
    )
    test_stats = aggregate_split_window_stats(
        test_segments,
        window_size=window_size,
        stride=stride,
        anomaly_ratio_threshold=anomaly_ratio_threshold,
        keep_benign_only=False,
    )
    raw_records_total = int(benign_payload["raw_records"] + attack_payload["raw_records"])

    rows = [
        make_row(
            dataset="SWaT",
            split="raw_total",
            records=raw_records_total,
            windows="-",
            benign_windows="-",
            anomalous_windows="-",
            attack_types="-",
            window_size=window_size,
            stride=stride,
            anomaly_ratio_threshold=anomaly_ratio_threshold,
        ),
        make_row(
            dataset="SWaT",
            split="train",
            records=int(train_stats["records"]),
            windows=int(train_stats["windows"]),
            benign_windows=int(train_stats["benign_windows"]),
            anomalous_windows=int(train_stats["anomalous_windows"]),
            attack_types="-",
            window_size=window_size,
            stride=stride,
            anomaly_ratio_threshold=anomaly_ratio_threshold,
        ),
        make_row(
            dataset="SWaT",
            split="calibration",
            records=int(calib_stats["records"]),
            windows=int(calib_stats["windows"]),
            benign_windows=int(calib_stats["benign_windows"]),
            anomalous_windows=int(calib_stats["anomalous_windows"]),
            attack_types="-",
            window_size=window_size,
            stride=stride,
            anomaly_ratio_threshold=anomaly_ratio_threshold,
        ),
        make_row(
            dataset="SWaT",
            split="test",
            records=int(test_stats["records"]),
            windows=int(test_stats["windows"]),
            benign_windows=int(test_stats["benign_windows"]),
            anomalous_windows=int(test_stats["anomalous_windows"]),
            attack_types=format_attack_types(test_stats["attack_type_windows"]),
            window_size=window_size,
            stride=stride,
            anomaly_ratio_threshold=anomaly_ratio_threshold,
        ),
    ]
    return rows, []


def build_unsw_rows(
    *,
    data_dir: Path,
    window_size: int,
    stride: int,
    anomaly_ratio_threshold: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    train_payload = load_payload(data_dir, "unsw_nb15", UNSW_TRAIN_FILE)
    test_payload = load_payload(data_dir, "unsw_nb15", UNSW_TEST_FILE)
    align_payload_features([train_payload, test_payload])

    train_segments = [(train_payload["labels"], train_payload["attack_types"])]
    test_segments = [(test_payload["labels"], test_payload["attack_types"])]
    train_stats = aggregate_split_window_stats(
        train_segments,
        window_size=window_size,
        stride=stride,
        anomaly_ratio_threshold=anomaly_ratio_threshold,
        keep_benign_only=True,
    )
    test_stats = aggregate_split_window_stats(
        test_segments,
        window_size=window_size,
        stride=stride,
        anomaly_ratio_threshold=anomaly_ratio_threshold,
        keep_benign_only=False,
    )
    benign_train_records = int(np.sum(train_payload["labels"] == 0))
    raw_records_total = int(train_payload["raw_records"] + test_payload["raw_records"])

    rows = [
        make_row(
            dataset="UNSW-NB15",
            split="raw_total",
            records=raw_records_total,
            windows="-",
            benign_windows="-",
            anomalous_windows="-",
            attack_types="-",
            window_size=window_size,
            stride=stride,
            anomaly_ratio_threshold=anomaly_ratio_threshold,
        ),
        make_row(
            dataset="UNSW-NB15",
            split="train",
            records=int(train_stats["records"]),
            windows=int(train_stats["windows"]),
            benign_windows=int(train_stats["benign_windows"]),
            anomalous_windows=int(train_stats["anomalous_windows"]),
            attack_types="-",
            window_size=window_size,
            stride=stride,
            anomaly_ratio_threshold=anomaly_ratio_threshold,
        ),
        make_row(
            dataset="UNSW-NB15",
            split="calibration(train_benign_reuse)",
            records=benign_train_records,
            windows=int(train_stats["windows"]),
            benign_windows=int(train_stats["windows"]),
            anomalous_windows=0,
            attack_types="-",
            window_size=window_size,
            stride=stride,
            anomaly_ratio_threshold=anomaly_ratio_threshold,
        ),
        make_row(
            dataset="UNSW-NB15",
            split="test",
            records=int(test_stats["records"]),
            windows=int(test_stats["windows"]),
            benign_windows=int(test_stats["benign_windows"]),
            anomalous_windows=int(test_stats["anomalous_windows"]),
            attack_types=format_attack_types(test_stats["attack_type_windows"]),
            window_size=window_size,
            stride=stride,
            anomaly_ratio_threshold=anomaly_ratio_threshold,
        ),
    ]
    notes = [
        "UNSW-NB15 calibration windows are reused from benign train windows (official split has no standalone calibration file).",
    ]
    return rows, notes


def cell(value: Any) -> str:
    return "-" if value is None else str(value)


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], out_md: Path, notes: list[str]) -> None:
    lines: list[str] = []
    lines.append("# Split and Window Statistics")
    lines.append("")
    lines.append(
        "Window label rule: A window is labeled anomalous when the attack-record ratio "
        "within that window is >= Anomaly Ratio Threshold."
    )
    lines.append("")
    lines.append("| " + " | ".join(FIELDNAMES) + " |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    cell(row["Dataset"]),
                    cell(row["Split"]),
                    cell(row["#Records"]),
                    cell(row["#Windows"]),
                    cell(row["#Benign Windows"]),
                    cell(row["#Anomalous Windows"]),
                    cell(row["Attack Types"]),
                    cell(row["Window Size"]),
                    cell(row["Stride"]),
                    cell(row["Anomaly Ratio Threshold"]),
                ]
            )
            + " |"
        )
    if notes:
        lines.append("")
        lines.append("Notes:")
        for note in notes:
            lines.append(f"- {note}")
    lines.append("- Attack type counts are computed on anomalous windows by dominant attack type per window.")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate split/window statistics tables.")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--anomaly-ratio-threshold", type=float, default=0.15)
    parser.add_argument("--cicids-data-dir", default="dataset/CICIDS2017")
    parser.add_argument("--swat-data-dir", default="dataset/SWaT")
    parser.add_argument("--unsw-data-dir", default="dataset/UNSW-NB15")
    parser.add_argument("--out-csv", default=str(ROOT / "split_window_statistics.csv"))
    parser.add_argument("--out-md", default=str(ROOT / "split_window_statistics.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    window_size = int(args.window_size)
    stride = int(args.stride)
    anomaly_ratio_threshold = float(args.anomaly_ratio_threshold)

    cicids_rows, cicids_notes = build_cicids_rows(
        data_dir=(ROOT / args.cicids_data_dir).resolve(),
        window_size=window_size,
        stride=stride,
        anomaly_ratio_threshold=anomaly_ratio_threshold,
    )
    swat_rows, swat_notes = build_swat_rows(
        data_dir=(ROOT / args.swat_data_dir).resolve(),
        window_size=window_size,
        stride=stride,
        anomaly_ratio_threshold=anomaly_ratio_threshold,
        train_fraction=SWAT_TRAIN_FRACTION,
        calib_fraction=SWAT_CALIB_FRACTION,
    )
    unsw_rows, unsw_notes = build_unsw_rows(
        data_dir=(ROOT / args.unsw_data_dir).resolve(),
        window_size=window_size,
        stride=stride,
        anomaly_ratio_threshold=anomaly_ratio_threshold,
    )

    rows = cicids_rows + swat_rows + unsw_rows
    notes = cicids_notes + swat_notes + unsw_notes

    out_csv = Path(args.out_csv).resolve()
    out_md = Path(args.out_md).resolve()
    write_csv(rows, out_csv)
    write_markdown(rows, out_md, notes)

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote MD: {out_md}")


if __name__ == "__main__":
    main()
