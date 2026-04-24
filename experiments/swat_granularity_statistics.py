#!/usr/bin/env python3
"""SWaT granularity statistics (records/windows) without training any model.

This script reuses the same formal unsupervised split protocol as
`dataset_loaders.load_windowed_unsupervised_split`:
  - Train: benign prefix (benign-only windows kept)
  - Calibration: middle benign segment (benign-only windows kept)
  - Test: benign suffix + all attack windows

If a required CSV is only available as `.csv.tar.xz`, the script extracts it into
`dataset/SWaT/extracted/` (never overwriting the original archive).
"""

from __future__ import annotations

import argparse
import csv
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_loaders import (  # noqa: E402
    NON_FEATURE_COLS,
    _candidate_label_cols,
    _frame_to_numeric_and_labels,
    _read_csv,
)


FIELDNAMES = [
    "Granularity",
    "Benign Records",
    "Attack Records",
    "Train Records",
    "Calibration Records",
    "Test Benign Records",
    "Test Attack Records",
    "Train Windows",
    "Calibration Windows",
    "Test Benign Windows",
    "Test Anomalous Windows",
    "Total Test Windows",
    "Window Size",
    "Stride",
    "Anomaly Ratio Threshold",
]


@dataclass(frozen=True)
class GranularityResult:
    granularity: str
    benign_records: int
    attack_records: int
    train_records: int
    calib_records: int
    test_benign_records: int
    test_attack_records: int
    train_windows: int
    calib_windows: int
    test_benign_windows: int
    test_anom_windows: int
    test_total_windows: int
    window_size: int
    stride: int
    anomaly_ratio_threshold: float


@dataclass(frozen=True)
class InputFiles:
    granularity: str
    benign_csv: Path
    attack_csv: Path
    extracted_paths: list[Path]
    issues: list[str]
    benign_label_col: str | None
    attack_label_col: str | None


def _strip_csv_tar_xz_suffix(name: str) -> str:
    s = str(name)
    for suffix in [".csv.tar.xz", ".tar.xz", ".csv"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def _safe_extract_tar_xz(archive_path: Path, dest_dir: Path) -> tuple[list[Path], list[str]]:
    issues: list[str] = []
    extracted: list[Path] = []
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Extract only CSV members.
    with tarfile.open(archive_path, mode="r:xz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and str(m.name).lower().endswith(".csv")]
        if not members:
            return [], [f"Archive has no .csv members: {archive_path}"]
        if len(members) > 1:
            issues.append(
                "Archive has multiple .csv members; extracting all: {} -> {}".format(
                    archive_path.name, ", ".join(m.name for m in members)
                )
            )
        for m in members:
            # Prevent path traversal (tar slip).
            rel = Path(m.name)
            if rel.is_absolute() or ".." in rel.parts:
                issues.append(f"Refusing suspicious tar member path: {m.name}")
                continue
            out_path = (dest_dir / rel).resolve()
            if dest_dir.resolve() not in out_path.parents and out_path != dest_dir.resolve():
                issues.append(f"Refusing to extract outside destination: {m.name}")
                continue
            if out_path.exists():
                extracted.append(out_path)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as src, out_path.open("wb") as dst:
                if src is None:
                    issues.append(f"Failed to read tar member: {m.name}")
                    continue
                dst.write(src.read())
            extracted.append(out_path)

    return extracted, issues


def _locate_or_extract_csv(data_dir: Path, rel_csv: str) -> tuple[Path | None, list[Path], list[str]]:
    """Return (csv_path, extracted_paths, issues)."""
    issues: list[str] = []
    extracted_paths: list[Path] = []
    csv_path = (data_dir / rel_csv).resolve()
    if csv_path.exists():
        return csv_path, extracted_paths, issues

    archive_path = (data_dir / f"{rel_csv}.tar.xz").resolve()
    if not archive_path.exists():
        return None, extracted_paths, [f"Missing both {rel_csv} and {rel_csv}.tar.xz under {data_dir}"]

    extract_root = (data_dir / "extracted").resolve()
    extract_tag = _strip_csv_tar_xz_suffix(archive_path.name)
    dest_dir = extract_root / extract_tag
    extracted, extract_issues = _safe_extract_tar_xz(archive_path, dest_dir)
    extracted_paths.extend(extracted)
    issues.extend(extract_issues)

    # Try to find a single CSV to use.
    candidates = [p for p in extracted if p.name == Path(rel_csv).name]
    if not candidates:
        candidates = [p for p in extracted if p.suffix.lower() == ".csv"]
    if not candidates:
        return None, extracted_paths, issues + [f"Archive extracted but no CSV found to use: {archive_path}"]
    if len(candidates) > 1:
        issues.append(
            "Multiple extracted CSV candidates match; using the first: {}".format(
                ", ".join(str(p.relative_to(extract_root)) for p in candidates)
            )
        )
    return candidates[0], extracted_paths, issues


def _label_col_used(df: Any, dataset: str) -> str | None:
    try:
        cols = _candidate_label_cols(df, dataset)
        return str(cols[0]) if cols else None
    except Exception:
        return None


def locate_swat_files(data_dir: Path, granularity_sec: int) -> InputFiles:
    gran = f"{granularity_sec}sec"
    issues: list[str] = []
    extracted: list[Path] = []

    benign_rel = f"benign_data/benign_samples_{gran}.csv"
    attack_rel = f"attack_data/attack_samples_{gran}.csv"

    benign_csv, benign_extracted, benign_issues = _locate_or_extract_csv(data_dir, benign_rel)
    attack_csv, attack_extracted, attack_issues = _locate_or_extract_csv(data_dir, attack_rel)

    extracted.extend(benign_extracted)
    extracted.extend(attack_extracted)
    issues.extend(benign_issues)
    issues.extend(attack_issues)

    benign_label_col = None
    attack_label_col = None
    if benign_csv is not None:
        try:
            benign_label_col = _label_col_used(_read_csv(benign_csv), "swat")
        except Exception as e:
            issues.append(f"Failed to read benign CSV for label-col check: {benign_csv} ({e})")
    if attack_csv is not None:
        try:
            attack_label_col = _label_col_used(_read_csv(attack_csv), "swat")
        except Exception as e:
            issues.append(f"Failed to read attack CSV for label-col check: {attack_csv} ({e})")

    if benign_csv is None:
        issues.append(f"Missing benign file for granularity {gran}.")
    if attack_csv is None:
        issues.append(f"Missing attack file for granularity {gran}.")

    return InputFiles(
        granularity=gran,
        benign_csv=benign_csv or Path(),
        attack_csv=attack_csv or Path(),
        extracted_paths=sorted(set(extracted)),
        issues=issues,
        benign_label_col=benign_label_col,
        attack_label_col=attack_label_col,
    )


def _count_windows_labels(
    labels: np.ndarray,
    *,
    window_size: int,
    stride: int,
    anomaly_ratio_threshold: float,
) -> tuple[int, int, int]:
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    n = int(labels.size)
    if n < int(window_size):
        return 0, 0, 0
    total = n - int(window_size) + 1
    starts = np.arange(0, total, int(stride), dtype=np.int64)
    prefix = np.concatenate(([0], np.cumsum(labels, dtype=np.int64)))
    attack_counts = prefix[starts + int(window_size)] - prefix[starts]
    anomalous_mask = (attack_counts.astype(np.float64) / float(window_size)) >= float(anomaly_ratio_threshold)
    windows = int(starts.size)
    anom = int(np.sum(anomalous_mask))
    benign = windows - anom
    return windows, benign, anom


def _split_indices(
    n_benign: int,
    *,
    window_size: int,
    train_fraction: float,
    calib_fraction: float,
) -> tuple[int, int]:
    train_end = int(n_benign * float(train_fraction))
    calib_end = int(n_benign * float(train_fraction + calib_fraction))
    min_rows = int(window_size)
    train_end = max(train_end, min_rows)
    calib_end = max(calib_end, train_end + min_rows)
    calib_end = min(calib_end, n_benign - min_rows)
    if train_end <= 0 or calib_end <= train_end or calib_end >= n_benign:
        raise ValueError(
            f"benign records too small for split: n={n_benign}, train_end={train_end}, calib_end={calib_end}, window={window_size}"
        )
    return train_end, calib_end


def compute_granularity_stats(
    inputs: InputFiles,
    *,
    window_size: int,
    stride: int,
    anomaly_ratio_threshold: float,
    train_fraction: float,
    calib_fraction: float,
) -> tuple[GranularityResult | None, list[str]]:
    issues = list(inputs.issues)
    if not inputs.benign_csv.exists() or not inputs.attack_csv.exists():
        return None, issues

    try:
        benign_df = _read_csv(inputs.benign_csv)
        attack_df = _read_csv(inputs.attack_csv)
    except Exception as e:
        return None, issues + [f"Failed to read CSV: {e}"]

    try:
        benign_numeric, benign_labels = _frame_to_numeric_and_labels(benign_df, "swat")
        attack_numeric, attack_labels = _frame_to_numeric_and_labels(attack_df, "swat")
    except Exception as e:
        return None, issues + [f"Failed to preprocess SWaT CSVs: {e}"]

    if inputs.benign_label_col != inputs.attack_label_col and inputs.benign_label_col and inputs.attack_label_col:
        issues.append(
            f"Label column differs (benign={inputs.benign_label_col}, attack={inputs.attack_label_col}) for {inputs.granularity}."
        )

    # Check feature columns overlap (should be large; mismatch may signal file-version drift).
    benign_cols = [c for c in benign_df.columns if str(c).strip().lower() not in NON_FEATURE_COLS]
    attack_cols = [c for c in attack_df.columns if str(c).strip().lower() not in NON_FEATURE_COLS]
    common_cols = set(benign_cols) & set(attack_cols)
    if len(common_cols) == 0:
        issues.append(f"No common non-meta columns between benign/attack raw CSVs for {inputs.granularity}.")

    n_benign = int(len(benign_numeric))
    n_attack = int(len(attack_numeric))

    try:
        train_end, calib_end = _split_indices(
            n_benign,
            window_size=window_size,
            train_fraction=train_fraction,
            calib_fraction=calib_fraction,
        )
    except Exception as e:
        return None, issues + [f"Failed to build split indices for {inputs.granularity}: {e}"]

    train_labels = benign_labels[:train_end]
    calib_labels = benign_labels[train_end:calib_end]
    test_benign_labels = benign_labels[calib_end:]
    test_attack_labels = attack_labels

    # Windows per segment using the same label rule (attack_ratio >= threshold).
    train_w, train_b, _train_a = _count_windows_labels(
        train_labels, window_size=window_size, stride=stride, anomaly_ratio_threshold=anomaly_ratio_threshold
    )
    calib_w, calib_b, _calib_a = _count_windows_labels(
        calib_labels, window_size=window_size, stride=stride, anomaly_ratio_threshold=anomaly_ratio_threshold
    )
    test_ben_w, test_ben_b, test_ben_a = _count_windows_labels(
        test_benign_labels, window_size=window_size, stride=stride, anomaly_ratio_threshold=anomaly_ratio_threshold
    )
    test_att_w, test_att_b, test_att_a = _count_windows_labels(
        test_attack_labels, window_size=window_size, stride=stride, anomaly_ratio_threshold=anomaly_ratio_threshold
    )

    # Formal split keeps benign-only windows for train/calib.
    train_windows = int(train_b)
    calib_windows = int(calib_b)

    # Test combines benign suffix + all attack windows.
    test_total_windows = int(test_ben_w + test_att_w)
    test_benign_windows = int(test_ben_b + test_att_b)
    test_anom_windows = int(test_ben_a + test_att_a)

    res = GranularityResult(
        granularity=inputs.granularity,
        benign_records=n_benign,
        attack_records=n_attack,
        train_records=int(train_end),
        calib_records=int(calib_end - train_end),
        test_benign_records=int(n_benign - calib_end),
        test_attack_records=int(n_attack),
        train_windows=train_windows,
        calib_windows=calib_windows,
        test_benign_windows=test_benign_windows,
        test_anom_windows=test_anom_windows,
        test_total_windows=test_total_windows,
        window_size=int(window_size),
        stride=int(stride),
        anomaly_ratio_threshold=float(anomaly_ratio_threshold),
    )

    # Extra sanity checks that should never fail silently.
    if test_total_windows != test_benign_windows + test_anom_windows:
        issues.append(
            f"Internal inconsistency: total_test_windows != benign+anom for {inputs.granularity} ({test_total_windows} != {test_benign_windows}+{test_anom_windows})."
        )

    return res, issues


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


def write_md(rows: list[dict[str, Any]], out_md: Path, issues: list[str], extracted: list[Path]) -> None:
    lines: list[str] = []
    lines.append("# SWaT Granularity Statistics")
    lines.append("")
    lines.append(
        "Window label rule: A window is labeled anomalous when the attack-record ratio within that window "
        "is >= Anomaly Ratio Threshold."
    )
    lines.append("")
    lines.append(
        "Records are counted after applying the same preprocessing as the main pipeline "
        "(drop non-feature columns, keep numeric features, drop NaN/inf rows)."
    )
    lines.append("")
    lines.append("| " + " | ".join(FIELDNAMES) + " |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                str(r.get(k, "-"))
                for k in FIELDNAMES
            )
            + " |"
        )
    if extracted:
        lines.append("")
        lines.append("Auto-extracted archives:")
        for p in extracted:
            lines.append(f"- {p}")
    if issues:
        lines.append("")
        lines.append("Issues (not silently skipped):")
        for msg in issues:
            lines.append(f"- {msg}")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SWaT 1/3/5/10sec records/windows statistics without training.")
    p.add_argument("--data-dir", default="dataset/SWaT")
    p.add_argument("--granularities", default="1,3,5,10", help="comma-separated seconds, e.g. 1,3,5,10")
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--anomaly-ratio-threshold", type=float, default=0.15)
    p.add_argument("--unsup-train-fraction", type=float, default=0.6)
    p.add_argument("--unsup-calib-fraction", type=float, default=0.2)

    # Output paths are requested as `attack/...` when running from the parent directory.
    out_default_csv = (ROOT.parent / "attack" / "swat_granularity_statistics.csv").resolve()
    out_default_md = (ROOT.parent / "attack" / "swat_granularity_statistics.md").resolve()
    p.add_argument("--out-csv", default=str(out_default_csv))
    p.add_argument("--out-md", default=str(out_default_md))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = (ROOT / args.data_dir).resolve()

    window_size = int(args.window_size)
    stride = int(args.stride)
    anomaly_ratio_threshold = float(args.anomaly_ratio_threshold)
    train_fraction = float(args.unsup_train_fraction)
    calib_fraction = float(args.unsup_calib_fraction)

    granularities: list[int] = []
    for token in str(args.granularities).split(","):
        token = token.strip()
        if not token:
            continue
        granularities.append(int(token))
    granularities = sorted(dict.fromkeys(granularities))

    all_issues: list[str] = []
    all_extracted: list[Path] = []
    rows: list[dict[str, Any]] = []

    for g in granularities:
        inputs = locate_swat_files(data_dir, g)
        all_extracted.extend(inputs.extracted_paths)
        res, issues = compute_granularity_stats(
            inputs,
            window_size=window_size,
            stride=stride,
            anomaly_ratio_threshold=anomaly_ratio_threshold,
            train_fraction=train_fraction,
            calib_fraction=calib_fraction,
        )
        all_issues.extend(issues)

        if res is None:
            # Still output a row so the table clearly shows the missing granularity.
            rows.append(
                {
                    "Granularity": inputs.granularity,
                    "Benign Records": "-",
                    "Attack Records": "-",
                    "Train Records": "-",
                    "Calibration Records": "-",
                    "Test Benign Records": "-",
                    "Test Attack Records": "-",
                    "Train Windows": "-",
                    "Calibration Windows": "-",
                    "Test Benign Windows": "-",
                    "Test Anomalous Windows": "-",
                    "Total Test Windows": "-",
                    "Window Size": window_size,
                    "Stride": stride,
                    "Anomaly Ratio Threshold": anomaly_ratio_threshold,
                }
            )
            continue

        rows.append(
            {
                "Granularity": res.granularity,
                "Benign Records": res.benign_records,
                "Attack Records": res.attack_records,
                "Train Records": res.train_records,
                "Calibration Records": res.calib_records,
                "Test Benign Records": res.test_benign_records,
                "Test Attack Records": res.test_attack_records,
                "Train Windows": res.train_windows,
                "Calibration Windows": res.calib_windows,
                "Test Benign Windows": res.test_benign_windows,
                "Test Anomalous Windows": res.test_anom_windows,
                "Total Test Windows": res.test_total_windows,
                "Window Size": res.window_size,
                "Stride": res.stride,
                "Anomaly Ratio Threshold": res.anomaly_ratio_threshold,
            }
        )

    out_csv = Path(args.out_csv).resolve()
    out_md = Path(args.out_md).resolve()
    write_csv(rows, out_csv)
    write_md(rows, out_md, issues=sorted(dict.fromkeys(all_issues)), extracted=sorted(set(all_extracted)))

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote MD: {out_md}")
    if all_extracted:
        print("Auto-extracted:")
        for p in sorted(set(all_extracted)):
            print(f"- {p}")


if __name__ == "__main__":
    main()
