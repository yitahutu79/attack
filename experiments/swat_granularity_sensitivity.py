#!/usr/bin/env python3
"""Lightweight SWaT granularity sensitivity experiment runner.

Runs per-granularity (1s/3s/5s/10s) experiments without mixing data:
- Proposed (current repository SWaT unsupervised formal entrypoint)
- OneClassSVM
- IsolationForest
- GANomaly
- TranAD

Then aggregates to:
- swat_granularity_results.csv
- swat_granularity_results.md
- swat_granularity_f1_plot.png
- swat_granularity_fpr_plot.png
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_loaders import load_windowed_unsupervised_split  # noqa: E402


MODEL_ORDER = ["Proposed", "IsolationForest", "OneClassSVM", "GANomaly", "TranAD"]
BASELINE_METHODS = ["iforest", "ocsvm", "ganomaly", "tranad"]
BASELINE_METHOD_NAME_MAP = {
    "IsolationForest": "IsolationForest",
    "OneClassSVM": "OneClassSVM",
    "GANomaly": "GANomaly",
    "TranAD": "TranAD",
}

FIELDNAMES = [
    "Granularity",
    "Model",
    "Train Windows",
    "Calib Windows",
    "Test Benign Windows",
    "Test Anomaly Windows",
    "AUC",
    "AP",
    "F1",
    "Precision",
    "Recall",
    "Observed FPR",
    "FP/Test Benign",
    "Threshold",
    "Target FPR",
    "Window Size",
    "Stride",
    "Anomaly Ratio Threshold",
    "Eval Seconds",
    "Windows/s",
]


@dataclass(frozen=True)
class GranularityPaths:
    granularity: str
    benign_csv: Path
    attack_csv: Path
    extracted_paths: list[Path]
    issues: list[str]


@dataclass(frozen=True)
class SplitInfo:
    train_windows: int
    calib_windows: int
    test_benign_windows: int
    test_anom_windows: int


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
    with tarfile.open(archive_path, mode="r:xz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and str(m.name).lower().endswith(".csv")]
        if not members:
            return [], [f"Archive has no .csv members: {archive_path}"]
        for member in members:
            rel = Path(member.name)
            if rel.is_absolute() or ".." in rel.parts:
                issues.append(f"Refusing suspicious tar member path: {member.name}")
                continue
            out_path = (dest_dir / rel).resolve()
            if dest_dir.resolve() not in out_path.parents and out_path != dest_dir.resolve():
                issues.append(f"Refusing to extract outside destination: {member.name}")
                continue
            if out_path.exists():
                extracted.append(out_path)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(member) as src, out_path.open("wb") as dst:
                if src is None:
                    issues.append(f"Failed to read tar member: {member.name}")
                    continue
                dst.write(src.read())
            extracted.append(out_path)
    return extracted, issues


def _locate_or_extract_csv(data_dir: Path, rel_csv: str) -> tuple[Path | None, list[Path], list[str]]:
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

    candidates = [p for p in extracted if p.name == Path(rel_csv).name]
    if not candidates:
        candidates = [p for p in extracted if p.suffix.lower() == ".csv"]
    if not candidates:
        return None, extracted_paths, issues + [f"Archive extracted but no CSV found to use: {archive_path}"]
    if len(candidates) > 1:
        issues.append(
            "Multiple extracted CSV candidates match; using first: {}".format(
                ", ".join(str(p) for p in candidates)
            )
        )
    return candidates[0], extracted_paths, issues


def locate_granularity_files(data_dir: Path, granularity_sec: int) -> GranularityPaths:
    gran = f"{granularity_sec}sec"
    issues: list[str] = []
    extracted_paths: list[Path] = []

    benign_rel = f"benign_data/benign_samples_{gran}.csv"
    attack_rel = f"attack_data/attack_samples_{gran}.csv"

    benign_csv, benign_extracted, benign_issues = _locate_or_extract_csv(data_dir, benign_rel)
    attack_csv, attack_extracted, attack_issues = _locate_or_extract_csv(data_dir, attack_rel)
    extracted_paths.extend(benign_extracted)
    extracted_paths.extend(attack_extracted)
    issues.extend(benign_issues)
    issues.extend(attack_issues)

    if benign_csv is None:
        issues.append(f"Missing benign file for granularity {gran}.")
    if attack_csv is None:
        issues.append(f"Missing attack file for granularity {gran}.")

    return GranularityPaths(
        granularity=gran,
        benign_csv=benign_csv or Path(),
        attack_csv=attack_csv or Path(),
        extracted_paths=sorted(set(extracted_paths)),
        issues=issues,
    )


def build_split_info(
    *,
    data_dir: Path,
    benign_csv: Path,
    attack_csv: Path,
    window_size: int,
    stride: int,
    anomaly_ratio: float,
    train_fraction: float,
    calib_fraction: float,
) -> SplitInfo:
    ds = load_windowed_unsupervised_split(
        dataset="swat",
        data_dir=str(data_dir),
        benign_file=str(benign_csv),
        attack_file=str(attack_csv),
        window_size=window_size,
        stride=stride,
        anomaly_ratio=anomaly_ratio,
        benign_train_fraction=train_fraction,
        benign_calib_fraction=calib_fraction,
        scaler="minmax",
        clip_minmax=True,
        progress=False,
    )
    y_test = np.asarray(ds.y_test, dtype=np.int64)
    return SplitInfo(
        train_windows=int(len(ds.x_train)),
        calib_windows=0 if ds.x_calib is None else int(len(ds.x_calib)),
        test_benign_windows=int(np.sum(y_test == 0)),
        test_anom_windows=int(np.sum(y_test == 1)),
    )


def run_command(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}; see log: {log_path}")


def parse_proposed_row(
    *,
    granularity: str,
    json_path: Path,
    split_info: SplitInfo,
    target_fpr: float,
    window_size: int,
    stride: int,
    anomaly_ratio: float,
) -> dict[str, Any]:
    d = json.loads(json_path.read_text(encoding="utf-8"))
    calibrated = d.get("calibrated", {})
    metrics = d.get("metrics", {})
    fp = int(calibrated.get("fp", 0))
    test_benign = max(int(split_info.test_benign_windows), 1)
    eval_seconds = float(d.get("timing", {}).get("total_seconds", float("nan")))
    n_test_windows = int(d.get("n_test_windows", split_info.test_benign_windows + split_info.test_anom_windows))
    windows_per_sec = float(n_test_windows / eval_seconds) if np.isfinite(eval_seconds) and eval_seconds > 0 else float("nan")
    return {
        "Granularity": granularity,
        "Model": "Proposed",
        "Train Windows": int(d.get("n_train_windows", split_info.train_windows)),
        "Calib Windows": int(d.get("n_calib_windows", split_info.calib_windows)),
        "Test Benign Windows": int(split_info.test_benign_windows),
        "Test Anomaly Windows": int(split_info.test_anom_windows),
        "AUC": float(metrics.get("auc", float("nan"))),
        "AP": float(metrics.get("ap", float("nan"))),
        "F1": float(calibrated.get("f1", float("nan"))),
        "Precision": float(calibrated.get("precision", float("nan"))),
        "Recall": float(calibrated.get("recall", float("nan"))),
        "Observed FPR": float(calibrated.get("test_benign_fpr", calibrated.get("fpr", float("nan")))),
        "FP/Test Benign": f"{fp}/{test_benign}",
        "Threshold": float(calibrated.get("threshold", float("nan"))),
        "Target FPR": target_fpr,
        "Window Size": window_size,
        "Stride": stride,
        "Anomaly Ratio Threshold": anomaly_ratio,
        "Eval Seconds": eval_seconds,
        "Windows/s": windows_per_sec,
    }


def parse_baseline_rows(
    *,
    granularity: str,
    csv_path: Path,
    split_info: SplitInfo,
    target_fpr: float,
    window_size: int,
    stride: int,
    anomaly_ratio: float,
) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    rows: list[dict[str, Any]] = []
    for method_name, source_name in BASELINE_METHOD_NAME_MAP.items():
        sub = df[df["method"] == source_name]
        if sub.empty:
            continue
        row = sub.iloc[0]
        obs_fpr = float(row.get("test_benign_fpr", float("nan")))
        test_benign = max(int(split_info.test_benign_windows), 1)
        fp = int(round(obs_fpr * float(test_benign))) if np.isfinite(obs_fpr) else -1
        eval_seconds = float(row.get("eval_seconds", float("nan")))
        n_test_windows = int(row.get("n_test_windows", split_info.test_benign_windows + split_info.test_anom_windows))
        windows_per_sec = float(n_test_windows / eval_seconds) if np.isfinite(eval_seconds) and eval_seconds > 0 else float("nan")
        rows.append(
            {
                "Granularity": granularity,
                "Model": method_name,
                "Train Windows": int(row.get("n_train_windows", split_info.train_windows)),
                "Calib Windows": int(split_info.calib_windows),
                "Test Benign Windows": int(split_info.test_benign_windows),
                "Test Anomaly Windows": int(split_info.test_anom_windows),
                "AUC": float(row.get("auc", float("nan"))),
                "AP": float(row.get("ap", float("nan"))),
                "F1": float(row.get("calib_f1", float("nan"))),
                "Precision": float(row.get("calib_precision", float("nan"))),
                "Recall": float(row.get("calib_recall", float("nan"))),
                "Observed FPR": obs_fpr,
                "FP/Test Benign": f"{fp}/{test_benign}" if fp >= 0 else f"nan/{test_benign}",
                "Threshold": float(row.get("calib_threshold", float("nan"))),
                "Target FPR": target_fpr,
                "Window Size": window_size,
                "Stride": stride,
                "Anomaly Ratio Threshold": anomaly_ratio,
                "Eval Seconds": eval_seconds,
                "Windows/s": windows_per_sec,
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, Any]], out_md: Path, issues: list[str], extracted: list[Path]) -> None:
    lines: list[str] = []
    lines.append("# SWaT Granularity Sensitivity Results")
    lines.append("")
    lines.append(
        "Protocol: per-granularity SWaT formal unsupervised split; threshold calibrated only on benign calibration windows; "
        "target FPR=0.05; window label rule is attack_ratio >= anomaly_ratio_threshold."
    )
    lines.append("")
    headers = [
        "Granularity",
        "Model",
        "Train Windows",
        "Calib Windows",
        "Test Benign Windows",
        "Test Anomaly Windows",
        "AUC",
        "AP",
        "F1",
        "Precision",
        "Recall",
        "Observed FPR",
        "FP/Test Benign",
        "Threshold",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["Granularity"]),
                    str(row["Model"]),
                    str(row["Train Windows"]),
                    str(row["Calib Windows"]),
                    str(row["Test Benign Windows"]),
                    str(row["Test Anomaly Windows"]),
                    f"{float(row['AUC']):.6f}" if np.isfinite(float(row["AUC"])) else "nan",
                    f"{float(row['AP']):.6f}" if np.isfinite(float(row["AP"])) else "nan",
                    f"{float(row['F1']):.6f}" if np.isfinite(float(row["F1"])) else "nan",
                    f"{float(row['Precision']):.6f}" if np.isfinite(float(row["Precision"])) else "nan",
                    f"{float(row['Recall']):.6f}" if np.isfinite(float(row["Recall"])) else "nan",
                    f"{float(row['Observed FPR']):.6f}" if np.isfinite(float(row["Observed FPR"])) else "nan",
                    str(row["FP/Test Benign"]),
                    f"{float(row['Threshold']):.6f}" if np.isfinite(float(row["Threshold"])) else "nan",
                ]
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
        lines.append("Issues:")
        for msg in issues:
            lines.append(f"- {msg}")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_plot_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["GranularitySec"] = df["Granularity"].str.replace("sec", "", regex=False).astype(int)
    return df.sort_values(["Model", "GranularitySec"]).reset_index(drop=True)


def plot_metric(
    df: pd.DataFrame,
    *,
    metric_col: str,
    ylabel: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8.5, 5.2))
    for model in MODEL_ORDER:
        sub = df[df["Model"] == model].sort_values("GranularitySec")
        if sub.empty:
            continue
        x = sub["GranularitySec"].to_numpy()
        y = sub[metric_col].astype(float).to_numpy()
        plt.plot(x, y, marker="o", linewidth=2.0, label=model)
    plt.xlabel("Granularity (sec)")
    plt.ylabel(ylabel)
    plt.title(f"SWaT Granularity Sensitivity: {ylabel}")
    plt.xticks([1, 3, 5, 10], ["1sec", "3sec", "5sec", "10sec"])
    plt.grid(True, alpha=0.25)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SWaT granularity lightweight sensitivity experiment.")
    p.add_argument("--python", default="/Users/lijie/miniforge3/envs/attack/bin/python")
    p.add_argument("--data-dir", default="dataset/SWaT")
    p.add_argument("--granularities", default="1,3,5,10")
    p.add_argument(
        "--proposed-granularities",
        default="",
        help="Comma-separated granularities for Proposed only; empty means same as --granularities.",
    )
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--anomaly-ratio-threshold", type=float, default=0.15)
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--unsup-train-fraction", type=float, default=0.6)
    p.add_argument("--unsup-calib-fraction", type=float, default=0.2)
    p.add_argument("--proposed-epochs", type=int, default=6)
    p.add_argument("--baseline-epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--run-root", default=str((ROOT / "results" / "swat_granularity_sensitivity").resolve()))
    p.add_argument("--quick", action="store_true", help="Only run Proposed + OneClassSVM + GANomaly.")
    p.add_argument("--skip-proposed", action="store_true")
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--out-csv", default=str((ROOT / "swat_granularity_results.csv").resolve()))
    p.add_argument("--out-md", default=str((ROOT / "swat_granularity_results.md").resolve()))
    p.add_argument("--f1-plot", default=str((ROOT / "swat_granularity_f1_plot.png").resolve()))
    p.add_argument("--fpr-plot", default=str((ROOT / "swat_granularity_fpr_plot.png").resolve()))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    py = str(args.python)
    data_dir = (ROOT / args.data_dir).resolve()
    run_root = Path(args.run_root).resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    granularities = sorted({int(x.strip()) for x in str(args.granularities).split(",") if x.strip()})
    proposed_granularities = (
        sorted({int(x.strip()) for x in str(args.proposed_granularities).split(",") if x.strip()})
        if str(args.proposed_granularities).strip()
        else list(granularities)
    )
    methods = BASELINE_METHODS.copy()
    if args.quick:
        methods = ["ocsvm", "ganomaly"]

    window_size = int(args.window_size)
    stride = int(args.stride)
    anomaly_ratio = float(args.anomaly_ratio_threshold)
    target_fpr = float(args.target_fpr)
    train_fraction = float(args.unsup_train_fraction)
    calib_fraction = float(args.unsup_calib_fraction)

    all_rows: list[dict[str, Any]] = []
    all_issues: list[str] = []
    all_extracted: list[Path] = []

    for g in granularities:
        g_paths = locate_granularity_files(data_dir, g)
        all_issues.extend(g_paths.issues)
        all_extracted.extend(g_paths.extracted_paths)
        gran = g_paths.granularity
        if not g_paths.benign_csv.exists() or not g_paths.attack_csv.exists():
            all_issues.append(f"Skip {gran}: missing benign/attack CSV.")
            continue

        print(f"[granularity={gran}] building split info...", flush=True)
        split_info = build_split_info(
            data_dir=data_dir,
            benign_csv=g_paths.benign_csv,
            attack_csv=g_paths.attack_csv,
            window_size=window_size,
            stride=stride,
            anomaly_ratio=anomaly_ratio,
            train_fraction=train_fraction,
            calib_fraction=calib_fraction,
        )

        gran_dir = run_root / gran
        gran_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_baselines and methods:
            print(f"[granularity={gran}] running baselines: {' '.join(methods)}", flush=True)
            baseline_out = gran_dir / "baselines"
            baseline_log = gran_dir / "baselines.log"
            cmd = [
                py,
                str(ROOT / "baselines" / "window_baselines.py"),
                "--dataset",
                "swat",
                "--data-dir",
                str(data_dir),
                "--unsupervised-formal-split",
                "--unsup-benign-file",
                str(g_paths.benign_csv),
                "--unsup-attack-file",
                str(g_paths.attack_csv),
                "--unsup-train-fraction",
                str(train_fraction),
                "--unsup-calib-fraction",
                str(calib_fraction),
                "--window-size",
                str(window_size),
                "--stride",
                str(stride),
                "--anomaly-ratio",
                str(anomaly_ratio),
                "--target-fpr",
                str(target_fpr),
                "--methods",
                *methods,
                "--device",
                "cpu",
                "--epochs",
                str(int(args.baseline_epochs)),
                "--batch-size",
                str(int(args.batch_size)),
                "--output-dir",
                str(baseline_out),
            ]
            run_command(cmd, baseline_log)
            baseline_csv = baseline_out / "baseline_results.csv"
            if not baseline_csv.exists():
                all_issues.append(f"Missing baseline_results.csv for {gran}: {baseline_csv}")
            else:
                all_rows.extend(
                    parse_baseline_rows(
                        granularity=gran,
                        csv_path=baseline_csv,
                        split_info=split_info,
                        target_fpr=target_fpr,
                        window_size=window_size,
                        stride=stride,
                        anomaly_ratio=anomaly_ratio,
                    )
                )

        if not args.skip_proposed and g in proposed_granularities:
            print(f"[granularity={gran}] running Proposed...", flush=True)
            proposed_json = gran_dir / "ours.json"
            proposed_log = gran_dir / "ours.log"
            cmd = [
                py,
                str(ROOT / "pipelines" / "run_tcn_cross_dataset_minimal.py"),
                "--dataset",
                "swat",
                "--data-dir",
                str(data_dir),
                "--unsupervised-formal-split",
                "--unsup-benign-file",
                str(g_paths.benign_csv),
                "--unsup-attack-file",
                str(g_paths.attack_csv),
                "--unsup-train-fraction",
                str(train_fraction),
                "--unsup-calib-fraction",
                str(calib_fraction),
                "--window-size",
                str(window_size),
                "--stride",
                str(stride),
                "--anomaly-ratio",
                str(anomaly_ratio),
                "--epochs",
                str(int(args.proposed_epochs)),
                "--batch-size",
                str(int(args.batch_size)),
                "--test-batch-size",
                str(int(args.batch_size)),
                "--target-fpr",
                str(target_fpr),
                "--disc-pooling",
                "attn",
                "--score-mode",
                "fused",
                "--score-alpha",
                "0.24",
                "--save-best",
                str(gran_dir / "ours_ckpt.pt"),
                "--out-json",
                str(proposed_json),
            ]
            run_command(cmd, proposed_log)
            if not proposed_json.exists():
                all_issues.append(f"Missing ours.json for {gran}: {proposed_json}")
            else:
                all_rows.append(
                    parse_proposed_row(
                        granularity=gran,
                        json_path=proposed_json,
                        split_info=split_info,
                        target_fpr=target_fpr,
                        window_size=window_size,
                        stride=stride,
                        anomaly_ratio=anomaly_ratio,
                    )
                )
        elif not args.skip_proposed:
            print(f"[granularity={gran}] skip Proposed (not in --proposed-granularities).", flush=True)

    # Sort rows for stable table.
    model_rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    all_rows.sort(
        key=lambda r: (
            int(str(r["Granularity"]).replace("sec", "")),
            model_rank.get(str(r["Model"]), 999),
            str(r["Model"]),
        )
    )

    out_csv = Path(args.out_csv).resolve()
    out_md = Path(args.out_md).resolve()
    f1_plot = Path(args.f1_plot).resolve()
    fpr_plot = Path(args.fpr_plot).resolve()
    write_csv(all_rows, out_csv)
    write_md(out_rows := all_rows, out_md, sorted(set(all_issues)), sorted(set(all_extracted)))

    if out_rows:
        plot_df = _prepare_plot_frame(out_rows)
        plot_metric(plot_df, metric_col="F1", ylabel="F1", out_path=f1_plot)
        plot_metric(plot_df, metric_col="Observed FPR", ylabel="Observed Test Benign FPR", out_path=fpr_plot)

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote MD: {out_md}")
    print(f"Wrote F1 plot: {f1_plot}")
    print(f"Wrote FPR plot: {fpr_plot}")
    print(f"Run dir: {run_root}")
    if all_extracted:
        print("Auto-extracted:")
        for p in sorted(set(all_extracted)):
            print(f"- {p}")


if __name__ == "__main__":
    main()
