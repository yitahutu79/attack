#!/usr/bin/env python3
"""Build curated cross-dataset result tables for the paper."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/Users/lijie/Desktop/work/attack")
TABLES = ROOT / "paper" / "tables"


def read_csv_row(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as f:
        return next(csv.DictReader(f))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def row_from_tcn(path: Path, dataset_label: str) -> dict[str, object]:
    d = read_json(path)
    c = d["calibrated"]
    m = d["metrics"]
    return {
        "dataset": dataset_label,
        "method": "TCN-GAN",
        "type": "Ours",
        "target_fpr": float(c["target_fpr"]),
        "auc": float(m["auc"]),
        "ap": float(m["ap"]),
        "f1": float(c["f1"]),
        "recall": float(c["recall"]),
        "precision": float(c["precision"]),
        "test_fpr": float(c["test_benign_fpr"]),
        "source": str(path.relative_to(ROOT.parent)),
    }


def row_from_baseline_csv(path: Path, dataset_label: str, type_label: str) -> dict[str, object]:
    r = read_csv_row(path)
    return {
        "dataset": dataset_label,
        "method": r["method"],
        "type": type_label,
        "target_fpr": float(r["target_fpr"]),
        "auc": float(r["auc"]),
        "ap": float(r["ap"]),
        "f1": float(r["calib_f1"]),
        "recall": float(r["calib_recall"]),
        "precision": float(r["calib_precision"]),
        "test_fpr": float(r["test_benign_fpr"]),
        "source": str(path.relative_to(ROOT.parent)),
    }


def write_csv_md(base: str, rows: list[dict[str, object]], title: str) -> None:
    csv_path = TABLES / f"{base}.csv"
    md_path = TABLES / f"{base}.md"
    fieldnames = ["dataset", "method", "type", "target_fpr", "auc", "ap", "f1", "recall", "precision", "test_fpr", "source"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| dataset | method | type | target_fpr | auc | ap | f1 | recall | precision | test_fpr |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for r in rows:
            f.write(
                "| {dataset} | {method} | {type} | {target_fpr:.4f} | {auc:.4f} | {ap:.4f} | {f1:.4f} | {recall:.4f} | {precision:.4f} | {test_fpr:.4f} |\n".format(
                    **r
                )
            )


def build() -> None:
    swat_rows = [
        row_from_tcn(ROOT / "results/ablation_swat_attn_prob/ours.json", "SWaT"),
        row_from_baseline_csv(ROOT / "results/cross_dataset_formal_unsup/swat_baselines/20260419_125812/baseline_results.csv", "SWaT", "Unsupervised/Deep"),
        row_from_baseline_csv(ROOT / "results/sota_tranad_swat/baseline_results.csv", "SWaT", "SOTA"),
        row_from_baseline_csv(ROOT / "results/sota_deepsvdd_swat/baseline_results.csv", "SWaT", "SOTA"),
    ]

    # expand multi-row baseline file
    expanded_swat: list[dict[str, object]] = [swat_rows[0]]
    with (ROOT / "results/cross_dataset_formal_unsup/swat_baselines/20260419_125812/baseline_results.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            expanded_swat.append(
                {
                    "dataset": "SWaT",
                    "method": r["method"],
                    "type": "Unsupervised/Deep",
                    "target_fpr": float(r["target_fpr"]),
                    "auc": float(r["auc"]),
                    "ap": float(r["ap"]),
                    "f1": float(r["calib_f1"]),
                    "recall": float(r["calib_recall"]),
                    "precision": float(r["calib_precision"]),
                    "test_fpr": float(r["test_benign_fpr"]),
                    "source": "attack/results/cross_dataset_formal_unsup/swat_baselines/20260419_125812/baseline_results.csv",
                }
            )
    expanded_swat.append(swat_rows[2])
    expanded_swat.append(swat_rows[3])

    toniot_rows = [
        row_from_tcn(ROOT / "results/ablation_toniot_attn_prob/ours.json", "TON_IoT"),
        row_from_baseline_csv(ROOT / "results/sota_tranad_toniot/baseline_results.csv", "TON_IoT", "SOTA"),
        row_from_baseline_csv(ROOT / "results/sota_deepsvdd_toniot/baseline_results.csv", "TON_IoT", "SOTA"),
    ]
    expanded_toniot: list[dict[str, object]] = [toniot_rows[0]]
    with (ROOT / "results/cross_dataset_formal_unsup/ton_iot_baselines/20260419_130847/baseline_results.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            expanded_toniot.append(
                {
                    "dataset": "TON_IoT",
                    "method": r["method"],
                    "type": "Unsupervised/Deep",
                    "target_fpr": float(r["target_fpr"]),
                    "auc": float(r["auc"]),
                    "ap": float(r["ap"]),
                    "f1": float(r["calib_f1"]),
                    "recall": float(r["calib_recall"]),
                    "precision": float(r["calib_precision"]),
                    "test_fpr": float(r["test_benign_fpr"]),
                    "source": "attack/results/cross_dataset_formal_unsup/ton_iot_baselines/20260419_130847/baseline_results.csv",
                }
            )
    expanded_toniot.append(toniot_rows[1])
    expanded_toniot.append(toniot_rows[2])

    def sort_key(r: dict[str, object]) -> tuple[float, float]:
        return (float(r["test_fpr"]), -float(r["f1"]))

    expanded_swat.sort(key=sort_key)
    expanded_toniot.sort(key=sort_key)
    combined = expanded_swat + expanded_toniot

    write_csv_md(
        "swat_formal_full_target_fpr_0p05",
        expanded_swat,
        "SWaT Formal Full Comparison @ FPR=0.05",
    )
    write_csv_md(
        "toniot_formal_full_target_fpr_0p05",
        expanded_toniot,
        "TON_IoT Formal Full Comparison @ FPR=0.05",
    )
    write_csv_md(
        "cross_dataset_formal_full_target_fpr_0p05",
        combined,
        "Cross-Dataset Formal Full Comparison @ FPR=0.05",
    )

    # efficiency summary
    eff_rows = []
    for dataset, path in [
        ("SWaT", ROOT / "results/ablation_swat_attn_prob/ours.json"),
        ("TON_IoT", ROOT / "results/ablation_toniot_attn_prob/ours.json"),
    ]:
        d = read_json(path)
        hist = d.get("history", [])
        ckpt = path.with_name("ours_ckpt.pt")
        eff_rows.append(
            {
                "dataset": dataset,
                "train_windows": d.get("n_train_windows", ""),
                "test_windows": d.get("n_test_windows", ""),
                "epochs": len(hist),
                "train_seconds": float(d["timing"]["total_seconds"]),
                "seconds_per_epoch": (sum(float(h["seconds"]) for h in hist) / max(len(hist), 1)) if hist else 0.0,
                "checkpoint_mb": (ckpt.stat().st_size / 1024 / 1024) if ckpt.exists() else 0.0,
                "source": str(path.relative_to(ROOT.parent)),
            }
        )
    eff_csv = TABLES / "efficiency_summary.csv"
    with eff_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(eff_rows[0].keys()))
        w.writeheader()
        w.writerows(eff_rows)


if __name__ == "__main__":
    build()
