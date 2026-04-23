#!/usr/bin/env python3
"""
Generate TCN-GAN ablation tables from eval JSONs.

This script is paper-oriented and avoids pandas (stdlib only).

Typical inputs:
- attack/results/final_experiments/20260414_132416/ablation_2x2/20260414_204232/*/eval_w*_s*_*.json

Outputs:
- Markdown + CSV tables that include `disc_pooling` and `gan_loss`,
  enabling "mean vs attn" and "vanilla vs wgan-gp" ablations.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _norm_path(v: object) -> object:
    """
    Normalize legacy paths that appear inside older JSONs / reports.
    We do NOT rewrite the JSON files; we only rewrite strings in the output tables.
    """
    if not isinstance(v, str):
        return v
    s = v
    s = s.replace("attack/tcn_gan_autotune_runs/", "attack/_archive/cleanup_20260415/legacy_results/tcn_gan_autotune_runs/")
    s = s.replace("attack/xai_tcn/", "attack/results/xai_tcn/")
    s = s.replace("attack/paper_results/", "attack/_archive/cleanup_20260413/legacy_results/paper_results/")
    s = s.replace("attack/paper_logs/", "attack/results/paper_logs/")
    return s


def _f(x: object, nd: int = 4) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return ""
        return f"{x:.{nd}f}"
    s = str(x).strip()
    return s


def _extract_row(path: str) -> dict[str, object]:
    d = _load_json(path)
    metrics = d.get("metrics", {}) if isinstance(d.get("metrics"), dict) else {}
    calib = d.get("calibrated", {}) if isinstance(d.get("calibrated"), dict) else {}
    xai = d.get("xai", {}) if isinstance(d.get("xai"), dict) else {}

    # Backward compat: some older JSONs don't include these keys
    disc_pooling = d.get("disc_pooling", d.get("discriminator_pooling", ""))
    gan_loss = d.get("gan_loss", "")

    row: dict[str, object] = {
        "json": _norm_path(str(path)),
        "window": d.get("window_size", ""),
        "stride": d.get("stride", ""),
        "score_mode": d.get("score_mode", ""),
        "score_alpha": d.get("score_alpha", ""),
        "disc_pooling": disc_pooling,
        "gan_loss": gan_loss,
        "auc": metrics.get("auc", ""),
        "ap": metrics.get("ap", ""),
        "best_f1": metrics.get("best_f1", ""),
        "best_precision": metrics.get("best_precision", ""),
        "best_recall": metrics.get("best_recall", ""),
        "best_fpr": metrics.get("best_fpr", ""),
        "calib_threshold": calib.get("threshold", ""),
        "calib_f1": calib.get("f1", ""),
        "calib_precision": calib.get("precision", ""),
        "calib_recall": calib.get("recall", ""),
        "calib_fpr": calib.get("fpr", ""),
        "train_benign_fpr": calib.get("train_benign_fpr", ""),
        "test_benign_fpr": calib.get("test_benign_fpr", ""),
        "xai_enabled": bool(xai.get("enabled", False)),
        "xai_time_plot": _norm_path(xai.get("plot_time_importance", "")),
        "xai_feat_plot": _norm_path(xai.get("plot_feature_importance", "")),
        "xai_attn_plot": _norm_path(xai.get("plot_attn_weights", "")),
    }
    return row


def _sort_key(row: dict[str, object], prefer: str) -> float:
    if prefer == "calib_f1":
        v = row.get("calib_f1")
    elif prefer == "calib_recall":
        v = row.get("calib_recall")
    else:
        v = row.get("auc")
    try:
        return float(v) if v is not None and str(v).strip() else float("nan")
    except Exception:
        return float("nan")


def _write_csv(path: Path, rows: list[dict[str, object]], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])


def _write_md(path: Path, rows: list[dict[str, object]], cols: list[str], fmt_cols: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in rows:
        parts: list[str] = []
        for c in cols:
            v = r.get(c, "")
            if c in fmt_cols:
                parts.append(_f(v, 4))
            else:
                parts.append(str(v) if v is not None else "")
        lines.append("| " + " | ".join(parts) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate TCN-GAN ablation tables from eval JSONs.")
    ap.add_argument(
        "--glob",
        dest="globs",
        action="append",
        default=None,
        help="glob pattern(s) for JSONs; can be passed multiple times",
    )
    ap.add_argument("--prefer", choices=["calib_f1", "calib_recall", "auc"], default="calib_f1")
    ap.add_argument("--out-md", default="attack/docs/paper_tcn_gan_ablation.md")
    ap.add_argument("--out-csv", default="attack/results/tables/paper_tcn_gan_ablation.csv")
    ap.add_argument("--out-md-summary", default="attack/docs/paper_tcn_gan_ablation_summary.md")
    ap.add_argument("--out-csv-summary", default="attack/results/tables/paper_tcn_gan_ablation_summary.csv")
    args = ap.parse_args()

    paths: list[str] = []
    globs = args.globs or [
        "attack/results/final_experiments/*/ablation_2x2/*/*/eval_w*_s*_*.json",
    ]
    for g in globs:
        paths.extend(sorted(glob.glob(g)))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit(
            "No JSONs found. Try passing --glob 'attack/results/final_experiments/*/ablation_2x2/*/*/eval_w*_s*_*.json'"
        )

    rows = [_extract_row(p) for p in paths]

    # Normalize empty fields
    for r in rows:
        if not r.get("disc_pooling"):
            r["disc_pooling"] = "mean"  # backward compat: default was mean pooling
        if not r.get("gan_loss"):
            r["gan_loss"] = "vanilla"  # backward compat: default was vanilla BCE
        if not r.get("score_mode"):
            r["score_mode"] = ""
        if not r.get("score_alpha"):
            r["score_alpha"] = ""

    rows_sorted = sorted(
        rows,
        key=lambda rr: (
            str(rr.get("gan_loss", "")),
            str(rr.get("disc_pooling", "")),
            str(rr.get("score_mode", "")),
            -_sort_key(rr, str(args.prefer)) if not math.isnan(_sort_key(rr, str(args.prefer))) else float("inf"),
            int(rr.get("window") or 0),
            int(rr.get("stride") or 0),
        ),
    )

    cols = [
        "disc_pooling",
        "gan_loss",
        "window",
        "stride",
        "score_mode",
        "score_alpha",
        "auc",
        "ap",
        "calib_f1",
        "calib_recall",
        "calib_precision",
        "test_benign_fpr",
        "json",
        "xai_time_plot",
        "xai_feat_plot",
        "xai_attn_plot",
    ]
    fmt_cols = {"auc", "ap", "calib_f1", "calib_recall", "calib_precision", "test_benign_fpr", "score_alpha"}

    _write_md(Path(args.out_md), rows_sorted, cols, fmt_cols=fmt_cols)
    _write_csv(Path(args.out_csv), rows_sorted, cols)

    # Summary: best per (disc_pooling, gan_loss)
    best: dict[tuple[str, str], dict[str, object]] = {}
    for r in rows_sorted:
        key = (str(r["disc_pooling"]), str(r["gan_loss"]))
        cur = best.get(key)
        if cur is None:
            best[key] = r
            continue
        a = _sort_key(r, str(args.prefer))
        b = _sort_key(cur, str(args.prefer))
        if math.isnan(b) or (not math.isnan(a) and a > b):
            best[key] = r

    summary_rows = list(best.values())
    summary_rows = sorted(summary_rows, key=lambda rr: (-_sort_key(rr, str(args.prefer)), str(rr["disc_pooling"]), str(rr["gan_loss"])))
    summary_cols = [
        "disc_pooling",
        "gan_loss",
        "window",
        "stride",
        "score_mode",
        "score_alpha",
        "auc",
        "ap",
        "calib_f1",
        "calib_recall",
        "calib_precision",
        "test_benign_fpr",
        "json",
    ]
    _write_md(Path(args.out_md_summary), summary_rows, summary_cols, fmt_cols=fmt_cols)
    _write_csv(Path(args.out_csv_summary), summary_rows, summary_cols)

    print(f"Wrote: {args.out_md}")
    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_md_summary}")
    print(f"Wrote: {args.out_csv_summary}")


if __name__ == "__main__":
    main()
