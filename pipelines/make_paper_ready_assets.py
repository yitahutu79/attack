#!/usr/bin/env python3
"""Collect final experiment results into the paper asset package."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _safe_float(v: object) -> float:
    try:
        if v is None or str(v).strip() == "":
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def _fmt(v: object, nd: int = 4) -> str:
    x = _safe_float(v)
    if math.isnan(x):
        return ""
    return f"{x:.{nd}f}"


def _write_md(path: Path, rows: list[dict[str, object]], fields: list[str], title: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if title:
        lines.extend([f"# {title}", ""])
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        parts: list[str] = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                parts.append(_fmt(value))
            else:
                parts.append(str(value))
        lines.append("| " + " | ".join(parts) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _method_type(method: str) -> str:
    if method == "TCN-GAN":
        return "Ours"
    if method in {"IsolationForest", "OneClassSVM"}:
        return "Unsupervised"
    if method in {"RF (Window)", "MLP (Window)"}:
        return "Supervised"
    if method in {"LSTM-AE", "LSTM-AD"}:
        return "Deep"
    return ""


def _load_fpr_results(baseline_csv: Path, tcn_csv: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in _read_csv(baseline_csv):
        rows.append(
            {
                "method": row["method"],
                "type": _method_type(row["method"]),
                "target_fpr": _safe_float(row.get("target_fpr")),
                "auc": _safe_float(row.get("auc")),
                "ap": _safe_float(row.get("ap")),
                "f1": _safe_float(row.get("calib_f1")),
                "recall": _safe_float(row.get("calib_recall")),
                "precision": _safe_float(row.get("calib_precision")),
                "test_fpr": _safe_float(row.get("test_benign_fpr")),
                "eval_seconds": _safe_float(row.get("eval_seconds")),
            }
        )
    for row in _read_csv(tcn_csv):
        rows.append(
            {
                "method": "TCN-GAN",
                "type": "Ours",
                "target_fpr": _safe_float(row.get("target_fpr")),
                "auc": _safe_float(row.get("auc")),
                "ap": _safe_float(row.get("ap")),
                "f1": _safe_float(row.get("calib_f1")),
                "recall": _safe_float(row.get("calib_recall")),
                "precision": _safe_float(row.get("calib_precision")),
                "test_fpr": _safe_float(row.get("test_benign_fpr")),
                "eval_seconds": _safe_float(row.get("eval_seconds")),
            }
        )
    return rows


def _filter_target(rows: list[dict[str, object]], target: float) -> list[dict[str, object]]:
    out = [r for r in rows if abs(float(r["target_fpr"]) - target) < 1e-9]
    return sorted(out, key=lambda r: float(r["f1"]), reverse=True)


def _load_seed_rows(seed_csv: Path, tcn_csv: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    # Add seed=42 from final TCN FPR sweep.
    for row in _read_csv(tcn_csv):
        target = _safe_float(row.get("target_fpr"))
        if target in {0.05, 0.15}:
            rows.append(
                {
                    "seed": 42,
                    "target_fpr": target,
                    "auc": _safe_float(row.get("auc")),
                    "ap": _safe_float(row.get("ap")),
                    "f1": _safe_float(row.get("calib_f1")),
                    "recall": _safe_float(row.get("calib_recall")),
                    "precision": _safe_float(row.get("calib_precision")),
                    "test_fpr": _safe_float(row.get("test_benign_fpr")),
                }
            )
    for row in _read_csv(seed_csv):
        rows.append(
            {
                "seed": int(float(row["seed"])),
                "target_fpr": _safe_float(row.get("target_fpr")),
                "auc": _safe_float(row.get("auc")),
                "ap": _safe_float(row.get("ap")),
                "f1": _safe_float(row.get("calib_f1")),
                "recall": _safe_float(row.get("calib_recall")),
                "precision": _safe_float(row.get("calib_precision")),
                "test_fpr": _safe_float(row.get("test_benign_fpr")),
            }
        )
    return sorted(rows, key=lambda r: (float(r["target_fpr"]), int(r["seed"])))


def _mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def _seed_stats(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for target in sorted({float(r["target_fpr"]) for r in seed_rows}):
        sub = [r for r in seed_rows if abs(float(r["target_fpr"]) - target) < 1e-9]
        row: dict[str, object] = {"target_fpr": target, "n_seeds": len(sub)}
        for metric in ["auc", "ap", "f1", "recall", "precision", "test_fpr"]:
            mean, std = _mean_std([float(r[metric]) for r in sub])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_mean_std"] = f"{mean:.4f} ± {std:.4f}"
        out.append(row)
    return out


def _make_plots(out_dir: Path, all_rows: list[dict[str, object]], seed_rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    methods = ["TCN-GAN", "MLP (Window)", "RF (Window)", "OneClassSVM", "IsolationForest"]
    colors = {
        "TCN-GAN": "#d62728",
        "MLP (Window)": "#1f77b4",
        "RF (Window)": "#2ca02c",
        "OneClassSVM": "#9467bd",
        "IsolationForest": "#7f7f7f",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    for method in methods:
        sub = sorted([r for r in all_rows if r["method"] == method], key=lambda r: float(r["target_fpr"]))
        if not sub:
            continue
        xs = [float(r["target_fpr"]) for r in sub]
        axes[0].plot(xs, [float(r["f1"]) for r in sub], marker="o", label=method, color=colors.get(method))
        axes[1].plot(xs, [float(r["recall"]) for r in sub], marker="o", label=method, color=colors.get(method))
    axes[0].set_title("F1 Across FPR Budgets")
    axes[0].set_ylabel("F1")
    axes[1].set_title("Recall Across FPR Budgets")
    axes[1].set_ylabel("Recall")
    for ax in axes:
        ax.set_xlabel("Target FPR")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.02)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fpr_sweep_f1_recall.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    targets = [0.05, 0.15]
    width = 0.35
    seeds = sorted({int(r["seed"]) for r in seed_rows})
    for i, target in enumerate(targets):
        vals = []
        for seed in seeds:
            match = [r for r in seed_rows if int(r["seed"]) == seed and abs(float(r["target_fpr"]) - target) < 1e-9]
            vals.append(float(match[0]["f1"]) if match else float("nan"))
        offset = (i - 0.5) * width
        ax.bar([j + offset for j in range(len(seeds))], vals, width=width, label=f"target_fpr={target:.2f}")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel("F1")
    ax.set_title("TCN-GAN Seed Variance")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "seed_f1_bar.png", dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create paper/report result tables and figures.")
    ap.add_argument("--out-dir", default="attack/paper")
    ap.add_argument("--baseline-csv", default="attack/results/baseline_fpr_sweep/20260415_020742/baseline_results_with_target_fpr.csv")
    ap.add_argument("--tcn-csv", default="attack/results/final_experiments/20260414_132416/tcn_fpr_sweep/tcn_gan_fpr_sweep.csv")
    ap.add_argument("--seed-csv", default="attack/results/final_experiments/20260414_132416/seed_runs/seed_summary.csv")
    ap.add_argument("--ablation-csv", default="attack/results/final_experiments/20260414_132416/ablation_2x2/20260414_204232/fpr_sweep/ablation_fpr_sweep.csv")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    tables = out_dir / "tables"
    figures = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    all_rows = _load_fpr_results(Path(args.baseline_csv), Path(args.tcn_csv))
    fields = ["method", "type", "target_fpr", "auc", "ap", "f1", "recall", "precision", "test_fpr", "eval_seconds"]
    _write_csv(tables / "fpr_sweep_all_methods.csv", all_rows, fields)

    for target in [0.05, 0.15]:
        rows = _filter_target(all_rows, target)
        stem = f"main_comparison_target_fpr_{str(target).replace('.', 'p')}"
        _write_csv(tables / f"{stem}.csv", rows, fields)
        _write_md(tables / f"{stem}.md", rows, fields, title=f"Main Comparison Target FPR={target:.2f}")

    seed_rows = _load_seed_rows(Path(args.seed_csv), Path(args.tcn_csv))
    seed_fields = ["seed", "target_fpr", "auc", "ap", "f1", "recall", "precision", "test_fpr"]
    _write_csv(tables / "seed_results_5seeds.csv", seed_rows, seed_fields)
    _write_md(tables / "seed_results_5seeds.md", seed_rows, seed_fields, title="TCN-GAN Seed Results")

    stats_rows = _seed_stats(seed_rows)
    stats_fields = [
        "target_fpr",
        "n_seeds",
        "auc_mean_std",
        "ap_mean_std",
        "f1_mean_std",
        "recall_mean_std",
        "precision_mean_std",
        "test_fpr_mean_std",
    ]
    _write_csv(tables / "seed_stats_mean_std.csv", stats_rows, stats_fields)
    _write_md(tables / "seed_stats_mean_std.md", stats_rows, stats_fields, title="TCN-GAN Seed Mean Std")

    ablation_path = Path(args.ablation_csv)
    if ablation_path.exists():
        ablation_rows = _read_csv(ablation_path)
        shutil.copy2(ablation_path, tables / "ablation_fpr_sweep.csv")
        for target in [0.05, 0.15]:
            rows = [r for r in ablation_rows if abs(_safe_float(r.get("target_fpr")) - target) < 1e-9]
            rows = sorted(rows, key=lambda r: _safe_float(r.get("calib_f1")), reverse=True)
            cols = ["disc_pooling", "gan_loss", "target_fpr", "auc", "ap", "calib_f1", "calib_recall", "calib_precision", "test_benign_fpr"]
            stem = f"ablation_target_fpr_{str(target).replace('.', 'p')}"
            _write_csv(tables / f"{stem}.csv", rows, cols)
            _write_md(tables / f"{stem}.md", rows, cols, title=f"Ablation Target FPR={target:.2f}")

    _make_plots(out_dir, all_rows, seed_rows)

    copied = [
        ("attack/results/final_experiments/20260414_132416/alpha_sweep/alpha_sweep.png", figures / "alpha_sweep.png"),
        ("attack/results/final_experiments/20260414_132416/threshold_curve/threshold_tradeoff.png", figures / "threshold_tradeoff.png"),
        ("attack/results/xai_tcn/eval_w128_s16_fused_w128_s16_fused_a0.24/xai_time_importance.png", figures / "xai_time_importance.png"),
        ("attack/results/xai_tcn/eval_w128_s16_fused_w128_s16_fused_a0.24/xai_feature_importance.png", figures / "xai_feature_importance.png"),
        ("attack/results/xai_tcn/eval_w128_s16_fused_w128_s16_fused_a0.24/attn_weights.png", figures / "attn_weights.png"),
        ("attack/paper/figures/tcn_gan_paper_framework.png", figures / "tcn_gan_paper_framework.png"),
    ]
    for src, dst in copied:
        p = Path(src)
        if p.exists() and p.resolve() != dst.resolve():
            shutil.copy2(p, dst)

    summary = {
        "source_files": {
            "baseline_fpr_sweep": str(args.baseline_csv),
            "tcn_fpr_sweep": str(args.tcn_csv),
            "seed_summary": str(args.seed_csv),
            "ablation_fpr_sweep": str(args.ablation_csv),
        },
        "recommended_tables": [
            "tables/main_comparison_target_fpr_0p05.csv",
            "tables/main_comparison_target_fpr_0p15.csv",
            "tables/seed_stats_mean_std.csv",
            "tables/ablation_target_fpr_0p05.csv",
            "tables/ablation_target_fpr_0p15.csv",
        ],
        "recommended_figures": [
            "figures/fpr_sweep_f1_recall.png",
            "figures/seed_f1_bar.png",
            "figures/alpha_sweep.png",
            "figures/threshold_tradeoff.png",
            "figures/xai_feature_importance.png",
            "figures/xai_time_importance.png",
            "figures/attn_weights.png",
        ],
    }
    (out_dir / "ASSETS.md").write_text(
        "# Paper/Report Assets\n\n"
        "This file is generated by `attack/pipelines/make_paper_ready_assets.py`.\n"
        "The current paper/report tables and figures live under this directory.\n\n"
        "## Recommended Tables\n\n"
        + "\n".join(f"- `{p}`" for p in summary["recommended_tables"])
        + "\n\n## Recommended Figures\n\n"
        + "\n".join(f"- `{p}`" for p in summary["recommended_figures"])
        + "\n\n## Source Files\n\n"
        + "\n".join(f"- `{k}`: `{v}`" for k, v in summary["source_files"].items())
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "asset_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Paper/report assets written to: {out_dir}")


if __name__ == "__main__":
    main()
