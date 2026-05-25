#!/usr/bin/env python3
"""Create a clean 4-panel main-paper figure from existing V2 case-trace outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v2-dir", default="results/cicids_strict_mps/xai_case_trace_e8_v2")
    p.add_argument("--v1-dir", default="results/cicids_strict_mps/xai_case_trace_e8")
    return p.parse_args()


def build_heatmap(ig_tf_df: pd.DataFrame, ordered_features: list[str]) -> tuple[np.ndarray, float]:
    m = ig_tf_df.copy()
    m["time_step"] = m["time_step"].astype(int)
    pivot = (
        m[m["feature_name"].isin(ordered_features)]
        .pivot_table(index="feature_name", columns="time_step", values="abs_ig_attribution", aggfunc="mean")
        .reindex(index=ordered_features)
        .reindex(columns=list(range(128)))
        .fillna(0.0)
    )
    raw = pivot.to_numpy(dtype=np.float64)
    logv = np.log1p(raw)
    vmax = float(np.percentile(logv, 99))
    if vmax <= 0:
        vmax = float(np.max(logv)) if float(np.max(logv)) > 0 else 1e-6
    return np.clip(logv, 0.0, vmax), vmax


def main() -> None:
    args = parse_args()
    v2 = Path(args.v2_dir).resolve()
    v1 = Path(args.v1_dir).resolve()

    nonred = pd.read_csv(v2 / "case_top_features_nonredundant_e8_v2.csv").sort_values("nonredundant_rank")
    ordered_features = nonred["feature_name"].astype(str).tolist()

    ig_tf_df = pd.read_csv(v1 / "case_ig_time_feature_matrix_e8.csv")
    masking = pd.read_csv(v1 / "case_masking_verification_e8.csv").sort_values("k")
    score_df = pd.read_csv(v1 / "case_score_components_e8.csv")
    meta = json.loads((v1 / "case_metadata_e8.json").read_text(encoding="utf-8"))

    score_map = {str(r["component"]): float(r["value"]) for _, r in score_df.iterrows()}
    threshold = float(meta["threshold"])
    fused_score = float(meta["fused_score"])
    benign_ref = score_map.get("benign_reference_score")

    hm, vmax = build_heatmap(ig_tf_df, ordered_features)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(10.8, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[1.0, 1.0])

    # A. Alarm score
    ax_a = fig.add_subplot(gs[0, 0])
    labels = ["Threshold", "Case fused\\nscore"]
    vals = [threshold, fused_score]
    colors = ["#9e9e9e", "#1f77b4"]
    if benign_ref is not None:
        labels.append("Benign ref.\\nmean")
        vals.append(float(benign_ref))
        colors.append("#2ca02c")
    # Use explicit x positions with wider spacing so long labels do not collide
    x_pos = np.arange(len(vals), dtype=float) * 1.6
    ax_a.bar(x_pos, vals, color=colors, width=0.9)
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(labels)
    ax_a.set_title("A. Alarm score")
    ax_a.set_ylabel("Score")
    ax_a.tick_params(axis="x", labelrotation=12, labelsize=9)
    ax_a.margins(x=0.22)
    ax_a.grid(axis="y", alpha=0.35)
    for i, v in enumerate(vals):
        ax_a.text(x_pos[i], v + 0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    # B. Non-redundant IG heatmap
    ax_b = fig.add_subplot(gs[0, 1])
    im = ax_b.imshow(
        hm,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )
    ax_b.set_title("B. Non-redundant IG heatmap")
    ax_b.set_xlabel("Time step")
    ax_b.set_ylabel("Feature")
    ax_b.set_xticks([0, 32, 64, 96, 127])
    ax_b.set_yticks(np.arange(len(ordered_features)))
    ax_b.set_yticklabels(ordered_features)
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.02)
    cbar.set_label("|IG| (log1p + p99 clip)")

    # C. Top non-redundant features
    ax_c = fig.add_subplot(gs[1, 0])
    feat_vals = nonred["mean_abs_ig_attribution"].astype(float).to_numpy()
    ax_c.barh(np.arange(len(ordered_features)), feat_vals, color="#2ca02c")
    ax_c.set_yticks(np.arange(len(ordered_features)))
    ax_c.set_yticklabels(ordered_features)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Mean |IG|")
    ax_c.set_title("C. Top non-redundant features")
    ax_c.grid(axis="x", alpha=0.35)

    # D. Masking verification
    ax_d = fig.add_subplot(gs[1, 1])
    x_labels = ["Original"]
    x_vals = [fused_score]
    x_colors = ["#1f77b4"]
    for _, r in masking.iterrows():
        k = int(r["k"])
        x_labels.extend([f"IG@{k}", f"Rand@{k}"])
        x_vals.extend([float(r["ig_masked_fused_score"]), float(r["random_masked_fused_score_mean"])])
        x_colors.extend(["#ff7f0e", "#7f7f7f"])
    ax_d.bar(np.arange(len(x_vals)), x_vals, color=x_colors)
    ax_d.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.5, label="Threshold")
    ax_d.set_xticks(np.arange(len(x_vals)))
    ax_d.set_xticklabels(x_labels, rotation=20, ha="right")
    ax_d.set_ylabel("Fused score")
    ax_d.set_title("D. Masking verification")
    ax_d.grid(axis="y", alpha=0.35)
    ax_d.legend(frameon=False, loc="upper left")

    out_png = v2 / "case_trace_plot_e8_v2_mainpaper_4panel.png"
    out_pdf = v2 / "case_trace_plot_e8_v2_mainpaper_4panel.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()
