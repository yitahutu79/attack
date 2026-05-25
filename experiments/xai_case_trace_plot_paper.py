#!/usr/bin/env python3
"""Render paper-ready XAI case figures from existing case-trace outputs only."""

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
    p.add_argument(
        "--input-dir",
        default="results/cicids_strict_mps/xai_case_trace_e8",
        help="Directory containing existing case_trace CSV/JSON files.",
    )
    return p.parse_args()


def load_inputs(base: Path) -> dict[str, object]:
    with (base / "case_metadata_e8.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    score_df = pd.read_csv(base / "case_score_components_e8.csv")
    score_map = {str(r["component"]): float(r["value"]) for _, r in score_df.iterrows()}

    attn_path = base / "case_attention_time_weights_e8.csv"
    attn_df = pd.read_csv(attn_path) if attn_path.exists() else None

    ig_time_df = pd.read_csv(base / "case_ig_time_attribution_e8.csv")
    ig_feat_df = pd.read_csv(base / "case_ig_feature_attribution_e8.csv")
    ig_tf_df = pd.read_csv(base / "case_ig_time_feature_matrix_e8.csv")
    masking_df = pd.read_csv(base / "case_masking_verification_e8.csv")

    return {
        "meta": meta,
        "score_map": score_map,
        "attn_df": attn_df,
        "ig_time_df": ig_time_df,
        "ig_feat_df": ig_feat_df,
        "ig_tf_df": ig_tf_df,
        "masking_df": masking_df,
    }


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def build_top10_matrix(ig_feat_df: pd.DataFrame, ig_tf_df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    top10 = ig_feat_df.sort_values("rank", ascending=True).head(10).copy()
    top10["feature_index"] = top10["feature_index"].astype(int)
    top10_indices = top10["feature_index"].tolist()
    top10_names = top10["feature_name"].astype(str).tolist()

    m = ig_tf_df.copy()
    m["feature_index"] = m["feature_index"].astype(int)
    m["time_step"] = m["time_step"].astype(int)

    filt = m[m["feature_index"].isin(top10_indices)].copy()
    pivot = (
        filt.pivot_table(
            index="feature_index",
            columns="time_step",
            values="abs_ig_attribution",
            aggfunc="mean",
        )
        .reindex(index=top10_indices)
        .sort_index(axis=1)
    )
    heat = pivot.to_numpy(dtype=np.float64)

    idx_to_name = {int(r.feature_index): str(r.feature_name) for _, r in top10.iterrows()}
    ylabels = [idx_to_name[i] for i in top10_indices]
    return ylabels, heat


def overlap_top10(ig_time_df: pd.DataFrame, attn_df: pd.DataFrame | None) -> int:
    if attn_df is None:
        return 0
    ig_top = set(
        ig_time_df.sort_values("mean_abs_ig_attribution", ascending=False)
        .head(10)["time_step"]
        .astype(int)
        .tolist()
    )
    attn_top = set(attn_df.sort_values("attention_weight", ascending=False).head(10)["time_step"].astype(int).tolist())
    return int(len(ig_top & attn_top))


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "grid.color": "#CCCCCC",
            "grid.linewidth": 0.6,
        }
    )


def draw_compact(base: Path, data: dict[str, object]) -> None:
    style()
    meta = data["meta"]
    score_map = data["score_map"]
    ig_time_df = data["ig_time_df"]
    ig_feat_df = data["ig_feat_df"]
    ig_tf_df = data["ig_tf_df"]
    masking_df = data["masking_df"]

    threshold = float(score_map.get("threshold", meta["threshold"]))
    fused_score = float(score_map.get("fused_score", meta["fused_score"]))

    ylabels, heat = build_top10_matrix(ig_feat_df, ig_tf_df)
    vmax = float(np.percentile(heat, 99))
    if vmax <= 0:
        vmax = float(np.max(heat)) if float(np.max(heat)) > 0 else 1e-6

    top10 = ig_feat_df.sort_values("rank", ascending=True).head(10)

    fig = plt.figure(figsize=(7.2, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25])

    # A
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.bar(["Case score"], [fused_score], color="#1f77b4", width=0.55)
    ax_a.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.5, label="Threshold")
    ax_a.set_ylabel("Fused score")
    ax_a.set_title("A. Alarm score")
    ax_a.set_ylim(0.0, max(fused_score, threshold) * 1.35)
    ax_a.grid(axis="y", alpha=0.5)
    ax_a.text(0, fused_score + 0.01, f"score={fused_score:.4f}", ha="center", va="bottom", fontsize=8)
    ax_a.text(0.43, threshold + 0.005, f"threshold={threshold:.4f}", color="#d62728", fontsize=8)
    ax_a.legend(frameon=False, loc="upper right")

    # B
    ax_b = fig.add_subplot(gs[0, 1])
    im = ax_b.imshow(
        heat,
        aspect="auto",
        interpolation="nearest",
        cmap="cividis",
        vmin=0.0,
        vmax=vmax,
    )
    ax_b.set_title("B. IG evidence over time and features")
    ax_b.set_xlabel("Time step")
    ax_b.set_ylabel("Top IG features")
    ax_b.set_xlim(-0.5, 127.5)
    ax_b.set_xticks([0, 32, 64, 96, 127])
    ax_b.set_yticks(np.arange(len(ylabels)))
    ax_b.set_yticklabels(ylabels)
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.02)
    cbar.set_label("|IG attribution|")

    # C
    ax_c = fig.add_subplot(gs[1, 0])
    rev = top10.iloc[::-1]
    ax_c.barh(np.arange(len(rev)), rev["mean_abs_ig_attribution"].to_numpy(), color="#2ca02c")
    ax_c.set_yticks(np.arange(len(rev)))
    ax_c.set_yticklabels(rev["feature_name"].astype(str).tolist())
    ax_c.set_xlabel("Mean |IG|")
    ax_c.set_title("C. Top attributed features")
    ax_c.grid(axis="x", alpha=0.5)

    # D
    ax_d = fig.add_subplot(gs[1, 1])
    r = masking_df.sort_values("k")
    labels = ["Original"]
    vals = [fused_score]
    for _, row in r.iterrows():
        k = int(row["k"])
        labels.extend([f"IG@{k}", f"Rand@{k}"])
        vals.extend([float(row["ig_masked_fused_score"]), float(row["random_masked_fused_score_mean"])])
    colors = ["#1f77b4"] + ["#ff7f0e" if "IG" in l else "#7f7f7f" for l in labels[1:]]
    ax_d.bar(np.arange(len(vals)), vals, color=colors)
    ax_d.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.5, label="Threshold")
    ax_d.set_xticks(np.arange(len(vals)))
    ax_d.set_xticklabels(labels, rotation=25, ha="right")
    ax_d.set_ylabel("Fused score")
    ax_d.set_title("D. Masking verification")
    ax_d.grid(axis="y", alpha=0.5)
    ax_d.legend(frameon=False, loc="upper left")

    fig.savefig(base / "case_trace_plot_e8_compact_paper.png", dpi=300)
    fig.savefig(base / "case_trace_plot_e8_compact_paper.pdf")
    plt.close(fig)


def draw_full(base: Path, data: dict[str, object]) -> None:
    style()
    meta = data["meta"]
    score_map = data["score_map"]
    attn_df = data["attn_df"]
    ig_time_df = data["ig_time_df"]
    ig_feat_df = data["ig_feat_df"]
    ig_tf_df = data["ig_tf_df"]
    masking_df = data["masking_df"]

    threshold = float(score_map.get("threshold", meta["threshold"]))
    fused_score = float(score_map.get("fused_score", meta["fused_score"]))
    benign_ref = score_map.get("benign_reference_score")

    ylabels, heat = build_top10_matrix(ig_feat_df, ig_tf_df)
    vmax = float(np.percentile(heat, 99))
    if vmax <= 0:
        vmax = float(np.max(heat)) if float(np.max(heat)) > 0 else 1e-6

    overlap = overlap_top10(ig_time_df, attn_df)

    fig = plt.figure(figsize=(11.5, 10.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.95, 1.2, 1.1])

    # A: score contrast
    ax_a = fig.add_subplot(gs[0, 0])
    labels = ["Threshold", "Case fused"]
    values = [threshold, fused_score]
    colors = ["#9e9e9e", "#1f77b4"]
    if benign_ref is not None:
        labels.append("Benign ref")
        values.append(float(benign_ref))
        colors.append("#2ca02c")
    ax_a.bar(labels, values, color=colors)
    ax_a.set_ylabel("Score")
    ax_a.set_title("A. Score contrast")
    ax_a.grid(axis="y", alpha=0.5)
    for i, v in enumerate(values):
        ax_a.text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    # B: time-level evidence
    ax_b = fig.add_subplot(gs[0, 1])
    t = ig_time_df["time_step"].astype(int).to_numpy()
    ig_t = ig_time_df["mean_abs_ig_attribution"].astype(float).to_numpy()
    ig_t_n = normalize01(ig_t)
    ax_b.plot(t, ig_t_n, color="#1f77b4", linewidth=2.0, label="IG time attribution (norm)")
    if attn_df is not None:
        a = attn_df.sort_values("time_step")
        att_t = a["time_step"].astype(int).to_numpy()
        att_w = normalize01(a["attention_weight"].astype(float).to_numpy())
        ax_b.plot(att_t, att_w, color="#ff7f0e", linewidth=1.8, label="Attention weight (norm)")
    ax_b.set_xlim(0, 127)
    ax_b.set_xlabel("Time step")
    ax_b.set_ylabel("Normalized importance")
    ax_b.set_title("B. Time-level evidence")
    ax_b.grid(alpha=0.5)
    ax_b.legend(frameon=False, loc="upper right")
    ax_b.text(
        0.01,
        0.06,
        f"Attention and IG partially overlap (top-10 overlap={overlap}/10)",
        transform=ax_b.transAxes,
        fontsize=8,
    )

    # C: top-feature heatmap
    ax_c = fig.add_subplot(gs[1, :])
    im = ax_c.imshow(
        heat,
        aspect="auto",
        interpolation="nearest",
        cmap="cividis",
        vmin=0.0,
        vmax=vmax,
    )
    ax_c.set_title("C. Top-feature IG heatmap")
    ax_c.set_xlabel("Time step")
    ax_c.set_ylabel("Top IG features")
    ax_c.set_xlim(-0.5, 127.5)
    ax_c.set_xticks([0, 32, 64, 96, 127])
    ax_c.set_yticks(np.arange(len(ylabels)))
    ax_c.set_yticklabels(ylabels)
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.016, pad=0.01)
    cbar.set_label("|IG attribution|")

    # D: top 10 features
    ax_d = fig.add_subplot(gs[2, 0])
    top10 = ig_feat_df.sort_values("rank", ascending=True).head(10).iloc[::-1]
    ax_d.barh(np.arange(len(top10)), top10["mean_abs_ig_attribution"].to_numpy(), color="#2ca02c")
    ax_d.set_yticks(np.arange(len(top10)))
    ax_d.set_yticklabels(top10["feature_name"].astype(str).tolist())
    ax_d.set_xlabel("Mean |IG|")
    ax_d.set_title("D. Top 10 IG features")
    ax_d.grid(axis="x", alpha=0.5)

    # E: masking verification
    ax_e = fig.add_subplot(gs[2, 1])
    r = masking_df.sort_values("k")
    labels2 = ["Original"]
    vals2 = [fused_score]
    for _, row in r.iterrows():
        k = int(row["k"])
        labels2.extend([f"IG@{k}", f"Rand@{k}"])
        vals2.extend([float(row["ig_masked_fused_score"]), float(row["random_masked_fused_score_mean"])])
    colors2 = ["#1f77b4"] + ["#ff7f0e" if "IG" in l else "#7f7f7f" for l in labels2[1:]]
    ax_e.bar(np.arange(len(vals2)), vals2, color=colors2)
    ax_e.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.5, label="Threshold")
    ax_e.set_xticks(np.arange(len(vals2)))
    ax_e.set_xticklabels(labels2, rotation=25, ha="right")
    ax_e.set_ylabel("Fused score")
    ax_e.set_title("E. Masking verification")
    ax_e.grid(axis="y", alpha=0.5)
    ax_e.legend(frameon=False, loc="upper left")

    fig.savefig(base / "case_trace_plot_e8_paper.png", dpi=300)
    fig.savefig(base / "case_trace_plot_e8_paper.pdf")
    plt.close(fig)


def write_notes(base: Path, data: dict[str, object]) -> None:
    score_map = data["score_map"]
    masking_df = data["masking_df"].sort_values("k")
    threshold = float(score_map.get("threshold", 0.425420))

    row5 = masking_df[masking_df["k"] == 5].iloc[0]
    row10 = masking_df[masking_df["k"] == 10].iloc[0]
    row20 = masking_df[masking_df["k"] == 20].iloc[0]

    note = f"""# Case Trace Figure Notes (E8)

- The heatmap was changed from all feature indices to a **top-10 IG feature heatmap** to improve interpretability and avoid a low-contrast, overly gray full-feature map.
- No experiment results were recomputed. The new figures are rendered only from existing files in this directory:
  - `case_metadata_e8.json`
  - `case_score_components_e8.csv`
  - `case_attention_time_weights_e8.csv`
  - `case_ig_time_attribution_e8.csv`
  - `case_ig_feature_attribution_e8.csv`
  - `case_ig_time_feature_matrix_e8.csv`
  - `case_masking_verification_e8.csv`
- Top-feature heatmap settings:
  - Features: top 10 by IG rank from `case_ig_feature_attribution_e8.csv`
  - Values: absolute IG attribution from `case_ig_time_feature_matrix_e8.csv`
  - Colormap: `cividis`
  - Normalization: `vmin=0`, `vmax=p99` of the plotted top-10 matrix values
- Masking behavior from existing results:
  - `IG@5` score = {float(row5['ig_masked_fused_score']):.6f} (can increase score in this case)
  - `IG@10` score = {float(row10['ig_masked_fused_score']):.6f} < threshold {threshold:.6f}
  - `IG@20` score = {float(row20['ig_masked_fused_score']):.6f} < threshold {threshold:.6f}
  - Random masking means (`Rand@5/10/20`) remain above threshold in this case.
"""
    (base / "case_trace_figure_notes_e8.md").write_text(note, encoding="utf-8")


def main() -> None:
    args = parse_args()
    base = Path(args.input_dir).resolve()
    data = load_inputs(base)
    draw_full(base, data)
    draw_compact(base, data)
    write_notes(base, data)
    print(f"Saved paper figures and notes to: {base}")


if __name__ == "__main__":
    main()
