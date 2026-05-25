#!/usr/bin/env python3
"""Minor polishing pass for XAI case-trace V2 outputs (no model recomputation)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_FEATURE_ORDER = [
    "Bwd Packet Length Std",
    "PSH Flag Count",
    "URG Flag Count",
    "Idle Min",
    "SYN Flag Count",
    "Bwd IAT Total",
    "Destination Port",
    "Flow IAT Mean",
    "Min Packet Length",
    "FIN Flag Count",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v2-dir", default="results/cicids_strict_mps/xai_case_trace_e8_v2")
    p.add_argument("--v1-dir", default="results/cicids_strict_mps/xai_case_trace_e8")
    return p.parse_args()


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def evidence_group_for_feature(name: str) -> str:
    if name in {"Bwd Packet Length Std", "Min Packet Length"}:
        return "packet length variability"
    if name in {"PSH Flag Count", "URG Flag Count", "SYN Flag Count", "FIN Flag Count"}:
        return "TCP flag behavior"
    if name in {"Idle Min", "Bwd IAT Total", "Flow IAT Mean"}:
        return "temporal interval / idle behavior"
    if name == "Destination Port":
        return "port-related feature"
    return "other"


def build_heatmap(
    ig_tf_df: pd.DataFrame,
    ordered_features: list[str],
    n_steps: int = 128,
) -> tuple[np.ndarray, float]:
    m = ig_tf_df.copy()
    m["time_step"] = m["time_step"].astype(int)
    pivot = (
        m[m["feature_name"].isin(ordered_features)]
        .pivot_table(index="feature_name", columns="time_step", values="abs_ig_attribution", aggfunc="mean")
        .reindex(index=ordered_features)
        .reindex(columns=list(range(n_steps)))
        .fillna(0.0)
    )
    raw = pivot.to_numpy(dtype=np.float64)
    transformed = np.log1p(raw)
    vmax = float(np.percentile(transformed, 99))
    if vmax <= 0:
        vmax = float(np.max(transformed)) if float(np.max(transformed)) > 0 else 1e-6
    clipped = np.clip(transformed, 0.0, vmax)
    return clipped, vmax


def build_redundancy_summary(nonred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, r in nonred_df.iterrows():
        name = str(r["feature_name"])
        raw_rank = int(r["rank"])
        mean_abs_ig = float(r["mean_abs_ig_attribution"])
        max_corr = float(r.get("max_abs_corr_with_selected", 0.0))
        if raw_rank == 1:
            max_corr = 0.0
        group = evidence_group_for_feature(name)
        note = "representative after correlation pruning"
        rows.append(
            {
                "selected_feature": name,
                "original_raw_rank": raw_rank,
                "mean_abs_ig": mean_abs_ig,
                "max_abs_corr_with_previous_selected": max_corr,
                "evidence_group": group,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def draw_polished_figure(
    out_png: Path,
    out_pdf: Path,
    *,
    case_meta: dict,
    score_map: dict[str, float],
    ig_time_df: pd.DataFrame,
    attn_df: pd.DataFrame | None,
    masking_df: pd.DataFrame,
    nonred_df: pd.DataFrame,
    heatmap: np.ndarray,
    heatmap_vmax: float,
) -> None:
    threshold = float(case_meta["threshold"])
    fused_score = float(case_meta["fused_score"])
    benign_ref = score_map.get("benign_reference_score")

    t = ig_time_df["time_step"].astype(int).to_numpy()
    ig_t = normalize01(ig_time_df["mean_abs_ig_attribution"].astype(float).to_numpy())

    att_t = None
    att_w = None
    overlap_note = "attention unavailable"
    if attn_df is not None and len(attn_df) > 0:
        att_sorted = attn_df.sort_values("time_step")
        att_t = att_sorted["time_step"].astype(int).to_numpy()
        att_w = normalize01(att_sorted["attention_weight"].astype(float).to_numpy())
        ig_top = set(
            ig_time_df.sort_values("mean_abs_ig_attribution", ascending=False)
            .head(10)["time_step"]
            .astype(int)
            .tolist()
        )
        att_top = set(
            attn_df.sort_values("attention_weight", ascending=False)
            .head(10)["time_step"]
            .astype(int)
            .tolist()
        )
        overlap_note = f"partial overlap (top-10 overlap={len(ig_top & att_top)}/10)"

    m = masking_df.sort_values("k").copy()
    x_labels = ["Original"]
    x_vals = [fused_score]
    x_colors = ["#1f77b4"]
    for _, r in m.iterrows():
        k = int(r["k"])
        x_labels.extend([f"IG@{k}", f"Rand@{k}"])
        x_vals.extend([float(r["ig_masked_fused_score"]), float(r["random_masked_fused_score_mean"])])
        x_colors.extend(["#ff7f0e", "#7f7f7f"])

    ordered_names = nonred_df["feature_name"].astype(str).tolist()
    dvals = nonred_df["mean_abs_ig_attribution"].astype(float).to_numpy()

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

    fig = plt.figure(figsize=(12.5, 10.8), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.95, 1.25, 1.15])

    # Panel A
    ax_a = fig.add_subplot(gs[0, 0])
    a_labels = ["Threshold", "Case fused score"]
    a_vals = [threshold, fused_score]
    a_colors = ["#9e9e9e", "#1f77b4"]
    if benign_ref is not None:
        a_labels.append("Benign reference mean")
        a_vals.append(float(benign_ref))
        a_colors.append("#2ca02c")
    ax_a.bar(a_labels, a_vals, color=a_colors)
    ax_a.set_title("A. Score contrast")
    ax_a.set_ylabel("Score")
    ax_a.grid(axis="y", alpha=0.35)
    for i, v in enumerate(a_vals):
        ax_a.text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    # Panel B
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(t, ig_t, color="#1f77b4", lw=2.0, label="IG time attribution (norm)")
    if att_t is not None and att_w is not None:
        ax_b.plot(att_t, att_w, color="#ff7f0e", lw=1.8, label="Attention weight (norm)")
    ax_b.set_xlim(0, 127)
    ax_b.set_xlabel("Time step")
    ax_b.set_ylabel("Normalized importance")
    ax_b.set_title("B. Time-level evidence")
    ax_b.grid(alpha=0.35)
    ax_b.legend(frameon=False, loc="upper right")
    ax_b.text(0.01, 0.04, f"IG vs attention: {overlap_note}", transform=ax_b.transAxes, fontsize=8)

    # Panel C (order matches panel D exactly)
    ax_c = fig.add_subplot(gs[1, :])
    im = ax_c.imshow(
        heatmap,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=heatmap_vmax,
    )
    ax_c.set_title("C. Correlation-pruned non-redundant feature heatmap")
    ax_c.set_xlabel("Time step")
    ax_c.set_ylabel("Feature (same order as Panel D)")
    ax_c.set_xticks([0, 32, 64, 96, 127])
    ax_c.set_yticks(np.arange(len(ordered_names)))
    ax_c.set_yticklabels(ordered_names)
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.016, pad=0.01)
    cbar.set_label("|IG| (log1p + p99 clip)")

    # Panel D
    ax_d = fig.add_subplot(gs[2, 0])
    ax_d.barh(np.arange(len(ordered_names)), dvals, color="#2ca02c")
    ax_d.set_yticks(np.arange(len(ordered_names)))
    ax_d.set_yticklabels(ordered_names)
    ax_d.invert_yaxis()  # top-to-bottom matches requested order
    ax_d.set_xlabel("Mean |IG|")
    ax_d.set_title("D. Top correlation-pruned non-redundant features")
    ax_d.grid(axis="x", alpha=0.35)

    # Panel E
    ax_e = fig.add_subplot(gs[2, 1])
    ax_e.bar(np.arange(len(x_vals)), x_vals, color=x_colors)
    ax_e.axhline(threshold, color="#d62728", ls="--", lw=1.5, label="Threshold")
    ax_e.set_xticks(np.arange(len(x_vals)))
    ax_e.set_xticklabels(x_labels, rotation=26, ha="right")
    ax_e.set_ylabel("Fused score")
    ax_e.set_title("E. Masking verification")
    ax_e.grid(axis="y", alpha=0.35)
    ax_e.legend(frameon=False, loc="upper left")
    ax_e.text(0.01, 0.04, "IG@10/20 < threshold; Rand@10/20 > threshold", transform=ax_e.transAxes, fontsize=8)

    fig.suptitle(
        "Single-case trace with correlation-pruned non-redundant features\n"
        "(redundancy pruning uses feature correlation only to reduce repetitive evidence display; "
        "it is not a proof of statistical independence)",
        fontsize=11,
    )

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    v2 = Path(args.v2_dir).resolve()
    v1 = Path(args.v1_dir).resolve()

    # V2 inputs
    nonred_df = pd.read_csv(v2 / "case_top_features_nonredundant_e8_v2.csv")
    raw_df = pd.read_csv(v2 / "case_top_features_raw_e8_v2.csv")

    # V1 supporting files (no recomputation)
    case_meta = json.loads((v1 / "case_metadata_e8.json").read_text(encoding="utf-8"))
    score_df = pd.read_csv(v1 / "case_score_components_e8.csv")
    ig_time_df = pd.read_csv(v1 / "case_ig_time_attribution_e8.csv")
    ig_tf_df = pd.read_csv(v1 / "case_ig_time_feature_matrix_e8.csv")
    masking_df = pd.read_csv(v1 / "case_masking_verification_e8.csv")
    attn_path = v1 / "case_attention_time_weights_e8.csv"
    attn_df = pd.read_csv(attn_path) if attn_path.exists() else None

    # Enforce requested feature order
    ordered = TARGET_FEATURE_ORDER
    have = set(nonred_df["feature_name"].astype(str).tolist())
    missing = [x for x in ordered if x not in have]
    if missing:
        raise RuntimeError(f"Requested ordered features not found in nonredundant set: {missing}")

    nonred_df = nonred_df.set_index("feature_name").loc[ordered].reset_index()

    # Redundancy summary CSV
    summary_df = build_redundancy_summary(nonred_df)
    summary_df.to_csv(v2 / "case_feature_redundancy_summary_e8_v2.csv", index=False)

    # Heatmap matrix (same order as Panel D)
    heatmap, vmax = build_heatmap(ig_tf_df, ordered, n_steps=128)

    # Figure export
    score_map = {str(r["component"]): float(r["value"]) for _, r in score_df.iterrows()}
    draw_polished_figure(
        v2 / "case_trace_plot_e8_v2_polished.png",
        v2 / "case_trace_plot_e8_v2_polished.pdf",
        case_meta=case_meta,
        score_map=score_map,
        ig_time_df=ig_time_df,
        attn_df=attn_df,
        masking_df=masking_df,
        nonred_df=nonred_df,
        heatmap=heatmap,
        heatmap_vmax=vmax,
    )

    # Update summary markdown
    corr_threshold = float(nonred_df["corr_threshold"].iloc[0]) if "corr_threshold" in nonred_df.columns else 0.85
    lines: list[str] = []
    lines.append("# Case Trace Summary V2 (E8, Polished)")
    lines.append("")
    lines.append("## Case Invariants")
    lines.append("")
    lines.append("- Selected case index: `13570`")
    lines.append("- Attack type: `DDoS`")
    lines.append(f"- Threshold: `{float(case_meta['threshold']):.6f}`")
    lines.append(f"- Case fused score: `{float(case_meta['fused_score']):.6f}`")
    lines.append("- IG target: `fused_tad_score`")
    lines.append("- No model/IG recomputation in this polishing pass.")
    lines.append("")
    lines.append("## Heatmap and Redundancy-Reduction Settings")
    lines.append("")
    lines.append("- Heatmap transform: `log1p(abs_IG)` with `p99` clipping.")
    lines.append("- Colormap: `magma` (high-contrast).")
    lines.append("- Correlation metric: `Spearman`.")
    lines.append(f"- Correlation threshold: `|corr| >= {corr_threshold:.2f}`.")
    lines.append("- Correlation reference set: `correctly_detected_same_attack_type` (from V2 run metadata).")
    lines.append(
        "- This is a redundancy-reduction step for interpretability, not a proof of statistical independence."
    )
    lines.append("")
    lines.append("## Raw Top-10 vs Non-redundant Top-10")
    lines.append("")
    lines.append(
        "- Raw top-10 contains several highly correlated packet-length descriptors, which can make evidence display repetitive."
    )
    lines.append(
        "- Correlation-pruned non-redundant features keep representative evidence while reducing repeated correlated descriptors."
    )
    lines.append("")
    lines.append("### Raw Top-10")
    lines.append("")
    for _, r in raw_df.sort_values("rank").iterrows():
        lines.append(f"- {int(r['rank'])}. {r['feature_name']} (mean|IG|={float(r['mean_abs_ig_attribution']):.6e})")
    lines.append("")
    lines.append("### Non-redundant Top-10 (Ordered for Panels C/D)")
    lines.append("")
    for i, (_, r) in enumerate(nonred_df.iterrows(), start=1):
        lines.append(
            f"- {i}. {r['feature_name']} (raw_rank={int(r['rank'])}, mean|IG|={float(r['mean_abs_ig_attribution']):.6e})"
        )
    lines.append("")
    lines.append("## Masking Conclusions (Unchanged)")
    lines.append("")
    m10 = masking_df[masking_df["k"] == 10].iloc[0]
    m20 = masking_df[masking_df["k"] == 20].iloc[0]
    lines.append(f"- IG@10 score: `{float(m10['ig_masked_fused_score']):.6f}` (`{m10['ig_masked_decision']}`)")
    lines.append(f"- IG@20 score: `{float(m20['ig_masked_fused_score']):.6f}` (`{m20['ig_masked_decision']}`)")
    lines.append("- Random masking at k=10/20 remains above threshold in the reported summary criterion.")
    lines.append("")
    lines.append("## Figure Text Update")
    lines.append("")
    lines.append("- Use phrase: `correlation-pruned non-redundant features`.")
    lines.append("- Do not claim statistical independence.")
    lines.append(
        "- Redundancy pruning is based on feature correlation and is used only to reduce repetitive evidence display."
    )

    (v2 / "case_trace_summary_e8_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Polishing completed.")
    print(f"Wrote: {v2 / 'case_trace_plot_e8_v2_polished.png'}")
    print(f"Wrote: {v2 / 'case_trace_plot_e8_v2_polished.pdf'}")
    print(f"Wrote: {v2 / 'case_feature_redundancy_summary_e8_v2.csv'}")
    print(f"Updated: {v2 / 'case_trace_summary_e8_v2.md'}")


if __name__ == "__main__":
    main()

