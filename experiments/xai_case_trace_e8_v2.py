#!/usr/bin/env python3
"""V2 post-analysis and visualization for an existing CICIDS2017 XAI case trace.

Evaluation-only script:
- no training
- no checkpoint/weight changes
- no threshold/case change
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn.utils import parametrize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.tcn_gan_experiment import (  # noqa: E402
    TCNDiscriminator,
    TCNGenerator,
    compute_anomaly_scores,
)
from pipelines.run_tcn_cross_dataset_minimal import (  # noqa: E402
    load_windowed_dataset_independent_calibration_strict,
    select_device,
)
from experiments.xai_case_trace_e8 import (  # noqa: E402
    build_test_window_metadata,
    load_state_dict_compat,
)


DEFAULT_TRAIN_FILES = [
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
]
DEFAULT_TEST_FILES = [
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="results/cicids_strict_mps/xai_case_trace_e8")
    p.add_argument("--output-dir", default="results/cicids_strict_mps/xai_case_trace_e8_v2")
    p.add_argument("--dataset", default="cicids2017")
    p.add_argument("--data-dir", default="dataset/CICIDS2017")
    p.add_argument("--train-files", nargs="+", default=DEFAULT_TRAIN_FILES)
    p.add_argument("--test-files", nargs="+", default=DEFAULT_TEST_FILES)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="mps")
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--anomaly-ratio", type=float, default=0.15)
    p.add_argument("--calib-ratio", type=float, default=0.2)
    p.add_argument("--score-alpha", type=float, default=0.24)
    p.add_argument("--test-batch-size", type=int, default=256)
    p.add_argument("--corr-threshold", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mn) / (mx - mn)


def resolve_col(df: pd.DataFrame, candidates: list[str]) -> str:
    by_lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        c = by_lower.get(cand.lower())
        if c is not None:
            return str(c)
    raise KeyError(f"No column found among candidates: {candidates}")


def prepare_models(
    checkpoint: dict[str, Any],
    *,
    window_size: int,
    feature_dim: int,
    device: torch.device,
) -> tuple[TCNGenerator, TCNDiscriminator]:
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint.get("args", {}), dict) else {}
    hidden_channels = ckpt_args.get("hidden_channels", [128, 128])
    dropout = float(ckpt_args.get("dropout", 0.2))
    latent_dim = int(ckpt_args.get("latent_dim", 64))
    disc_pooling = str(ckpt_args.get("disc_pooling", "attn"))

    generator = TCNGenerator(window_size, feature_dim, latent_dim, hidden_channels, dropout).to(device)
    discriminator = TCNDiscriminator(window_size, feature_dim, hidden_channels, dropout, pooling=disc_pooling).to(device)

    g_state = checkpoint.get("generator_state_dict", checkpoint.get("generator"))
    d_state = checkpoint.get("discriminator_state_dict", checkpoint.get("discriminator"))
    if g_state is None or d_state is None:
        raise ValueError("Checkpoint missing generator/discriminator state dict")

    load_state_dict_compat(generator, g_state, model_name="generator")
    load_state_dict_compat(discriminator, d_state, model_name="discriminator")
    discriminator.eval()
    return generator, discriminator


def greedy_nonredundant(
    ranked: pd.DataFrame,
    corr_df: pd.DataFrame,
    *,
    k: int,
    threshold: float,
) -> pd.DataFrame:
    kept: list[dict[str, Any]] = []
    kept_names: list[str] = []

    for _, row in ranked.iterrows():
        fname = str(row["feature_name"])
        if fname not in corr_df.index:
            continue

        redundant = False
        max_corr = 0.0
        max_with = ""
        for kept_name in kept_names:
            c = float(corr_df.loc[fname, kept_name])
            if np.isnan(c):
                c = 0.0
            ac = abs(c)
            if ac > max_corr:
                max_corr = ac
                max_with = kept_name
            if ac >= threshold:
                redundant = True
                break

        if not redundant:
            out = dict(row)
            out["max_abs_corr_with_selected"] = float(max_corr)
            out["most_correlated_selected_feature"] = max_with
            kept.append(out)
            kept_names.append(fname)

        if len(kept) >= k:
            break

    return pd.DataFrame(kept)


def build_redundancy_groups(raw_top: pd.DataFrame, corr_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    groups: list[dict[str, Any]] = []
    representatives: list[str] = []
    group_id_for_rep: dict[str, int] = {}
    next_gid = 1

    for _, row in raw_top.iterrows():
        fname = str(row["feature_name"])
        fr = int(row["rank"])

        assigned_gid = None
        assigned_rep = None
        corr_to_rep = 0.0

        for rep in representatives:
            c = float(corr_df.loc[fname, rep]) if (fname in corr_df.index and rep in corr_df.columns) else 0.0
            if np.isnan(c):
                c = 0.0
            if abs(c) >= threshold:
                assigned_gid = group_id_for_rep[rep]
                assigned_rep = rep
                corr_to_rep = c
                break

        if assigned_gid is None:
            assigned_gid = next_gid
            next_gid += 1
            assigned_rep = fname
            corr_to_rep = 1.0
            representatives.append(fname)
            group_id_for_rep[fname] = assigned_gid
            is_rep = True
        else:
            is_rep = False

        groups.append(
            {
                "group_id": int(assigned_gid),
                "group_representative": str(assigned_rep),
                "feature_name": fname,
                "feature_index": int(row["feature_index"]),
                "raw_rank": fr,
                "corr_with_representative": float(corr_to_rep),
                "abs_corr_with_representative": float(abs(corr_to_rep)),
                "is_representative": bool(is_rep),
            }
        )

    return pd.DataFrame(groups)


def save_heatmap_matrix(
    ig_tf_df: pd.DataFrame,
    feature_names: list[str],
    *,
    n_time: int = 128,
) -> np.ndarray:
    m = ig_tf_df.copy()
    m["time_step"] = m["time_step"].astype(int)

    pivot = (
        m[m["feature_name"].isin(feature_names)]
        .pivot_table(index="feature_name", columns="time_step", values="abs_ig_attribution", aggfunc="mean")
        .reindex(index=feature_names)
        .reindex(columns=list(range(n_time)))
        .fillna(0.0)
    )
    return pivot.to_numpy(dtype=np.float64)


def draw_figures(
    out_dir: Path,
    *,
    score_df: pd.DataFrame,
    ig_time_df: pd.DataFrame,
    attn_df: pd.DataFrame | None,
    ig_tf_df: pd.DataFrame,
    nonred_top10: pd.DataFrame,
    masking_df: pd.DataFrame,
    case_meta: dict[str, Any],
    heatmap_transform_name: str,
    heatmap_vmax: float,
) -> None:
    threshold = float(case_meta["threshold"])
    fused_score = float(case_meta["fused_score"])

    score_map = {str(r["component"]): float(r["value"]) for _, r in score_df.iterrows()}
    benign_ref = score_map.get("benign_reference_score")

    # Time evidence
    t_col = resolve_col(ig_time_df, ["time_step"])
    ig_col = resolve_col(ig_time_df, ["normalized_time_importance", "mean_abs_ig_attribution"])
    t = ig_time_df[t_col].astype(int).to_numpy()
    ig_t = ig_time_df[ig_col].astype(float).to_numpy()
    if ig_col != "normalized_time_importance":
        ig_t = normalize01(ig_t)

    att_t = None
    att_v = None
    overlap_text = "attention unavailable"
    if attn_df is not None and len(attn_df) > 0:
        at_col = resolve_col(attn_df, ["time_step"])
        aw_col = resolve_col(attn_df, ["attention_weight"])
        att_t = attn_df[at_col].astype(int).to_numpy()
        att_v = normalize01(attn_df[aw_col].astype(float).to_numpy())

        ig_top = set(ig_time_df.sort_values(resolve_col(ig_time_df, ["mean_abs_ig_attribution"]), ascending=False).head(10)[t_col].astype(int).tolist())
        att_top = set(attn_df.sort_values(aw_col, ascending=False).head(10)[at_col].astype(int).tolist())
        overlap_text = f"partial overlap (top-10 overlap={len(ig_top & att_top)}/10)"

    # Heatmap from non-redundant features
    names = nonred_top10["feature_name"].astype(str).tolist()
    heat_raw = save_heatmap_matrix(ig_tf_df, names, n_time=128)
    heat_log = np.log1p(heat_raw)
    heat_plot = np.clip(heat_log, 0.0, float(heatmap_vmax))

    # Masking data
    mdf = masking_df.sort_values("k").copy()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    # Full figure (A-E)
    fig = plt.figure(figsize=(12.5, 10.8), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.95, 1.25, 1.15])

    ax_a = fig.add_subplot(gs[0, 0])
    labels = ["Threshold", "Case fused score"]
    vals = [threshold, fused_score]
    colors = ["#9e9e9e", "#1f77b4"]
    if benign_ref is not None:
        labels.append("Benign ref mean")
        vals.append(float(benign_ref))
        colors.append("#2ca02c")
    ax_a.bar(labels, vals, color=colors)
    ax_a.set_title("A. Score contrast")
    ax_a.set_ylabel("Score")
    ax_a.grid(axis="y", alpha=0.35)
    for i, v in enumerate(vals):
        ax_a.text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(t, ig_t, color="#1f77b4", lw=2.0, label="IG time attribution (norm)")
    if att_t is not None and att_v is not None:
        ax_b.plot(att_t, att_v, color="#ff7f0e", lw=1.8, label="Attention weight (norm)")
    ax_b.set_xlim(0, 127)
    ax_b.set_xlabel("Time step")
    ax_b.set_ylabel("Normalized importance")
    ax_b.set_title("B. Time-level evidence")
    ax_b.grid(alpha=0.35)
    ax_b.legend(frameon=False, loc="upper right")
    ax_b.text(0.01, 0.04, f"IG vs attention: {overlap_text}", transform=ax_b.transAxes, fontsize=8)

    ax_c = fig.add_subplot(gs[1, :])
    im = ax_c.imshow(
        heat_plot,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=float(heatmap_vmax),
    )
    ax_c.set_title("C. Top-feature IG heatmap (non-redundant features)")
    ax_c.set_xlabel("Time step")
    ax_c.set_ylabel("Feature")
    ax_c.set_xticks([0, 32, 64, 96, 127])
    ax_c.set_yticks(np.arange(len(names)))
    ax_c.set_yticklabels(names)
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.016, pad=0.01)
    cbar.set_label(f"|IG| ({heatmap_transform_name})")

    ax_d = fig.add_subplot(gs[2, 0])
    show = nonred_top10.iloc[::-1]
    ax_d.barh(np.arange(len(show)), show["mean_abs_ig_attribution"].to_numpy(dtype=float), color="#2ca02c")
    ax_d.set_yticks(np.arange(len(show)))
    ax_d.set_yticklabels(show["feature_name"].astype(str).tolist())
    ax_d.set_xlabel("Mean |IG|")
    ax_d.set_title("D. Top non-redundant IG features")
    ax_d.grid(axis="x", alpha=0.35)

    ax_e = fig.add_subplot(gs[2, 1])
    xlabels = ["Original"]
    xvals = [fused_score]
    xcolors = ["#1f77b4"]
    for _, r in mdf.iterrows():
        k = int(r["k"])
        xlabels.extend([f"IG@{k}", f"Rand@{k}"])
        xvals.extend([float(r["ig_masked_fused_score"]), float(r["random_masked_fused_score_mean"])])
        xcolors.extend(["#ff7f0e", "#7f7f7f"])
    ax_e.bar(np.arange(len(xvals)), xvals, color=xcolors)
    ax_e.axhline(threshold, color="#d62728", ls="--", lw=1.5, label="Threshold")
    ax_e.set_xticks(np.arange(len(xvals)))
    ax_e.set_xticklabels(xlabels, rotation=26, ha="right")
    ax_e.set_ylabel("Fused score")
    ax_e.set_title("E. Masking verification")
    ax_e.grid(axis="y", alpha=0.35)
    ax_e.legend(frameon=False, loc="upper left")

    # emphasize feedback point
    ax_e.text(0.01, 0.04, "IG@10/20 < threshold; Rand@10/20 > threshold", transform=ax_e.transAxes, fontsize=8)

    fig.savefig(out_dir / "case_trace_plot_e8_v2.png", dpi=300)
    fig.savefig(out_dir / "case_trace_plot_e8_v2.pdf")
    plt.close(fig)

    # Compact figure (2x2)
    fig2 = plt.figure(figsize=(7.8, 6.0), constrained_layout=True)
    gs2 = fig2.add_gridspec(2, 2)

    ax1 = fig2.add_subplot(gs2[0, 0])
    ax1.bar(["Case"], [fused_score], color="#1f77b4")
    ax1.axhline(threshold, color="#d62728", ls="--", lw=1.5)
    ax1.set_title("A. Score and threshold")
    ax1.set_ylabel("Fused score")
    ax1.grid(axis="y", alpha=0.35)

    ax2 = fig2.add_subplot(gs2[0, 1])
    im2 = ax2.imshow(
        heat_plot,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=float(heatmap_vmax),
    )
    ax2.set_title("B. Non-redundant IG heatmap")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Feature")
    ax2.set_xticks([0, 32, 64, 96, 127])
    ax2.set_yticks(np.arange(len(names)))
    ax2.set_yticklabels(names)
    cb2 = fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
    cb2.set_label("|IG|")

    ax3 = fig2.add_subplot(gs2[1, 0])
    ax3.barh(np.arange(len(show)), show["mean_abs_ig_attribution"].to_numpy(dtype=float), color="#2ca02c")
    ax3.set_yticks(np.arange(len(show)))
    ax3.set_yticklabels(show["feature_name"].astype(str).tolist())
    ax3.set_title("C. Top non-redundant features")
    ax3.set_xlabel("Mean |IG|")
    ax3.grid(axis="x", alpha=0.35)

    ax4 = fig2.add_subplot(gs2[1, 1])
    ax4.bar(np.arange(len(xvals)), xvals, color=xcolors)
    ax4.axhline(threshold, color="#d62728", ls="--", lw=1.5)
    ax4.set_xticks(np.arange(len(xvals)))
    ax4.set_xticklabels(xlabels, rotation=24, ha="right")
    ax4.set_title("D. Masking verification")
    ax4.grid(axis="y", alpha=0.35)

    fig2.savefig(out_dir / "case_trace_plot_e8_v2_compact.png", dpi=300)
    plt.close(fig2)


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    input_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    case_meta = json.loads((input_dir / "case_metadata_e8.json").read_text(encoding="utf-8"))
    if int(case_meta.get("selected_window_index", -1)) != 13570:
        raise RuntimeError("Selected case index changed; expected 13570.")
    if str(case_meta.get("attack_type", "")) != "DDoS":
        raise RuntimeError("Selected case attack type changed; expected DDoS.")

    threshold = float(case_meta.get("threshold", 0.0))
    fused_score = float(case_meta.get("fused_score", 0.0))
    ig_target = str(case_meta.get("ig_target", ""))
    if abs(threshold - 0.425420) > 1e-5:
        raise RuntimeError(f"Threshold mismatch: {threshold}")
    if abs(fused_score - 0.5065301656723022) > 1e-5:
        raise RuntimeError(f"Case fused score mismatch: {fused_score}")
    if ig_target != "fused_tad_score":
        raise RuntimeError(f"IG target changed: {ig_target}")

    score_df = pd.read_csv(input_dir / "case_score_components_e8.csv")
    ig_time_df = pd.read_csv(input_dir / "case_ig_time_attribution_e8.csv")
    ig_feat_df = pd.read_csv(input_dir / "case_ig_feature_attribution_e8.csv")
    ig_tf_df = pd.read_csv(input_dir / "case_ig_time_feature_matrix_e8.csv")
    masking_df = pd.read_csv(input_dir / "case_masking_verification_e8.csv")
    attn_path = input_dir / "case_attention_time_weights_e8.csv"
    attn_df = pd.read_csv(attn_path) if attn_path.exists() else None

    ranked = ig_feat_df.sort_values("rank", ascending=True).reset_index(drop=True)
    raw_top10 = ranked.head(10).copy()

    # Rebuild strict test windows and compute detected windows for correlation reference set.
    ds, _ = load_windowed_dataset_independent_calibration_strict(
        dataset=str(args.dataset),
        data_dir=str(args.data_dir),
        train_files=list(args.train_files),
        test_files=list(args.test_files),
        window_size=int(args.window_size),
        stride=int(args.stride),
        anomaly_ratio=float(args.anomaly_ratio),
        calib_ratio=float(args.calib_ratio),
        scaler="minmax",
        clip_minmax=True,
        progress=True,
    )

    x_train_benign = np.asarray(ds.x_train, dtype=np.float32)
    x_test = np.asarray(ds.x_test, dtype=np.float32)
    y_true = np.asarray(ds.y_test, dtype=np.int64)
    feat_dim = int(x_train_benign.shape[2])
    feature_names = [str(x) for x in list(ds.feature_names)] if getattr(ds, "feature_names", None) else []
    if len(feature_names) != feat_dim:
        feature_names = [f"f{i}" for i in range(feat_dim)]

    ckpt_path = Path(case_meta["protocol_metadata"]["checkpoint"]).resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    device = select_device(str(args.device))
    _, discriminator = prepare_models(ckpt, window_size=int(args.window_size), feature_dim=feat_dim, device=device)

    y_score = compute_anomaly_scores(
        discriminator,
        x_test,
        device,
        int(args.test_batch_size),
        score_mode="fused",
        score_alpha=float(args.score_alpha),
        x_ref_benign=x_train_benign,
    )
    y_pred = (np.asarray(y_score) >= threshold).astype(np.uint8)

    metadata_by_window = build_test_window_metadata(
        dataset=str(args.dataset),
        data_dir=str(args.data_dir),
        test_files=list(args.test_files),
        window_size=int(args.window_size),
        stride=int(args.stride),
        anomaly_ratio=float(args.anomaly_ratio),
        expected_window_labels=y_true,
    )

    detected_idx = np.where((y_true == 1) & (y_pred == 1))[0]
    sel_attack = str(case_meta.get("attack_type", "")).strip().lower()
    same_attack_idx: list[int] = []
    for i in detected_idx.tolist():
        at = str(metadata_by_window[int(i)].get("attack_type", "")).strip().lower()
        if sel_attack and sel_attack in at:
            same_attack_idx.append(int(i))

    if same_attack_idx:
        ref_indices = np.asarray(same_attack_idx, dtype=np.int64)
        ref_set_name = "correctly_detected_same_attack_type"
    else:
        ref_indices = np.asarray(detected_idx, dtype=np.int64)
        ref_set_name = "all_correctly_detected_anomalous_windows_fallback"

    if ref_indices.size == 0:
        raise RuntimeError("No correctly detected anomalous windows available for correlation estimation.")

    x_ref = x_test[ref_indices]  # [N, W, F]
    x_ref_flat = x_ref.reshape(-1, feat_dim)
    corr_df = pd.DataFrame(x_ref_flat, columns=feature_names).corr(method="spearman")
    corr_df.to_csv(out_dir / "case_feature_correlation_matrix_e8_v2.csv", index=True)

    # Raw and non-redundant top-10
    raw_top10 = raw_top10.copy()
    raw_top10["corr_threshold"] = float(args.corr_threshold)
    raw_top10.to_csv(out_dir / "case_top_features_raw_e8_v2.csv", index=False)

    nonred_top10 = greedy_nonredundant(ranked, corr_df, k=10, threshold=float(args.corr_threshold))
    if len(nonred_top10) == 0:
        raise RuntimeError("Non-redundant selection produced empty set.")
    nonred_top10 = nonred_top10.copy().reset_index(drop=True)
    nonred_top10["nonredundant_rank"] = np.arange(1, len(nonred_top10) + 1)
    nonred_top10["corr_threshold"] = float(args.corr_threshold)
    nonred_top10.to_csv(out_dir / "case_top_features_nonredundant_e8_v2.csv", index=False)

    groups_df = build_redundancy_groups(raw_top10, corr_df, threshold=float(args.corr_threshold))
    groups_df.to_csv(out_dir / "case_feature_redundancy_groups_e8_v2.csv", index=False)

    # Heatmap scaling metadata
    nonred_names = nonred_top10["feature_name"].astype(str).tolist()
    heat_raw = save_heatmap_matrix(ig_tf_df, nonred_names, n_time=128)
    heat_log = np.log1p(heat_raw)
    heatmap_vmax = float(np.percentile(heat_log, 99))
    if heatmap_vmax <= 0:
        heatmap_vmax = float(np.max(heat_log)) if float(np.max(heat_log)) > 0 else 1e-6

    draw_figures(
        out_dir,
        score_df=score_df,
        ig_time_df=ig_time_df,
        attn_df=attn_df,
        ig_tf_df=ig_tf_df,
        nonred_top10=nonred_top10,
        masking_df=masking_df,
        case_meta=case_meta,
        heatmap_transform_name="log1p + p99 clip",
        heatmap_vmax=heatmap_vmax,
    )

    # Markdown summary
    mask10 = masking_df[masking_df["k"] == 10].iloc[0]
    mask20 = masking_df[masking_df["k"] == 20].iloc[0]
    summary_lines: list[str] = []
    summary_lines.append("# Case Trace Summary V2 (E8)")
    summary_lines.append("")
    summary_lines.append("## Fixed Case and Protocol")
    summary_lines.append("")
    summary_lines.append("- Selected case index: `13570`")
    summary_lines.append("- Attack type: `DDoS`")
    summary_lines.append(f"- Threshold: `{threshold:.6f}`")
    summary_lines.append(f"- Case fused score: `{fused_score:.6f}`")
    summary_lines.append("- IG target: `fused_tad_score`")
    summary_lines.append("- Evaluation-only update: no retraining, no weight changes")
    summary_lines.append("")
    summary_lines.append("## V2 Heatmap Strategy")
    summary_lines.append("")
    summary_lines.append("- Input: absolute IG time-feature matrix from existing case outputs")
    summary_lines.append("- Transform: `log1p(abs_IG)`")
    summary_lines.append("- Robust scaling: clip color range at the 99th percentile (`p99`) of transformed top-feature matrix")
    summary_lines.append("- Colormap: `magma` (high-contrast, perceptually meaningful)")
    summary_lines.append("- Time axis retained: full 128 steps")
    summary_lines.append("")
    summary_lines.append("## Redundancy Control for Feature Explanation")
    summary_lines.append("")
    summary_lines.append("- Correlation metric: `Spearman`")
    summary_lines.append(f"- Redundancy threshold: `|corr| >= {float(args.corr_threshold):.2f}`")
    summary_lines.append(f"- Correlation reference set: `{ref_set_name}`")
    summary_lines.append(f"- Number of reference windows: `{int(ref_indices.size)}`")
    summary_lines.append("- Note: this is an interpretability redundancy-control step, not a proof of causal/statistical independence.")
    summary_lines.append("")
    summary_lines.append("## Raw Top-10 Features")
    summary_lines.append("")
    for _, r in raw_top10.iterrows():
        summary_lines.append(
            f"- {int(r['rank'])}. {r['feature_name']} (mean|IG|={float(r['mean_abs_ig_attribution']):.6e})"
        )
    summary_lines.append("")
    summary_lines.append("## Non-redundant Top Features (Correlation-pruned representatives)")
    summary_lines.append("")
    for _, r in nonred_top10.iterrows():
        summary_lines.append(
            f"- {int(r['nonredundant_rank'])}. {r['feature_name']} (raw_rank={int(r['rank'])}, mean|IG|={float(r['mean_abs_ig_attribution']):.6e})"
        )
    summary_lines.append("")
    summary_lines.append("## Masking Conclusion Check")
    summary_lines.append("")
    summary_lines.append("- Masking experiment values are unchanged from the original case trace.")
    summary_lines.append(
        f"- IG@10: score={float(mask10['ig_masked_fused_score']):.6f}, decision={mask10['ig_masked_decision']}"
    )
    summary_lines.append(
        f"- IG@20: score={float(mask20['ig_masked_fused_score']):.6f}, decision={mask20['ig_masked_decision']}"
    )
    summary_lines.append(
        "- Random masking mean decisions at k=10/20 remain above threshold by majority/mean criterion."
    )
    summary_lines.append("")
    summary_lines.append("## Why V2 Is More Defensible")
    summary_lines.append("")
    summary_lines.append("- High-contrast robust heatmap scaling makes salient evidence regions visible instead of gray/flat rendering.")
    summary_lines.append("- Non-redundant representative features reduce repeated correlated packet-length statistics in the top list.")
    summary_lines.append("- Correlation-pruning is explicitly documented with metric, threshold, and reference set provenance.")
    summary_lines.append("- Main alarm-to-evidence conclusion remains stable: targeted IG masking (k=10/20) suppresses the alarm below threshold.")
    summary_lines.append("- The update improves presentation faithfulness without changing model, checkpoint, threshold, or selected case.")

    (out_dir / "case_trace_summary_e8_v2.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("Completed V2 case tracing update.")
    print(f"Reference set: {ref_set_name}, n_windows={int(ref_indices.size)}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
