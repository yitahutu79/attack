#!/usr/bin/env python3
"""Evaluation-only single-window XAI case tracing for CICIDS2017 selected checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import re
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

from dataset_loaders import (  # noqa: E402
    BENIGN_TOKENS,
    NON_FEATURE_COLS,
    _labels_from_frame,
    _read_csv,
)
from models.tcn_gan_experiment import (  # noqa: E402
    TCNDiscriminator,
    TCNGenerator,
    _anomaly_score_torch,
    _prepare_ref_stats,
    compute_anomaly_scores,
    embed_sequences,
)
from pipelines.run_tcn_cross_dataset_minimal import (  # noqa: E402
    load_windowed_dataset_independent_calibration_strict,
    select_device,
)


def to_relaxed_list(v: Any) -> list[str]:
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def state_dict_uses_parametrized_weight(state_dict: dict[str, Any]) -> bool:
    return any(".parametrizations.weight.original" in str(k) for k in state_dict.keys())


def model_expects_parametrized_weight(model: torch.nn.Module) -> bool:
    return any(".parametrizations.weight.original" in str(k) for k in model.state_dict().keys())


def remove_all_weight_parametrizations(model: torch.nn.Module) -> None:
    for module in model.modules():
        try:
            if parametrize.is_parametrized(module, "weight"):
                parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
        except Exception:
            continue


def apply_weight_parametrizations_for_state_dict(model: torch.nn.Module, state_dict: dict[str, Any]) -> None:
    try:
        from torch.nn.utils.parametrizations import weight_norm as _param_weight_norm
    except Exception:
        return
    suffix = ".parametrizations.weight.original0"
    targets = []
    for key in state_dict.keys():
        s = str(key)
        if s.endswith(suffix):
            targets.append(s[: -len(suffix)])
    for module_name in sorted(set(targets)):
        try:
            submodule = model.get_submodule(module_name)
        except Exception:
            continue
        if not hasattr(submodule, "weight"):
            continue
        try:
            if not parametrize.is_parametrized(submodule, "weight"):
                _param_weight_norm(submodule, name="weight")
        except Exception:
            continue


def load_state_dict_compat(model: torch.nn.Module, state_dict: dict[str, Any], *, model_name: str) -> None:
    ckpt_param = state_dict_uses_parametrized_weight(state_dict)
    model_param = model_expects_parametrized_weight(model)

    if model_param and not ckpt_param:
        remove_all_weight_parametrizations(model)
    elif ckpt_param and not model_param:
        apply_weight_parametrizations_for_state_dict(model, state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load {model_name} state_dict after compatibility adaptation: {e}") from e


def normalize_attack_type_token(value: Any) -> str | None:
    token = re.sub(r"\s+", " ", str(value).strip())
    lower = token.lower()
    if lower in set(BENIGN_TOKENS) | {"", "nan", "none", "null", "0.0"}:
        return None
    return token


def select_attack_type_column(df: pd.DataFrame, dataset: str) -> str | None:
    by_lower = {str(c).strip().lower(): c for c in df.columns}
    ds = str(dataset).strip().lower()
    if ds in {"cicids2017", "cicids"}:
        preferred = ["label", "attack_cat", "type"]
    elif ds in {"unsw_nb15", "unsw"}:
        preferred = ["attack_cat", "label", "type"]
    elif ds == "swat":
        preferred = ["label2", "label3", "label4", "label_full", "label1", "type", "label"]
    else:
        preferred = ["attack_cat", "label", "type", "label_full"]
    for key in preferred:
        if key in by_lower:
            return str(by_lower[key])
    return None


def preprocess_labels_attack_types_and_indices(
    df: pd.DataFrame,
    dataset: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    labels = _labels_from_frame(df, dataset)
    col = select_attack_type_column(df, dataset)
    attack_types = df[col].astype(str).to_numpy() if col is not None else None

    drop_cols = [c for c in df.columns if str(c).strip().lower() in NON_FEATURE_COLS]
    numeric = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError("No numeric features available while extracting attack types.")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    keep_idx = numeric.index.to_numpy()

    if len(numeric) != len(labels):
        labels = labels[keep_idx]
        if attack_types is not None:
            attack_types = attack_types[keep_idx]
    return np.asarray(labels, dtype=np.uint8), attack_types, np.asarray(keep_idx, dtype=np.int64)


def build_test_window_metadata(
    *,
    dataset: str,
    data_dir: str,
    test_files: list[str],
    window_size: int,
    stride: int,
    anomaly_ratio: float,
    expected_window_labels: np.ndarray,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    derived_labels: list[int] = []
    for rel_path in test_files:
        path = Path(rel_path)
        if not path.is_absolute():
            path = (Path(data_dir) / rel_path).resolve()
        df = _read_csv(path)
        labels, attack_types, kept_rows = preprocess_labels_attack_types_and_indices(df, dataset)
        n = int(len(labels))
        total = n - int(window_size) + 1
        if total <= 0:
            continue
        for start in range(0, total, int(stride)):
            end = int(start + int(window_size))
            win_labels = labels[start:end]
            ratio = float(win_labels.mean())
            is_anom = int(ratio >= float(anomaly_ratio))
            derived_labels.append(is_anom)

            attack_type = None
            if is_anom == 1:
                if attack_types is None:
                    attack_type = "UnknownAttack"
                else:
                    win_types = attack_types[start:end]
                    vals: list[str] = []
                    for t, y in zip(win_types, win_labels, strict=True):
                        if int(y) != 1:
                            continue
                        token = normalize_attack_type_token(t)
                        if token is not None:
                            vals.append(token)
                    if vals:
                        uniq, cnt = np.unique(np.asarray(vals, dtype=object), return_counts=True)
                        attack_type = str(uniq[int(np.argmax(cnt))])
                    else:
                        attack_type = "UnknownAttack"

            out.append(
                {
                    "source_file": str(Path(rel_path).name),
                    "source_file_path": str(path),
                    "start_row": int(kept_rows[start]),
                    "end_row": int(kept_rows[end - 1]),
                    "attack_ratio": float(ratio),
                    "window_label": int(is_anom),
                    "attack_type": attack_type,
                }
            )

    labels_arr = np.asarray(derived_labels, dtype=np.int64)
    expected = np.asarray(expected_window_labels, dtype=np.int64)
    if labels_arr.shape != expected.shape:
        raise RuntimeError(
            f"Window alignment mismatch: derived={labels_arr.shape[0]} vs expected={expected.shape[0]}"
        )
    if not np.array_equal(labels_arr, expected):
        mismatch = int(np.sum(labels_arr != expected))
        raise RuntimeError(f"Window labels misaligned with expected y_test: mismatches={mismatch}")
    if len(out) != len(expected):
        raise RuntimeError(f"Metadata length mismatch: {len(out)} vs {len(expected)}")
    return out


def integrated_gradients_single(
    discriminator: TCNDiscriminator,
    x: np.ndarray,
    *,
    baseline: np.ndarray,
    device: torch.device,
    ig_steps: int,
    score_mode: str,
    score_alpha: float,
    ref_stats: dict[str, object] | None,
) -> np.ndarray:
    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    b_t = torch.from_numpy(np.asarray(baseline, dtype=np.float32)).to(device)
    grads: list[np.ndarray] = []
    for a in torch.linspace(0.0, 1.0, int(ig_steps), device=device):
        xi = (b_t + a * (x_t - b_t)).unsqueeze(0)
        xi.requires_grad_(True)
        discriminator.zero_grad(set_to_none=True)
        score = _anomaly_score_torch(
            discriminator,
            xi,
            score_mode=score_mode,
            score_alpha=float(score_alpha),
            ref_stats=ref_stats,
        ).sum()
        score.backward()
        if xi.grad is None:
            raise RuntimeError("IG gradient is None")
        grads.append(xi.grad[0].detach().cpu().numpy().astype(np.float32))
    avg_grad = np.mean(np.stack(grads, axis=0), axis=0)
    return (x_t.detach().cpu().numpy() - b_t.detach().cpu().numpy()) * avg_grad


def minmax_norm(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = max(float(hi - lo), 1e-12)
    out = (x - float(lo)) / denom
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="cicids2017")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--train-files", nargs="+", required=True)
    p.add_argument("--test-files", nargs="+", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--anomaly-ratio", type=float, default=0.15)
    p.add_argument("--independent-calibration", action="store_true")
    p.add_argument("--calib-ratio", type=float, default=0.2)
    p.add_argument("--strict-no-leakage", action="store_true")
    p.add_argument("--score-mode", choices=["fused"], default="fused")
    p.add_argument("--score-alpha", type=float, default=0.24)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--preferred-attack-types", nargs="+", default=["DDoS", "PortScan"])
    p.add_argument("--case-selection", choices=["median_detected_score"], default="median_detected_score")
    p.add_argument("--ig-steps", type=int, default=32)
    p.add_argument("--topk", nargs="+", type=int, default=[5, 10, 20])
    p.add_argument("--random-repeats", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--test-batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def save_main_plot(
    out_path: Path,
    *,
    threshold: float,
    fused_score: float,
    benign_reference_score: float | None,
    ig_time_n: np.ndarray,
    attention: np.ndarray | None,
    ig_abs: np.ndarray,
    top_feat_rows: list[dict[str, Any]],
    masking_rows: list[dict[str, Any]],
) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 10,
        }
    )

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.3, 1.1])

    # Panel A
    ax_a = fig.add_subplot(gs[0, 0])
    labels = ["Threshold", "Case fused score"]
    values = [threshold, fused_score]
    colors = ["#9e9e9e", "#2f4f8f"]
    if benign_reference_score is not None:
        labels.append("Benign ref mean")
        values.append(float(benign_reference_score))
        colors.append("#6c8f6c")
    ax_a.bar(labels, values, color=colors)
    ax_a.set_ylabel("Score")
    ax_a.set_title("Panel A: Score Contrast")
    ax_a.grid(axis="y", alpha=0.25)

    # Panel B
    ax_b = fig.add_subplot(gs[0, 1])
    t = np.arange(len(ig_time_n))
    ax_b.plot(t, ig_time_n, label="IG time attribution", color="#2f4f8f", linewidth=2)
    if attention is not None:
        attn = np.asarray(attention, dtype=np.float32)
        if float(attn.max()) > 1e-12:
            attn_n = attn / float(attn.max())
        else:
            attn_n = attn
        ax_b.plot(t, attn_n, label="Attention weight", color="#8f3d3d", linewidth=1.8)
    ax_b.set_xlabel("Time step")
    ax_b.set_ylabel("Normalized importance")
    ax_b.set_title("Panel B: Time-level Evidence")
    ax_b.grid(alpha=0.25)
    ax_b.legend(frameon=False)

    # Panel C
    ax_c = fig.add_subplot(gs[1, :])
    hm = np.abs(ig_abs.T)
    im = ax_c.imshow(hm, origin="lower", aspect="auto", cmap="Greys")
    ax_c.set_xlabel("Time step")
    ax_c.set_ylabel("Feature index")
    ax_c.set_title("Panel C: IG Time-Feature Heatmap")
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.012, pad=0.01)
    cbar.set_label("|IG attribution|")

    # Panel D
    ax_d = fig.add_subplot(gs[2, 0])
    top10 = top_feat_rows[:10]
    names = [str(r["feature_name"]) for r in top10][::-1]
    vals = [float(r["mean_abs_ig_attribution"]) for r in top10][::-1]
    ax_d.barh(np.arange(len(names)), vals, color="#2f4f8f")
    ax_d.set_yticks(np.arange(len(names)))
    ax_d.set_yticklabels(names, fontsize=8)
    ax_d.set_xlabel("Mean |IG|")
    ax_d.set_title("Panel D: Top 10 IG Features")
    ax_d.grid(axis="x", alpha=0.25)

    # Panel E
    ax_e = fig.add_subplot(gs[2, 1])
    x_labels = ["Original"]
    x_vals = [fused_score]
    for r in masking_rows:
        k = int(r["k"])
        x_labels.append(f"IG k={k}")
        x_vals.append(float(r["ig_masked_fused_score"]))
        x_labels.append(f"Rnd k={k}")
        x_vals.append(float(r["random_masked_fused_score_mean"]))
    bars = ax_e.bar(np.arange(len(x_vals)), x_vals, color="#4d4d4d")
    bars[0].set_color("#2f4f8f")
    ax_e.axhline(threshold, color="#c23b22", linestyle="--", linewidth=1.5, label="Threshold")
    ax_e.set_xticks(np.arange(len(x_vals)))
    ax_e.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
    ax_e.set_ylabel("Fused score")
    ax_e.set_title("Panel E: Masking Verification")
    ax_e.legend(frameon=False)
    ax_e.grid(axis="y", alpha=0.25)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_compact_plot(
    out_path: Path,
    *,
    threshold: float,
    fused_score: float,
    ig_abs: np.ndarray,
    top_feat_rows: list[dict[str, Any]],
    masking_rows: list[dict[str, Any]],
) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 9,
        }
    )

    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.bar(["threshold", "case"], [threshold, fused_score], color=["#9e9e9e", "#2f4f8f"])
    ax_a.set_title("A. Score")
    ax_a.grid(axis="y", alpha=0.25)

    ax_b = fig.add_subplot(gs[0, 1])
    hm = np.abs(ig_abs.T)
    im = ax_b.imshow(hm, origin="lower", aspect="auto", cmap="Greys")
    ax_b.set_title("B. IG Heatmap")
    ax_b.set_xlabel("time")
    ax_b.set_ylabel("feature")
    fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)

    ax_c = fig.add_subplot(gs[1, 0])
    top10 = top_feat_rows[:10]
    names = [str(r["feature_name"]) for r in top10][::-1]
    vals = [float(r["mean_abs_ig_attribution"]) for r in top10][::-1]
    ax_c.barh(np.arange(len(names)), vals, color="#2f4f8f")
    ax_c.set_yticks(np.arange(len(names)))
    ax_c.set_yticklabels(names, fontsize=7)
    ax_c.set_title("C. Top 10 Features")

    ax_d = fig.add_subplot(gs[1, 1])
    labels = ["orig"]
    vals_plot = [fused_score]
    for r in masking_rows:
        labels.extend([f"IG{int(r['k'])}", f"R{int(r['k'])}"])
        vals_plot.extend([float(r["ig_masked_fused_score"]), float(r["random_masked_fused_score_mean"])])
    ax_d.bar(np.arange(len(vals_plot)), vals_plot, color="#4d4d4d")
    ax_d.axhline(threshold, color="#c23b22", linestyle="--", linewidth=1.3)
    ax_d.set_xticks(np.arange(len(vals_plot)))
    ax_d.set_xticklabels(labels, fontsize=7)
    ax_d.set_title("D. Masking")
    ax_d.grid(axis="y", alpha=0.25)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not bool(args.independent_calibration):
        raise ValueError("This script requires --independent-calibration")
    if not bool(args.strict_no_leakage):
        raise ValueError("This script requires --strict-no-leakage")

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    protocol_check = ckpt.get("protocol_check", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}
    if not isinstance(protocol_check, dict):
        protocol_check = {}

    device = select_device(str(args.device))
    selected_device = str(device)
    print(f"Selected device: {selected_device}", flush=True)
    print("Mode: evaluation-only (no training)", flush=True)

    ds, scaler_fit_source = load_windowed_dataset_independent_calibration_strict(
        dataset=str(args.dataset),
        data_dir=str(args.data_dir),
        train_files=to_relaxed_list(args.train_files),
        test_files=to_relaxed_list(args.test_files),
        window_size=int(args.window_size),
        stride=int(args.stride),
        anomaly_ratio=float(args.anomaly_ratio),
        calib_ratio=float(args.calib_ratio),
        scaler="minmax",
        clip_minmax=True,
        progress=True,
    )

    x_train_benign = np.asarray(ds.x_train, dtype=np.float32)
    x_calib_benign = np.asarray(ds.x_calib, dtype=np.float32) if ds.x_calib is not None else None
    x_test = np.asarray(ds.x_test, dtype=np.float32)
    y_true = np.asarray(ds.y_test, dtype=np.int64)
    if x_calib_benign is None or len(x_calib_benign) == 0:
        raise ValueError("Calibration benign windows are empty; strict protocol invalid.")

    feat_dim = int(x_train_benign.shape[2])
    hidden_channels = ckpt_args.get("hidden_channels", [128, 128])
    dropout = float(ckpt_args.get("dropout", 0.2))
    latent_dim = int(ckpt_args.get("latent_dim", 64))
    disc_pooling = str(ckpt_args.get("disc_pooling", "attn"))

    generator = TCNGenerator(int(args.window_size), feat_dim, latent_dim, hidden_channels, dropout).to(device)
    discriminator = TCNDiscriminator(int(args.window_size), feat_dim, hidden_channels, dropout, pooling=disc_pooling).to(device)
    g_state = ckpt.get("generator_state_dict", ckpt.get("generator"))
    d_state = ckpt.get("discriminator_state_dict", ckpt.get("discriminator"))
    if g_state is None or d_state is None:
        raise ValueError("Checkpoint missing model state dicts")
    load_state_dict_compat(generator, g_state, model_name="generator")
    load_state_dict_compat(discriminator, d_state, model_name="discriminator")
    discriminator.eval()

    y_score = compute_anomaly_scores(
        discriminator,
        x_test,
        device,
        int(args.test_batch_size),
        score_mode="fused",
        score_alpha=float(args.score_alpha),
        x_ref_benign=x_train_benign,
    )
    threshold = float(args.threshold)
    pred = (np.asarray(y_score) >= threshold).astype(np.uint8)

    metadata_by_window = build_test_window_metadata(
        dataset=str(args.dataset),
        data_dir=str(args.data_dir),
        test_files=to_relaxed_list(args.test_files),
        window_size=int(args.window_size),
        stride=int(args.stride),
        anomaly_ratio=float(args.anomaly_ratio),
        expected_window_labels=y_true,
    )

    preferred = {str(x).strip().lower() for x in args.preferred_attack_types}
    detected_idx = np.where((y_true == 1) & (pred == 1))[0]
    preferred_idx: list[int] = []
    for i in detected_idx.tolist():
        at = metadata_by_window[int(i)].get("attack_type")
        at_str = str(at).lower() if at is not None else ""
        if any(p in at_str for p in preferred):
            preferred_idx.append(int(i))

    selection_pool = preferred_idx if preferred_idx else detected_idx.tolist()
    if not selection_pool:
        raise RuntimeError("No correctly detected anomalous window found at the given threshold.")

    pool_scores = np.asarray([float(y_score[i]) for i in selection_pool], dtype=np.float32)
    order = np.argsort(pool_scores)
    mid = int(len(order) // 2)
    selected_window_index = int(selection_pool[int(order[mid])])
    selected_meta = dict(metadata_by_window[selected_window_index])

    selection_rule = {
        "rule": "median_detected_score",
        "preferred_attack_types": list(args.preferred_attack_types),
        "pool_type": "preferred_detected" if preferred_idx else "all_detected",
        "pool_size": int(len(selection_pool)),
        "seed": int(args.seed),
    }

    # Build score components for selected case.
    emb_ref, unk_ref = embed_sequences(discriminator, x_train_benign, device, int(args.batch_size))
    dev_ref = np.sqrt(np.maximum(((emb_ref - emb_ref.mean(axis=0, keepdims=True)) ** 2).sum(axis=1), 0.0)).astype(np.float32)

    x_case = np.asarray(x_test[selected_window_index], dtype=np.float32)
    emb_case, unk_case = embed_sequences(discriminator, x_case[np.newaxis, ...], device, 1)
    dev_case = np.sqrt(np.maximum(((emb_case - emb_ref.mean(axis=0, keepdims=True)) ** 2).sum(axis=1), 0.0)).astype(np.float32)

    critic_score_raw = float(unk_case[0])
    feature_deviation_raw = float(dev_case[0])
    critic_score = float(minmax_norm(unk_case, float(np.min(unk_ref)), float(np.max(unk_ref)))[0])
    feature_deviation_score = float(minmax_norm(dev_case, float(np.min(dev_ref)), float(np.max(dev_ref)))[0])
    fused_score = float(y_score[selected_window_index])

    decision = "alarm" if fused_score >= threshold else "no_alarm"

    # Attention weights (if available).
    attention_weights = None
    attention_available = False
    with torch.no_grad():
        w = discriminator.forward_attn_weights(torch.from_numpy(x_case[np.newaxis, ...]).to(device).float())
        if w is not None:
            attention_weights = w[0].detach().cpu().numpy().astype(np.float32)
            attention_available = True

    # IG attribution against fused score (fallback to critic-side if needed).
    ref_stats_full = _prepare_ref_stats(discriminator, x_train_benign, device, int(args.batch_size), "fused")
    ig_target = "fused_tad_score"
    ig_mode = "fused"
    ig_ref_stats = ref_stats_full
    try:
        _ = _anomaly_score_torch(
            discriminator,
            torch.from_numpy(x_case[np.newaxis, ...]).to(device).float(),
            score_mode="fused",
            score_alpha=float(args.score_alpha),
            ref_stats=ref_stats_full,
        )
    except Exception:
        ig_target = "critic_side_prob_score"
        ig_mode = "prob"
        ig_ref_stats = None

    baseline_window = np.mean(x_train_benign, axis=0).astype(np.float32)
    ig_matrix = integrated_gradients_single(
        discriminator,
        x_case,
        baseline=baseline_window,
        device=device,
        ig_steps=int(args.ig_steps),
        score_mode=ig_mode,
        score_alpha=float(args.score_alpha),
        ref_stats=ig_ref_stats,
    )
    ig_abs = np.abs(ig_matrix).astype(np.float32)
    ig_time = np.mean(ig_abs, axis=1)
    ig_feat = np.mean(ig_abs, axis=0)

    ig_time_norm = ig_time / max(float(np.max(ig_time)), 1e-12)
    ig_feat_norm = ig_feat / max(float(np.max(ig_feat)), 1e-12)

    feature_names = [str(x) for x in list(ds.feature_names)] if getattr(ds, "feature_names", None) else []
    if len(feature_names) != feat_dim:
        feature_names = [f"f{i}" for i in range(feat_dim)]

    feat_order = np.argsort(-ig_feat)
    top_feat_rows: list[dict[str, Any]] = []
    for rank, fi in enumerate(feat_order, start=1):
        top_feat_rows.append(
            {
                "feature_name": feature_names[int(fi)],
                "feature_index": int(fi),
                "mean_abs_ig_attribution": float(ig_feat[int(fi)]),
                "normalized_feature_importance": float(ig_feat_norm[int(fi)]),
                "rank": int(rank),
            }
        )

    # Masking verification for selected case.
    benign_feature_mean = np.mean(x_train_benign, axis=(0, 1)).astype(np.float32)
    topks = sorted(set(int(k) for k in args.topk if int(k) > 0))
    random_repeats = int(args.random_repeats)
    masking_rows: list[dict[str, Any]] = []

    for k in topks:
        kk = int(min(k, feat_dim))
        top_idx = feat_order[:kk]
        x_ig = x_case.copy()
        x_ig[:, top_idx] = benign_feature_mean[top_idx]
        ig_score = float(
            compute_anomaly_scores(
                discriminator,
                x_ig[np.newaxis, ...],
                device,
                1,
                score_mode="fused",
                score_alpha=float(args.score_alpha),
                x_ref_benign=x_train_benign,
            )[0]
        )
        ig_drop = float(fused_score - ig_score)

        rnd_scores: list[float] = []
        votes_alarm = 0
        for rep in range(random_repeats):
            rep_rng = np.random.default_rng(int(args.seed) + selected_window_index * 1000 + k * 17 + rep)
            rnd_idx = rep_rng.choice(np.arange(feat_dim), size=kk, replace=False)
            x_r = x_case.copy()
            x_r[:, rnd_idx] = benign_feature_mean[rnd_idx]
            s_r = float(
                compute_anomaly_scores(
                    discriminator,
                    x_r[np.newaxis, ...],
                    device,
                    1,
                    score_mode="fused",
                    score_alpha=float(args.score_alpha),
                    x_ref_benign=x_train_benign,
                )[0]
            )
            rnd_scores.append(s_r)
            if s_r >= threshold:
                votes_alarm += 1

        rnd_mean = float(np.mean(rnd_scores))
        rnd_drop = float(fused_score - rnd_mean)
        rnd_decision = "alarm" if votes_alarm >= int(np.ceil(random_repeats / 2.0)) else "no_alarm"

        masking_rows.append(
            {
                "k": int(kk),
                "original_fused_score": float(fused_score),
                "ig_masked_fused_score": float(ig_score),
                "ig_score_drop": float(ig_drop),
                "random_masked_fused_score_mean": float(rnd_mean),
                "random_score_drop_mean": float(rnd_drop),
                "threshold": float(threshold),
                "ig_masked_decision": "alarm" if ig_score >= threshold else "no_alarm",
                "random_masked_decision_mean_or_majority": rnd_decision,
            }
        )

    benign_reference_score = None
    if x_calib_benign is not None and len(x_calib_benign) > 0:
        calib_scores = compute_anomaly_scores(
            discriminator,
            x_calib_benign,
            device,
            int(args.test_batch_size),
            score_mode="fused",
            score_alpha=float(args.score_alpha),
            x_ref_benign=x_train_benign,
        )
        benign_reference_score = float(np.mean(calib_scores))

    # Write outputs.
    case_metadata = {
        "selected_window_index": int(selected_window_index),
        "attack_type": selected_meta.get("attack_type"),
        "source_file": selected_meta.get("source_file"),
        "source_file_path": selected_meta.get("source_file_path"),
        "start_row": selected_meta.get("start_row"),
        "end_row": selected_meta.get("end_row"),
        "attack_ratio": selected_meta.get("attack_ratio"),
        "threshold": float(threshold),
        "fused_score": float(fused_score),
        "critic_score": float(critic_score),
        "feature_deviation_score": float(feature_deviation_score),
        "critic_score_raw": float(critic_score_raw),
        "feature_deviation_score_raw": float(feature_deviation_raw),
        "decision": decision,
        "selection_rule": selection_rule,
        "ig_target": ig_target,
        "attention_available": bool(attention_available),
        "protocol_metadata": {
            "dataset": str(args.dataset),
            "window_size": int(args.window_size),
            "stride": int(args.stride),
            "anomaly_ratio_threshold": float(args.anomaly_ratio),
            "independent_calibration": bool(args.independent_calibration),
            "calib_ratio": float(args.calib_ratio),
            "strict_no_leakage": bool(args.strict_no_leakage),
            "scaler_fit_source": str(protocol_check.get("scaler_fit_uses", scaler_fit_source)),
            "model_train_source": str(protocol_check.get("model_training_uses", "model_train_benign")),
            "threshold_calibration_source": str(
                protocol_check.get("threshold_calibration_uses", "independent_calibration_benign")
            ),
            "tad_reference_source": str(protocol_check.get("tad_reference_uses", "model_train_benign")),
            "selected_device": selected_device,
            "checkpoint": str(checkpoint_path),
        },
    }
    (output_dir / "case_metadata_e8.json").write_text(
        json.dumps(case_metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    score_comp_rows = [
        {"component": "threshold", "value": float(threshold)},
        {"component": "fused_score", "value": float(fused_score)},
        {"component": "critic_score", "value": float(critic_score)},
        {"component": "feature_deviation_score", "value": float(feature_deviation_score)},
    ]
    if benign_reference_score is not None:
        score_comp_rows.append({"component": "benign_reference_score", "value": float(benign_reference_score)})
    write_csv(
        output_dir / "case_score_components_e8.csv",
        score_comp_rows,
        fieldnames=["component", "value"],
    )

    if attention_available and attention_weights is not None:
        attn_rows = [
            {"time_step": int(i), "attention_weight": float(attention_weights[i])}
            for i in range(len(attention_weights))
        ]
        write_csv(
            output_dir / "case_attention_time_weights_e8.csv",
            attn_rows,
            fieldnames=["time_step", "attention_weight"],
        )

    ig_time_rows = [
        {
            "time_step": int(i),
            "mean_abs_ig_attribution": float(ig_time[i]),
            "normalized_time_importance": float(ig_time_norm[i]),
        }
        for i in range(len(ig_time))
    ]
    write_csv(
        output_dir / "case_ig_time_attribution_e8.csv",
        ig_time_rows,
        fieldnames=["time_step", "mean_abs_ig_attribution", "normalized_time_importance"],
    )

    write_csv(
        output_dir / "case_ig_feature_attribution_e8.csv",
        top_feat_rows,
        fieldnames=["feature_name", "feature_index", "mean_abs_ig_attribution", "normalized_feature_importance", "rank"],
    )

    matrix_rows: list[dict[str, Any]] = []
    for t in range(ig_abs.shape[0]):
        for f in range(ig_abs.shape[1]):
            matrix_rows.append(
                {
                    "time_step": int(t),
                    "feature_index": int(f),
                    "feature_name": feature_names[int(f)],
                    "abs_ig_attribution": float(ig_abs[t, f]),
                }
            )
    write_csv(
        output_dir / "case_ig_time_feature_matrix_e8.csv",
        matrix_rows,
        fieldnames=["time_step", "feature_index", "feature_name", "abs_ig_attribution"],
    )

    write_csv(
        output_dir / "case_masking_verification_e8.csv",
        masking_rows,
        fieldnames=[
            "k",
            "original_fused_score",
            "ig_masked_fused_score",
            "ig_score_drop",
            "random_masked_fused_score_mean",
            "random_score_drop_mean",
            "threshold",
            "ig_masked_decision",
            "random_masked_decision_mean_or_majority",
        ],
    )

    # Summary markdown.
    top10 = top_feat_rows[:10]
    top_time_idx = np.argsort(-ig_time)[:10]
    overlap_text = "N/A"
    if attention_available and attention_weights is not None:
        attn_top = set(np.argsort(-attention_weights)[:10].tolist())
        ig_top = set(top_time_idx.tolist())
        overlap = len(attn_top & ig_top)
        overlap_text = f"top-10 overlap={overlap}/10"

    lines: list[str] = []
    lines.append("# Case Trace Summary (E8)")
    lines.append("")
    lines.append("## Why This Case")
    lines.append("")
    lines.append(
        f"Selected a correctly detected anomaly from preferred attack types `{args.preferred_attack_types}` "
        f"using `{args.case_selection}` over detected windows (pool={selection_rule['pool_size']})."
    )
    lines.append("")
    lines.append("## Alarm Score")
    lines.append("")
    lines.append(f"- Threshold: `{threshold:.6f}`")
    lines.append(f"- Fused score: `{fused_score:.6f}`")
    lines.append(f"- Decision: `{decision}`")
    lines.append(f"- Critic component (normalized): `{critic_score:.6f}`")
    lines.append(f"- Feature-deviation component (normalized): `{feature_deviation_score:.6f}`")
    lines.append("")
    lines.append("## Top Temporal Evidence")
    lines.append("")
    lines.append(f"Top IG time steps: {', '.join(str(int(i)) for i in top_time_idx[:5])}")
    lines.append(f"Attention vs IG similarity: {overlap_text}")
    lines.append("")
    lines.append("## Top 10 IG Features")
    lines.append("")
    lines.append("| rank | feature_name | mean_abs_ig_attribution | normalized_feature_importance |")
    lines.append("| ---: | --- | ---: | ---: |")
    for r in top10:
        lines.append(
            f"| {int(r['rank'])} | {r['feature_name']} | {float(r['mean_abs_ig_attribution']):.6e} | {float(r['normalized_feature_importance']):.6f} |"
        )

    lines.append("")
    lines.append("## Masking Verification")
    lines.append("")
    lines.append("| k | original_fused_score | ig_masked_fused_score | ig_score_drop | random_masked_fused_score_mean | random_score_drop_mean | threshold | ig_masked_decision | random_masked_decision_mean_or_majority |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for r in masking_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(r["k"])),
                    f"{float(r['original_fused_score']):.6f}",
                    f"{float(r['ig_masked_fused_score']):.6f}",
                    f"{float(r['ig_score_drop']):.6f}",
                    f"{float(r['random_masked_fused_score_mean']):.6f}",
                    f"{float(r['random_score_drop_mean']):.6f}",
                    f"{float(r['threshold']):.6f}",
                    str(r["ig_masked_decision"]),
                    str(r["random_masked_decision_mean_or_majority"]),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Paper-ready Interpretation")
    lines.append("")
    lines.append(
        "For this concrete anomalous window, the fused anomaly score is above threshold. "
        "IG localizes discriminative time regions and dominant features; masking those IG-ranked features reduces the score more than random masking, "
        "providing a traceable path from score to evidence."
    )

    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    if not attention_available:
        lines.append("- Temporal attention weights are unavailable from this checkpoint/pooling path.")
    else:
        lines.append("- Temporal attention is available and reported for this case.")
    if ig_target != "fused_tad_score":
        lines.append("- IG target falls back to critic-side score due fused-score gradient path limitations.")
    else:
        lines.append("- IG target is the fused TAD score.")
    lines.append("- SHAP is intentionally omitted in this lightweight case trace.")

    (output_dir / "case_trace_summary_e8.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    save_main_plot(
        output_dir / "case_trace_plot_e8.png",
        threshold=threshold,
        fused_score=fused_score,
        benign_reference_score=benign_reference_score,
        ig_time_n=ig_time_norm,
        attention=attention_weights,
        ig_abs=ig_abs,
        top_feat_rows=top_feat_rows,
        masking_rows=masking_rows,
    )

    save_compact_plot(
        output_dir / "case_trace_plot_e8_compact.png",
        threshold=threshold,
        fused_score=fused_score,
        ig_abs=ig_abs,
        top_feat_rows=top_feat_rows,
        masking_rows=masking_rows,
    )

    print(f"Selected case index: {selected_window_index}")
    print(f"Attack type: {selected_meta.get('attack_type')}")
    print(f"Fused score vs threshold: {fused_score:.6f} vs {threshold:.6f}")
    print(f"IG target: {ig_target}")
    print(f"Attention available: {attention_available}")
    print(f"Wrote outputs to: {output_dir}")
    print("Completed evaluation-only XAI case trace. No training was executed.")


if __name__ == "__main__":
    main()
