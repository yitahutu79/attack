#!/usr/bin/env python3
"""Evaluation-only XAI faithfulness via IG top-k masking vs random masking."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
)
from pipelines.run_tcn_cross_dataset_minimal import (  # noqa: E402
    load_windowed_dataset_independent_calibration_strict,
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


def select_device_relaxed(device_arg: str) -> tuple[torch.device, str]:
    d = str(device_arg).strip().lower()
    if d == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), "mps"
        print("[warn] --device mps requested but MPS unavailable; fallback to cpu", flush=True)
        return torch.device("cpu"), "cpu_fallback_from_mps"
    if d == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        print("[warn] --device cuda requested but CUDA unavailable; fallback to cpu", flush=True)
        return torch.device("cpu"), "cpu_fallback_from_cuda"
    if d == "cpu":
        return torch.device("cpu"), "cpu"
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda(auto)"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps(auto)"
    return torch.device("cpu"), "cpu(auto)"


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


def normalize_attack_type_token(value: Any) -> str | None:
    token = re.sub(r"\s+", " ", str(value).strip())
    lower = token.lower()
    if lower in set(BENIGN_TOKENS) | {"", "nan", "none", "null", "0.0"}:
        return None
    return token


def preprocess_labels_and_attack_types(df: pd.DataFrame, dataset: str) -> tuple[np.ndarray, np.ndarray | None]:
    labels = _labels_from_frame(df, dataset)
    col = select_attack_type_column(df, dataset)
    attack_types = df[col].astype(str).to_numpy() if col is not None else None

    drop_cols = [c for c in df.columns if str(c).strip().lower() in NON_FEATURE_COLS]
    numeric = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError("No numeric features available while extracting attack types.")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    if len(numeric) != len(labels):
        keep_idx = numeric.index.to_numpy()
        labels = labels[keep_idx]
        if attack_types is not None:
            attack_types = attack_types[keep_idx]
    return np.asarray(labels, dtype=np.uint8), attack_types


def build_test_window_attack_types(
    *,
    dataset: str,
    data_dir: str,
    test_files: list[str],
    window_size: int,
    stride: int,
    anomaly_ratio: float,
    expected_window_labels: np.ndarray,
) -> np.ndarray:
    per_window_types: list[str | None] = []
    per_window_labels: list[int] = []
    for rel_path in test_files:
        path = Path(rel_path)
        if not path.is_absolute():
            path = (Path(data_dir) / rel_path).resolve()
        df = _read_csv(path)
        labels, attack_types = preprocess_labels_and_attack_types(df, dataset)
        n = int(len(labels))
        total = n - int(window_size) + 1
        if total <= 0:
            continue
        starts = range(0, total, int(stride))
        for start in starts:
            end = int(start + int(window_size))
            win_labels = labels[start:end]
            ratio = float(win_labels.mean())
            is_anom = int(ratio >= float(anomaly_ratio))
            per_window_labels.append(is_anom)
            if is_anom == 0:
                per_window_types.append(None)
                continue
            if attack_types is None:
                per_window_types.append("UnknownAttack")
                continue
            win_types = attack_types[start:end]
            values: list[str] = []
            for t, y in zip(win_types, win_labels, strict=True):
                if int(y) != 1:
                    continue
                token = normalize_attack_type_token(t)
                if token is not None:
                    values.append(token)
            if not values:
                per_window_types.append("UnknownAttack")
            else:
                uniq, cnt = np.unique(np.asarray(values, dtype=object), return_counts=True)
                per_window_types.append(str(uniq[int(np.argmax(cnt))]))

    labels_arr = np.asarray(per_window_labels, dtype=np.int64)
    expected = np.asarray(expected_window_labels, dtype=np.int64)
    if labels_arr.shape != expected.shape:
        raise RuntimeError(
            f"Attack-type window alignment mismatch: derived={labels_arr.shape[0]} vs expected={expected.shape[0]}"
        )
    if not np.array_equal(labels_arr, expected):
        mismatch = int(np.sum(labels_arr != expected))
        raise RuntimeError(
            f"Attack-type labels do not align with test windows: mismatches={mismatch} / {labels_arr.size}"
        )
    return np.asarray(per_window_types, dtype=object)


def score_windows(
    discriminator: TCNDiscriminator,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    score_mode: str,
    score_alpha: float,
    ref_stats: dict[str, object] | None,
) -> np.ndarray:
    out: list[np.ndarray] = []
    discriminator.eval()
    with torch.no_grad():
        for start in range(0, len(x), int(batch_size)):
            xb = torch.from_numpy(np.asarray(x[start : start + int(batch_size)], dtype=np.float32)).to(device)
            s = _anomaly_score_torch(
                discriminator,
                xb,
                score_mode=score_mode,
                score_alpha=float(score_alpha),
                ref_stats=ref_stats,
            )
            out.append(s.detach().cpu().numpy().astype(np.float32))
    if not out:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(out, axis=0)


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


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
    p.add_argument("--score-mode", choices=["prob", "feat_l2", "feat_mahal", "fused"], default="fused")
    p.add_argument("--score-alpha", type=float, default=0.24)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--num-windows", type=int, default=64)
    p.add_argument("--ig-steps", type=int, default=32)
    p.add_argument("--topk", nargs="+", type=int, default=[5, 10, 20])
    p.add_argument("--random-repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--test-batch-size", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.independent_calibration or not args.strict_no_leakage:
        raise ValueError("This experiment requires --independent-calibration and --strict-no-leakage.")
    if str(args.score_mode).lower().strip() != "fused":
        raise ValueError("This experiment is intended for --score-mode fused.")

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device, actual_device_label = select_device_relaxed(args.device)
    print(f"Requested device: {args.device}; selected: {actual_device_label}", flush=True)
    print("Mode: evaluation-only (no training)", flush=True)

    ckpt_path = Path(args.checkpoint).resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    protocol_check = ckpt.get("protocol_check", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}
    if not isinstance(protocol_check, dict):
        protocol_check = {}

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
    y_test = np.asarray(ds.y_test, dtype=np.uint8)
    if x_calib_benign is None or len(x_calib_benign) == 0:
        raise ValueError("Calibration benign windows are empty in strict independent calibration mode.")

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
        raise ValueError("Checkpoint missing generator/discriminator state dicts.")
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
    detected_anom_idx = np.where((y_test == 1) & (pred == 1))[0]

    n_select = min(int(args.num_windows), int(len(detected_anom_idx)))
    if n_select <= 0:
        raise RuntimeError("No correctly detected anomalous windows found at the given threshold.")
    rng = np.random.default_rng(int(args.seed))
    selected_idx = np.sort(rng.choice(detected_anom_idx, size=n_select, replace=False))
    print(
        f"Detected anomalous windows at threshold={threshold:.6f}: {len(detected_anom_idx)}; selected={n_select}",
        flush=True,
    )

    attack_types = build_test_window_attack_types(
        dataset=str(args.dataset),
        data_dir=str(args.data_dir),
        test_files=to_relaxed_list(args.test_files),
        window_size=int(args.window_size),
        stride=int(args.stride),
        anomaly_ratio=float(args.anomaly_ratio),
        expected_window_labels=y_test,
    )

    ref_stats_full = _prepare_ref_stats(
        discriminator,
        x_train_benign,
        device,
        int(args.batch_size),
        "fused",
    )

    ig_target = "fused_tad_score"
    ig_score_mode = "fused"
    ig_ref_stats = ref_stats_full
    try:
        _ = _anomaly_score_torch(
            discriminator,
            torch.from_numpy(x_test[:1]).to(device),
            score_mode=ig_score_mode,
            score_alpha=float(args.score_alpha),
            ref_stats=ig_ref_stats,
        )
    except Exception:
        ig_target = "critic_side_prob_score"
        ig_score_mode = "prob"
        ig_ref_stats = None

    feature_names = [str(x) for x in list(ds.feature_names)] if getattr(ds, "feature_names", None) else []
    if len(feature_names) != feat_dim:
        feature_names = [f"f{i}" for i in range(feat_dim)]

    benign_feature_mean = np.mean(x_train_benign, axis=(0, 1)).astype(np.float32)
    baseline_window = np.mean(x_train_benign, axis=0).astype(np.float32)

    selected_x = x_test[selected_idx]
    original_selected_scores = np.asarray(y_score)[selected_idx].astype(np.float32)

    ig_feature_importances: list[np.ndarray] = []
    detail_rows: list[dict[str, Any]] = []

    topks = [int(k) for k in args.topk if int(k) > 0]
    topks = sorted(set(topks))
    random_repeats = int(args.random_repeats)

    for local_i, global_idx in enumerate(selected_idx):
        window = np.asarray(selected_x[local_i], dtype=np.float32)
        original_score = float(original_selected_scores[local_i])

        ig_attr = integrated_gradients_single(
            discriminator,
            window,
            baseline=baseline_window,
            device=device,
            ig_steps=int(args.ig_steps),
            score_mode=ig_score_mode,
            score_alpha=float(args.score_alpha),
            ref_stats=ig_ref_stats,
        )
        feat_imp = np.mean(np.abs(ig_attr), axis=0).astype(np.float32)
        ig_feature_importances.append(feat_imp)

        order = np.argsort(-feat_imp)
        all_feat_indices = np.arange(feat_dim)

        for k in topks:
            kk = int(min(k, feat_dim))
            top_idx = order[:kk]

            ig_masked = window.copy()
            ig_masked[:, top_idx] = benign_feature_mean[top_idx]
            ig_masked_score = float(
                score_windows(
                    discriminator,
                    ig_masked[np.newaxis, ...],
                    device=device,
                    batch_size=1,
                    score_mode="fused",
                    score_alpha=float(args.score_alpha),
                    ref_stats=ref_stats_full,
                )[0]
            )
            ig_drop = float(original_score - ig_masked_score)

            random_scores: list[float] = []
            for rep in range(random_repeats):
                rep_rng = np.random.default_rng(int(args.seed) + int(global_idx) * 1000 + k * 17 + rep)
                rnd_idx = rep_rng.choice(all_feat_indices, size=kk, replace=False)
                rnd_masked = window.copy()
                rnd_masked[:, rnd_idx] = benign_feature_mean[rnd_idx]
                rnd_score = float(
                    score_windows(
                        discriminator,
                        rnd_masked[np.newaxis, ...],
                        device=device,
                        batch_size=1,
                        score_mode="fused",
                        score_alpha=float(args.score_alpha),
                        ref_stats=ref_stats_full,
                    )[0]
                )
                random_scores.append(rnd_score)

            random_masked_score_mean = float(np.mean(random_scores)) if random_scores else float("nan")
            random_drop_mean = float(original_score - random_masked_score_mean)
            delta_drop = float(ig_drop - random_drop_mean)

            attack_type = "UnknownAttack"
            if attack_types.size > int(global_idx):
                raw_type = attack_types[int(global_idx)]
                attack_type = str(raw_type) if raw_type is not None else "BENIGN"

            detail_rows.append(
                {
                    "window_index": int(global_idx),
                    "attack_label": int(y_test[int(global_idx)]),
                    "attack_type": attack_type,
                    "original_score": float(original_score),
                    "k": int(kk),
                    "ig_masked_score": float(ig_masked_score),
                    "ig_score_drop": float(ig_drop),
                    "random_masked_score_mean": float(random_masked_score_mean),
                    "random_score_drop_mean": float(random_drop_mean),
                    "delta_drop": float(delta_drop),
                    "ig_topk_features": ";".join(feature_names[int(i)] for i in top_idx),
                    "ig_target": ig_target,
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for k in topks:
        rows_k = [r for r in detail_rows if int(r["k"]) == int(k)]
        if not rows_k:
            continue
        original_mean = float(np.mean([float(r["original_score"]) for r in rows_k]))
        ig_masked_mean = float(np.mean([float(r["ig_masked_score"]) for r in rows_k]))
        ig_drop_mean = float(np.mean([float(r["ig_score_drop"]) for r in rows_k]))
        random_masked_mean = float(np.mean([float(r["random_masked_score_mean"]) for r in rows_k]))
        random_drop_mean = float(np.mean([float(r["random_score_drop_mean"]) for r in rows_k]))
        delta_drop_mean = float(np.mean([float(r["delta_drop"]) for r in rows_k]))
        ig_beats_random_frac = float(np.mean([1.0 if float(r["delta_drop"]) > 0 else 0.0 for r in rows_k]))

        summary_rows.append(
            {
                "k": int(k),
                "n_windows": int(len(rows_k)),
                "original_mean_fused_score": original_mean,
                "ig_masked_mean_score": ig_masked_mean,
                "ig_mean_score_drop": ig_drop_mean,
                "random_masked_mean_score": random_masked_mean,
                "random_mean_score_drop": random_drop_mean,
                "delta_drop": delta_drop_mean,
                "ig_beats_random_fraction": ig_beats_random_frac,
                "random_repeats": int(random_repeats),
                "ig_target": ig_target,
            }
        )

    top_feature_rows: list[dict[str, Any]] = []
    if ig_feature_importances:
        ig_mat = np.stack(ig_feature_importances, axis=0)
        mean_imp = ig_mat.mean(axis=0)
        std_imp = ig_mat.std(axis=0)
        top_count_5 = np.zeros(feat_dim, dtype=np.int64)
        for row in ig_mat:
            idx = np.argsort(-row)[: min(5, feat_dim)]
            top_count_5[idx] += 1
        order = np.argsort(-mean_imp)
        for rank, fi in enumerate(order, start=1):
            top_feature_rows.append(
                {
                    "rank": int(rank),
                    "feature": feature_names[int(fi)],
                    "mean_abs_ig_importance": float(mean_imp[int(fi)]),
                    "std_abs_ig_importance": float(std_imp[int(fi)]),
                    "top5_count": int(top_count_5[int(fi)]),
                    "top5_coverage": float(top_count_5[int(fi)] / max(len(ig_mat), 1)),
                }
            )

    detail_csv = output_dir / "xai_faithfulness_ig_masking_e8.csv"
    summary_csv = output_dir / "xai_faithfulness_ig_masking_e8_summary.csv"
    summary_md = output_dir / "xai_faithfulness_ig_masking_e8_summary.md"
    top_feat_csv = output_dir / "top_features_ig_e8.csv"

    write_csv(
        detail_csv,
        detail_rows,
        fieldnames=[
            "window_index",
            "attack_label",
            "attack_type",
            "original_score",
            "k",
            "ig_masked_score",
            "ig_score_drop",
            "random_masked_score_mean",
            "random_score_drop_mean",
            "delta_drop",
            "ig_topk_features",
            "ig_target",
        ],
    )

    write_csv(
        summary_csv,
        summary_rows,
        fieldnames=[
            "k",
            "n_windows",
            "original_mean_fused_score",
            "ig_masked_mean_score",
            "ig_mean_score_drop",
            "random_masked_mean_score",
            "random_mean_score_drop",
            "delta_drop",
            "ig_beats_random_fraction",
            "random_repeats",
            "ig_target",
        ],
    )

    write_csv(
        top_feat_csv,
        top_feature_rows,
        fieldnames=[
            "rank",
            "feature",
            "mean_abs_ig_importance",
            "std_abs_ig_importance",
            "top5_count",
            "top5_coverage",
        ],
    )

    lines: list[str] = []
    lines.append("# XAI Faithfulness (IG Masking) Summary")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Checkpoint: `{ckpt_path}`")
    lines.append(f"- Requested device: `{args.device}`; selected device: `{actual_device_label}`")
    lines.append(f"- Score mode (evaluation): `fused`, alpha={float(args.score_alpha):.2f}")
    lines.append(f"- IG target: `{ig_target}`")
    lines.append(f"- Threshold: `{threshold:.6f}`")
    lines.append(f"- Selected correctly detected anomalous windows: `{n_select}` / `{len(detected_anom_idx)}`")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("- scaler fit uses model_train_benign only")
    lines.append("- model training uses model_train_benign")
    lines.append("- threshold calibration uses independent_calibration_benign")
    lines.append("- TAD reference uses model_train_benign")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| k | n_windows | original_mean_fused_score | ig_masked_mean_score | ig_mean_score_drop | random_masked_mean_score | random_mean_score_drop | delta_drop | ig_beats_random_fraction |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(r["k"])),
                    str(int(r["n_windows"])),
                    f"{float(r['original_mean_fused_score']):.6f}",
                    f"{float(r['ig_masked_mean_score']):.6f}",
                    f"{float(r['ig_mean_score_drop']):.6f}",
                    f"{float(r['random_masked_mean_score']):.6f}",
                    f"{float(r['random_mean_score_drop']):.6f}",
                    f"{float(r['delta_drop']):.6f}",
                    f"{float(r['ig_beats_random_fraction']):.6f}",
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    metadata = {
        "dataset": str(args.dataset),
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "anomaly_ratio_threshold": float(args.anomaly_ratio),
        "independent_calibration": bool(args.independent_calibration),
        "calib_ratio": float(args.calib_ratio),
        "strict_no_leakage": bool(args.strict_no_leakage),
        "score_mode": "fused",
        "score_alpha": float(args.score_alpha),
        "threshold": float(threshold),
        "seed": int(args.seed),
        "ig_steps": int(args.ig_steps),
        "random_repeats": int(random_repeats),
        "scaler_fit_source": str(protocol_check.get("scaler_fit_uses", scaler_fit_source)),
        "model_train_source": str(protocol_check.get("model_training_uses", "model_train_benign")),
        "threshold_calibration_source": str(
            protocol_check.get("threshold_calibration_uses", "independent_calibration_benign")
        ),
        "tad_reference_source": str(protocol_check.get("tad_reference_uses", "model_train_benign")),
    }
    lines.append("```json")
    lines.append(json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True))
    lines.append("```")

    if ig_target != "fused_tad_score":
        lines.append("")
        lines.append("## Limitation")
        lines.append("")
        lines.append(
            "Integrated Gradients for full fused score was unavailable in this run; IG attribution used critic-side score (`prob`) only."
        )

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {detail_csv}")
    print(f"Wrote: {summary_md}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {top_feat_csv}")
    print("Completed evaluation-only XAI faithfulness experiment. No training was executed.")


if __name__ == "__main__":
    main()
