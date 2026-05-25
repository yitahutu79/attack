#!/usr/bin/env python3
"""Post-hoc threshold sweep for an existing Proposed checkpoint (evaluation-only)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.utils import parametrize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.tcn_gan_experiment import (  # noqa: E402
    TCNDiscriminator,
    TCNGenerator,
    compute_anomaly_scores,
    metrics_at_threshold,
    threshold_from_benign_fpr,
)
from pipelines.run_tcn_cross_dataset_minimal import (  # noqa: E402
    load_windowed_dataset,
    load_windowed_dataset_independent_calibration_strict,
    select_device,
    split_windows_for_independent_calibration,
)
from dataset_loaders import (  # noqa: E402
    BENIGN_TOKENS,
    NON_FEATURE_COLS,
    _labels_from_frame,
    _read_csv,
)


def parse_target_fprs(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    out = []
    for v in vals:
        if not (0.0 < v < 1.0):
            raise ValueError(f"target_fpr must be in (0,1), got: {v}")
        out.append(float(v))
    if not out:
        raise ValueError("No valid target_fpr provided")
    return out


def to_relaxed_list(v: Any) -> list[str]:
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def default_output_paths(resume_from: Path) -> tuple[Path, Path]:
    if resume_from.parent.name == "checkpoints":
        out_dir = resume_from.parent.parent
    else:
        out_dir = resume_from.parent
    return out_dir / "posthoc_threshold_sweep.csv", out_dir / "posthoc_threshold_sweep.md"


def default_attack_breakdown_paths(resume_from: Path) -> tuple[Path, Path]:
    if resume_from.parent.name == "checkpoints":
        out_dir = resume_from.parent.parent
    else:
        out_dir = resume_from.parent
    return (
        out_dir / "posthoc_threshold_sweep_attack_type_breakdown.csv",
        out_dir / "posthoc_threshold_sweep_attack_type_breakdown.md",
    )


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


def apply_weight_parametrizations_for_state_dict(
    model: torch.nn.Module, state_dict: dict[str, Any]
) -> None:
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


def load_state_dict_compat(
    model: torch.nn.Module, state_dict: dict[str, Any], *, model_name: str
) -> None:
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


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    if not rows:
        raise ValueError("No rows to write")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, Any]], out_md: Path) -> None:
    headers = [
        "target_fpr",
        "threshold",
        "observed_test_benign_fpr",
        "FP/Test Benign",
        "precision",
        "recall",
        "F1",
        "AUC",
        "AP",
        "TP",
        "FP",
        "TN",
        "FN",
    ]
    lines: list[str] = []
    lines.append("# Post-hoc Threshold Sweep (Evaluation-only)")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{float(r['target_fpr']):.2f}",
                    f"{float(r['threshold']):.6f}",
                    f"{float(r['observed_test_benign_fpr']):.6f}",
                    str(r["FP/Test Benign"]),
                    f"{float(r['precision']):.6f}",
                    f"{float(r['recall']):.6f}",
                    f"{float(r['F1']):.6f}",
                    f"{float(r['AUC']):.6f}",
                    f"{float(r['AP']):.6f}",
                    str(int(r["TP"])),
                    str(int(r["FP"])),
                    str(int(r["TN"])),
                    str(int(r["FN"])),
                ]
            )
            + " |"
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_attack_type_md(rows: list[dict[str, Any]], out_md: Path) -> None:
    headers = [
        "attack_type",
        "target_fpr",
        "total_attack_windows",
        "detected_attack_windows",
        "missed_attack_windows",
        "recall",
        "mean_score",
        "median_score",
    ]
    lines: list[str] = []
    lines.append("# Post-hoc Attack-Type Breakdown (Evaluation-only)")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["attack_type"]),
                    f"{float(r['target_fpr']):.2f}",
                    str(int(r["total_attack_windows"])),
                    str(int(r["detected_attack_windows"])),
                    str(int(r["missed_attack_windows"])),
                    f"{float(r['recall']):.6f}",
                    f"{float(r['mean_score']):.6f}",
                    f"{float(r['median_score']):.6f}",
                ]
            )
            + " |"
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluation-only threshold sweep from an existing checkpoint.")
    p.add_argument("--resume-from", required=True, help="Checkpoint path, e.g. latest.pt")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--target-fprs", default="0.01,0.05,0.10,0.15,0.20")
    p.add_argument("--out-csv", default="")
    p.add_argument("--out-md", default="")
    p.add_argument("--out-attack-csv", default="")
    p.add_argument("--out-attack-md", default="")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--data-dir", default="", help="Optional override; defaults to checkpoint args")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    resume_from = Path(args.resume_from).resolve()
    ckpt = torch.load(resume_from, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    if not isinstance(ckpt_args, dict):
        raise ValueError("Checkpoint missing args dict; cannot reconstruct protocol.")

    dataset = str(ckpt_args.get("dataset", "cicids2017"))
    data_dir = str(args.data_dir).strip() or str(ckpt_args.get("data_dir", ""))
    train_files = to_relaxed_list(ckpt_args.get("train_files"))
    test_files = to_relaxed_list(ckpt_args.get("test_files"))
    window_size = int(ckpt_args.get("window_size", 128))
    stride = int(ckpt_args.get("stride", 16))
    anomaly_ratio = float(ckpt_args.get("anomaly_ratio", 0.15))
    score_mode = str(ckpt_args.get("score_mode", "fused"))
    alpha = float(ckpt_args.get("score_alpha", 0.24))
    independent_calibration = bool(ckpt_args.get("independent_calibration", False))
    calib_ratio = float(ckpt_args.get("calib_ratio", 0.2))
    strict_no_leakage = bool(
        ckpt_args.get("strict_no_leakage", False)
        or ckpt_args.get("fit_scaler_on_model_train_only", False)
    )
    protocol_check = ckpt.get("protocol_check", {})
    if not isinstance(protocol_check, dict):
        protocol_check = {}

    if not data_dir:
        raise ValueError("data_dir is missing in checkpoint args; provide --data-dir")
    if not train_files or not test_files:
        raise ValueError("train_files/test_files missing in checkpoint args")

    device = select_device(args.device)
    selected_device = str(device)
    print(f"Selected device: {selected_device}", flush=True)
    print("Mode: evaluation-only (no training)", flush=True)

    if strict_no_leakage and independent_calibration:
        ds, scaler_fit_source = load_windowed_dataset_independent_calibration_strict(
            dataset=dataset,
            data_dir=data_dir,
            train_files=train_files,
            test_files=test_files,
            window_size=window_size,
            stride=stride,
            anomaly_ratio=anomaly_ratio,
            calib_ratio=calib_ratio,
            scaler="minmax",
            clip_minmax=True,
            progress=True,
        )
        x_train_benign = np.asarray(ds.x_train)
        x_calib_benign = np.asarray(ds.x_calib) if ds.x_calib is not None else None
    else:
        ds = load_windowed_dataset(
            dataset=dataset,
            data_dir=data_dir,
            train_files=train_files,
            test_files=test_files,
            window_size=window_size,
            stride=stride,
            anomaly_ratio=anomaly_ratio,
            train_benign_only=True,
            scaler="minmax",
            clip_minmax=True,
            progress=True,
        )
        x_train_benign = np.asarray(ds.x_train)
        x_calib_benign = None
        scaler_fit_source = "train benign (default loader behavior)"
        if independent_calibration:
            x_train_benign, x_calib_benign = split_windows_for_independent_calibration(
                x_train_benign, calib_ratio
            )

    if x_calib_benign is None or len(x_calib_benign) == 0:
        raise ValueError("Calibration benign windows are empty; cannot perform threshold sweep.")

    feat_dim = int(x_train_benign.shape[2])
    hidden_channels = ckpt_args.get("hidden_channels", [128, 128])
    dropout = float(ckpt_args.get("dropout", 0.2))
    latent_dim = int(ckpt_args.get("latent_dim", 64))
    disc_pooling = str(ckpt_args.get("disc_pooling", "attn"))

    generator = TCNGenerator(window_size, feat_dim, latent_dim, hidden_channels, dropout).to(device)
    discriminator = TCNDiscriminator(window_size, feat_dim, hidden_channels, dropout, pooling=disc_pooling).to(device)
    g_state = ckpt.get("generator_state_dict", ckpt.get("generator"))
    d_state = ckpt.get("discriminator_state_dict", ckpt.get("discriminator"))
    if g_state is None or d_state is None:
        raise ValueError("Checkpoint missing model state dicts.")
    load_state_dict_compat(generator, g_state, model_name="generator")
    load_state_dict_compat(discriminator, d_state, model_name="discriminator")
    discriminator.eval()

    y_true = np.asarray(ds.y_test, dtype=np.int64)
    x_ref = x_train_benign if score_mode.lower().strip() != "prob" else None
    y_score = compute_anomaly_scores(
        discriminator,
        ds.x_test,
        device,
        int(args.batch_size),
        score_mode=score_mode,
        score_alpha=alpha,
        x_ref_benign=x_ref,
    )
    calib_scores = compute_anomaly_scores(
        discriminator,
        x_calib_benign,
        device,
        int(args.batch_size),
        score_mode=score_mode,
        score_alpha=alpha,
        x_ref_benign=x_ref,
    )

    auc = float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size > 1 else float("nan")
    ap = float(average_precision_score(y_true, y_score)) if float(y_true.sum()) > 0 else 0.0
    target_fprs = parse_target_fprs(args.target_fprs)
    test_benign_count = int((y_true == 0).sum())
    attack_type_by_window = build_test_window_attack_types(
        dataset=dataset,
        data_dir=data_dir,
        test_files=test_files,
        window_size=window_size,
        stride=stride,
        anomaly_ratio=anomaly_ratio,
        expected_window_labels=y_true,
    )

    scaler_source = str(protocol_check.get("scaler_fit_uses", scaler_fit_source))
    model_train_source = str(protocol_check.get("model_training_uses", "model_train_benign"))
    threshold_source = str(
        protocol_check.get("threshold_calibration_uses", "independent_calibration_benign")
    )
    tad_source = str(protocol_check.get("tad_reference_uses", "model_train_benign"))

    rows: list[dict[str, Any]] = []
    attack_rows: list[dict[str, Any]] = []
    attack_types_present = sorted(
        {
            str(t)
            for t in attack_type_by_window
            if (t is not None and str(t).strip() and str(t) != "UnknownAttack")
        }
    )
    for tfpr in target_fprs:
        threshold = float(threshold_from_benign_fpr(calib_scores, tfpr))
        m = metrics_at_threshold(y_true, y_score, threshold)
        fp = int(round(float(m.get("fp", 0.0))))
        row = {
            "target_fpr": float(tfpr),
            "threshold": float(threshold),
            "observed_test_benign_fpr": float(m["fpr"]),
            "FP/Test Benign": f"{fp}/{test_benign_count}",
            "precision": float(m["precision"]),
            "recall": float(m["recall"]),
            "F1": float(m["f1"]),
            "AUC": float(auc),
            "AP": float(ap),
            "TP": int(round(float(m["tp"]))),
            "FP": int(round(float(m["fp"]))),
            "TN": int(round(float(m["tn"]))),
            "FN": int(round(float(m["fn"]))),
            "checkpoint_path": str(resume_from),
            "dataset": dataset,
            "window_size": int(window_size),
            "stride": int(stride),
            "anomaly_ratio_threshold": float(anomaly_ratio),
            "score_mode": score_mode,
            "alpha": float(alpha),
            "strict_no_leakage": bool(strict_no_leakage),
            "independent_calibration": bool(independent_calibration),
            "calib_ratio": float(calib_ratio),
            "scaler_fit_source": scaler_source,
            "model_train_source": model_train_source,
            "threshold_calibration_source": threshold_source,
            "TAD_reference_source": tad_source,
            "selected_device": selected_device,
        }
        rows.append(row)
        pred = (np.asarray(y_score) >= threshold).astype(np.uint8)
        for attack_type in attack_types_present:
            mask = (attack_type_by_window == attack_type)
            total_attack = int(np.sum(mask))
            if total_attack <= 0:
                continue
            detected = int(np.sum((pred == 1) & mask))
            missed = int(total_attack - detected)
            scores = np.asarray(y_score)[mask]
            attack_rows.append(
                {
                    "attack_type": str(attack_type),
                    "target_fpr": float(tfpr),
                    "total_attack_windows": int(total_attack),
                    "detected_attack_windows": int(detected),
                    "missed_attack_windows": int(missed),
                    "recall": float(detected / total_attack),
                    "mean_score": float(np.mean(scores)),
                    "median_score": float(np.median(scores)),
                    "checkpoint_path": str(resume_from),
                    "dataset": dataset,
                    "window_size": int(window_size),
                    "stride": int(stride),
                    "anomaly_ratio_threshold": float(anomaly_ratio),
                    "score_mode": score_mode,
                    "alpha": float(alpha),
                    "strict_no_leakage": bool(strict_no_leakage),
                    "independent_calibration": bool(independent_calibration),
                    "calib_ratio": float(calib_ratio),
                    "scaler_fit_source": scaler_source,
                    "model_train_source": model_train_source,
                    "threshold_calibration_source": threshold_source,
                    "TAD_reference_source": tad_source,
                    "selected_device": selected_device,
                }
            )

    default_csv, default_md = default_output_paths(resume_from)
    out_csv = Path(args.out_csv).resolve() if str(args.out_csv).strip() else default_csv.resolve()
    out_md = Path(args.out_md).resolve() if str(args.out_md).strip() else default_md.resolve()
    default_attack_csv, default_attack_md = default_attack_breakdown_paths(resume_from)
    out_attack_csv = (
        Path(args.out_attack_csv).resolve()
        if str(args.out_attack_csv).strip()
        else default_attack_csv.resolve()
    )
    out_attack_md = (
        Path(args.out_attack_md).resolve()
        if str(args.out_attack_md).strip()
        else default_attack_md.resolve()
    )
    write_csv(rows, out_csv)
    write_md(rows, out_md)
    write_csv(attack_rows, out_attack_csv)
    write_attack_type_md(attack_rows, out_attack_md)

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote MD: {out_md}")
    print(f"Wrote attack CSV: {out_attack_csv}")
    print(f"Wrote attack MD: {out_attack_md}")
    print("Completed evaluation-only sweep. No training was executed.")


if __name__ == "__main__":
    main()
