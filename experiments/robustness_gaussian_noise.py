#!/usr/bin/env python3
"""Evaluation-only Gaussian-noise robustness for a strict CICIDS2017 checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
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


def parse_noise_stds(values: list[str]) -> list[float]:
    out: list[float] = []
    for item in values:
        v = float(item)
        if v < 0:
            raise ValueError(f"noise std must be >= 0, got {v}")
        out.append(float(v))
    if not out:
        raise ValueError("No noise stds provided")
    seen = set()
    dedup: list[float] = []
    for v in out:
        key = round(v, 12)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(v)
    return dedup


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    fieldnames = [
        "noise_std",
        "scenario",
        "AUC",
        "AP",
        "observed_test_benign_fpr",
        "precision",
        "recall",
        "F1",
        "TP",
        "FP",
        "TN",
        "FN",
        "threshold",
        "score_mode",
        "alpha",
        "checkpoint_path",
        "protocol_metadata",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(rows: list[dict[str, Any]], out_md: Path) -> None:
    lines: list[str] = []
    lines.append("# Gaussian Noise Robustness (Evaluation-only)")
    lines.append("")
    lines.append("| noise_std | scenario | AUC | AP | observed_test_benign_fpr | precision | recall | F1 | TP | FP | TN | FN |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{float(r['noise_std']):.2f}",
                    str(r["scenario"]),
                    f"{float(r['AUC']):.6f}",
                    f"{float(r['AP']):.6f}",
                    f"{float(r['observed_test_benign_fpr']):.6f}",
                    f"{float(r['precision']):.6f}",
                    f"{float(r['recall']):.6f}",
                    f"{float(r['F1']):.6f}",
                    str(int(r["TP"])),
                    str(int(r["FP"])),
                    str(int(r["TN"])),
                    str(int(r["FN"])),
                ]
            )
            + " |"
        )

    clean = next((r for r in rows if float(r["noise_std"]) == 0.0), None)
    mild = [r for r in rows if 0.0 < float(r["noise_std"]) <= 0.03]
    if clean is not None and mild:
        mild_f1_min = min(float(r["F1"]) for r in mild)
        mild_fpr_max = max(float(r["observed_test_benign_fpr"]) for r in mild)
        lines.append("")
        lines.append("## Threshold Usability Check")
        lines.append("")
        lines.append(
            "Fixed threshold usability under mild noise (<=0.03): "
            f"clean F1={float(clean['F1']):.6f}, min mild-noise F1={mild_f1_min:.6f}, "
            f"max mild-noise benign FPR={mild_fpr_max:.6f}."
        )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    p.add_argument("--noise-stds", nargs="+", default=["0.00", "0.01", "0.03", "0.05"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--test-batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not bool(args.independent_calibration):
        raise ValueError("This script requires --independent-calibration")
    if not bool(args.strict_no_leakage):
        raise ValueError("This script requires --strict-no-leakage")

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    noise_stds = parse_noise_stds(list(args.noise_stds))
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

    # Safety checks against accidental protocol drift.
    if not bool(ckpt_args.get("independent_calibration", False)):
        raise ValueError("Checkpoint args indicate independent_calibration=False; strict protocol mismatch.")
    if not bool(ckpt_args.get("strict_no_leakage", False) or ckpt_args.get("fit_scaler_on_model_train_only", False)):
        raise ValueError("Checkpoint args indicate strict_no_leakage=False; strict protocol mismatch.")

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
    x_test = np.asarray(ds.x_test, dtype=np.float32)
    y_true = np.asarray(ds.y_test, dtype=np.int64)

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

    test_benign_count = int((y_true == 0).sum())

    rows: list[dict[str, Any]] = []
    for idx, noise_std in enumerate(noise_stds):
        if float(noise_std) > 0.0:
            rng = np.random.default_rng(int(args.seed) + int(idx) * 9973)
            noise = rng.normal(0.0, float(noise_std), size=x_test.shape).astype(np.float32)
            x_eval = np.clip(x_test + noise, 0.0, 1.0)
            scenario = f"gaussian_{noise_std:.2f}"
        else:
            x_eval = x_test
            scenario = "clean"

        y_score = compute_anomaly_scores(
            discriminator,
            x_eval,
            device,
            int(args.test_batch_size),
            score_mode="fused",
            score_alpha=float(args.score_alpha),
            x_ref_benign=x_train_benign,
        )

        auc = float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size > 1 else float("nan")
        ap = float(average_precision_score(y_true, y_score)) if float(y_true.sum()) > 0 else 0.0
        m = metrics_at_threshold(y_true, y_score, float(args.threshold))

        fp = int(round(float(m["fp"])))
        protocol_metadata = {
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
            "fixed_threshold": float(args.threshold),
            "test_benign_count": int(test_benign_count),
            "FP/Test Benign": f"{fp}/{test_benign_count}",
            "seed": int(args.seed),
        }

        rows.append(
            {
                "noise_std": float(noise_std),
                "scenario": scenario,
                "AUC": float(auc),
                "AP": float(ap),
                "observed_test_benign_fpr": float(m["fpr"]),
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "F1": float(m["f1"]),
                "TP": int(round(float(m["tp"]))),
                "FP": int(round(float(m["fp"]))),
                "TN": int(round(float(m["tn"]))),
                "FN": int(round(float(m["fn"]))),
                "threshold": float(args.threshold),
                "score_mode": "fused",
                "alpha": float(args.score_alpha),
                "checkpoint_path": str(checkpoint_path),
                "protocol_metadata": json.dumps(protocol_metadata, ensure_ascii=True, sort_keys=True),
            }
        )

    rows.sort(key=lambda r: float(r["noise_std"]))

    out_csv = output_dir / "robustness_gaussian_noise_e8.csv"
    out_md = output_dir / "robustness_gaussian_noise_e8_summary.md"
    write_csv(rows, out_csv)
    write_summary_md(rows, out_md)

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote summary: {out_md}")
    print("Completed evaluation-only Gaussian-noise robustness. No training was executed.")


if __name__ == "__main__":
    main()
