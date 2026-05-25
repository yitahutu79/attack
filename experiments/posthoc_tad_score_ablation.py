#!/usr/bin/env python3
"""Evaluation-only TAD score ablation for an existing strict checkpoint."""

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
    threshold_from_benign_fpr,
)
from pipelines.run_tcn_cross_dataset_minimal import (  # noqa: E402
    load_windowed_dataset_independent_calibration_strict,
    select_device,
)


def parse_target_fprs(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    out: list[float] = []
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


def default_output_paths(resume_from: Path) -> tuple[Path, Path, Path]:
    if resume_from.parent.name == "checkpoints":
        out_dir = resume_from.parent.parent
    else:
        out_dir = resume_from.parent
    return (
        out_dir / "posthoc_tad_score_ablation.csv",
        out_dir / "posthoc_tad_score_ablation.md",
        out_dir / "posthoc_tad_score_ablation_summary.md",
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


def parse_score_modes(text: str) -> list[tuple[str, str, float | None]]:
    tokens = [x.strip().lower() for x in str(text).split(",") if x.strip()]
    if not tokens:
        raise ValueError("No score modes provided")

    out: list[tuple[str, str, float | None]] = []
    for tok in tokens:
        if tok in {"critic", "critic_only", "critic-only", "prob"}:
            out.append(("critic_only", "prob", None))
        elif tok in {
            "feature",
            "feature_only",
            "feature-only",
            "feature_deviation",
            "feature_deviation_only",
            "feat_l2",
        }:
            out.append(("feature_deviation_only", "feat_l2", None))
        elif tok in {"fused", "tad", "tad_fused"}:
            out.append(("fused", "fused", None))
        else:
            raise ValueError(f"Unsupported score mode token: {tok}")

    dedup: list[tuple[str, str, float | None]] = []
    seen = set()
    for item in out:
        if item[0] in seen:
            continue
        seen.add(item[0])
        dedup.append(item)
    return dedup


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    if not rows:
        raise ValueError("No rows to write")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "score_mode",
        "alpha",
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
        "checkpoint path",
        "protocol metadata",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, Any]], out_md: Path) -> None:
    headers = [
        "score_mode",
        "alpha",
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
        "checkpoint path",
        "protocol metadata",
    ]
    lines: list[str] = []
    lines.append("# Post-hoc TAD Score Ablation (Evaluation-only)")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for r in rows:
        alpha_text = "-" if (isinstance(r["alpha"], float) and np.isnan(r["alpha"])) else f"{float(r['alpha']):.2f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["score_mode"]),
                    alpha_text,
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
                    str(r["checkpoint path"]),
                    str(r["protocol metadata"]),
                ]
            )
            + " |"
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(rows: list[dict[str, Any]], out_md: Path) -> None:
    lines: list[str] = ["# TAD Score Ablation Summary", ""]
    eligible = [r for r in rows if float(r["observed_test_benign_fpr"]) <= 0.05]
    if not eligible:
        lines.append("No configuration meets `observed_test_benign_fpr <= 5%`.")
    else:
        best = sorted(
            eligible,
            key=lambda r: (
                -float(r["F1"]),
                -float(r["recall"]),
                -float(r["precision"]),
                float(r["observed_test_benign_fpr"]),
            ),
        )[0]
        lines.append(
            "Best trade-off under `observed_test_benign_fpr <= 5%`: "
            f"`{best['score_mode']}` at `target_fpr={float(best['target_fpr']):.2f}` "
            f"(observed FPR={float(best['observed_test_benign_fpr']):.4f}, "
            f"F1={float(best['F1']):.4f}, precision={float(best['precision']):.4f}, recall={float(best['recall']):.4f})."
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluation-only TAD score ablation from an existing checkpoint.")
    p.add_argument("--resume-from", required=True, help="Checkpoint path")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--target-fprs", default="0.15,0.20,0.25,0.30")
    p.add_argument("--score-modes", default="critic,feature,fused")
    p.add_argument("--fused-alpha", type=float, default=0.24)
    p.add_argument("--out-csv", default="")
    p.add_argument("--out-md", default="")
    p.add_argument("--out-summary-md", default="")
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
    if not (strict_no_leakage and independent_calibration):
        raise ValueError(
            "This ablation requires strict no-leakage + independent calibration, "
            "but checkpoint flags do not satisfy both constraints."
        )

    device = select_device(args.device)
    selected_device = str(device)
    print(f"Selected device: {selected_device}", flush=True)
    print("Mode: evaluation-only (no training)", flush=True)

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
    target_fprs = parse_target_fprs(args.target_fprs)
    modes = parse_score_modes(args.score_modes)
    fused_alpha = float(args.fused_alpha)
    test_benign_count = int((y_true == 0).sum())

    scaler_source = str(protocol_check.get("scaler_fit_uses", scaler_fit_source))
    model_train_source = str(protocol_check.get("model_training_uses", "model_train_benign"))
    threshold_source = str(
        protocol_check.get("threshold_calibration_uses", "independent_calibration_benign")
    )
    tad_source = str(protocol_check.get("tad_reference_uses", "model_train_benign"))

    rows: list[dict[str, Any]] = []
    for mode_name, compute_mode, _ in modes:
        mode_alpha = fused_alpha if compute_mode == "fused" else float("nan")
        x_ref = x_train_benign if compute_mode != "prob" else None
        y_score = compute_anomaly_scores(
            discriminator,
            ds.x_test,
            device,
            int(args.batch_size),
            score_mode=compute_mode,
            score_alpha=fused_alpha,
            x_ref_benign=x_ref,
        )
        calib_scores = compute_anomaly_scores(
            discriminator,
            x_calib_benign,
            device,
            int(args.batch_size),
            score_mode=compute_mode,
            score_alpha=fused_alpha,
            x_ref_benign=x_ref,
        )

        auc = float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size > 1 else float("nan")
        ap = float(average_precision_score(y_true, y_score)) if float(y_true.sum()) > 0 else 0.0

        protocol_metadata = {
            "dataset": dataset,
            "window_size": int(window_size),
            "stride": int(stride),
            "anomaly_ratio_threshold": float(anomaly_ratio),
            "strict_no_leakage": bool(strict_no_leakage),
            "independent_calibration": bool(independent_calibration),
            "calib_ratio": float(calib_ratio),
            "scaler_fit_source": scaler_source,
            "model_train_source": model_train_source,
            "threshold_calibration_source": threshold_source,
            "TAD_reference_source": tad_source,
            "selected_device": selected_device,
            "compute_score_mode": compute_mode,
        }

        for tfpr in target_fprs:
            threshold = float(threshold_from_benign_fpr(calib_scores, tfpr))
            m = metrics_at_threshold(y_true, y_score, threshold)
            fp = int(round(float(m.get("fp", 0.0))))
            rows.append(
                {
                    "score_mode": mode_name,
                    "alpha": float(mode_alpha),
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
                    "checkpoint path": str(resume_from),
                    "protocol metadata": json.dumps(protocol_metadata, ensure_ascii=True, sort_keys=True),
                }
            )

    default_csv, default_md, default_summary_md = default_output_paths(resume_from)
    out_csv = Path(args.out_csv).resolve() if str(args.out_csv).strip() else default_csv.resolve()
    out_md = Path(args.out_md).resolve() if str(args.out_md).strip() else default_md.resolve()
    out_summary_md = (
        Path(args.out_summary_md).resolve() if str(args.out_summary_md).strip() else default_summary_md.resolve()
    )

    write_csv(rows, out_csv)
    write_md(rows, out_md)
    write_summary_md(rows, out_summary_md)

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote MD: {out_md}")
    print(f"Wrote summary MD: {out_summary_md}")
    print("Completed evaluation-only TAD score ablation. No training was executed.")


if __name__ == "__main__":
    main()
