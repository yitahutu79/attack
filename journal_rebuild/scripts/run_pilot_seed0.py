#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from journal_rebuild.src.calibration.thresholding import threshold_from_benign_fpr  # noqa: E402
from journal_rebuild.src.datasets.cicids2017 import (  # noqa: E402
    build_canonical_artifacts,
    load_data_manifest,
    load_split_arrays,
    load_window_manifest,
)
from journal_rebuild.src.evaluation.metrics import compute_auc_ap, metrics_at_threshold  # noqa: E402
from journal_rebuild.src.models.tcn_adversarial import (  # noqa: E402
    compare_model_configs_except_loss,
    count_parameters,
    load_checkpoint_validated,
    save_checkpoint,
    train_model,
)
from journal_rebuild.src.scoring.gan_scores import score_windows  # noqa: E402
from journal_rebuild.src.utils.config import load_yaml_like, merge_dicts  # noqa: E402
from journal_rebuild.src.utils.repro import ensure_omp_ok, git_commit, select_device, set_seed, utc_timestamp  # noqa: E402


def _run_id(prefix: str, model_name: str, seed: int) -> str:
    return f"{prefix}_{model_name}_seed{int(seed)}"


def _starts_and_labels(manifest_df: pd.DataFrame, role: str) -> tuple[np.ndarray, np.ndarray]:
    sub = manifest_df[manifest_df["split_role"] == role].reset_index(drop=True)
    return sub["split_start_offset"].to_numpy(dtype=np.int64), sub["label"].to_numpy(dtype=np.uint8)


def _window_frame(manifest_df: pd.DataFrame, role: str) -> pd.DataFrame:
    return manifest_df[manifest_df["split_role"] == role].reset_index(drop=True).copy()


def _subset_manifest(sub: pd.DataFrame, max_windows: int | None) -> pd.DataFrame:
    if max_windows is None:
        return sub.reset_index(drop=True)
    return sub.iloc[: int(max_windows)].reset_index(drop=True)


def _score_export_frame(window_df: pd.DataFrame, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    out = window_df.loc[:, ["window_id", "split_role", "source_file", "source_day", "start_row", "end_row", "label", "attack_family", "benign_flag"]].copy()
    for key, values in scores.items():
        out[key] = np.asarray(values, dtype=np.float32)
    return out


def _attack_family_recall(window_df: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict[str, float]:
    out: dict[str, float] = {}
    attack_rows = window_df[window_df["label"] == 1]
    for family, sub in attack_rows.groupby("attack_family"):
        preds = (scores[sub.index.to_numpy(dtype=np.int64)] >= float(threshold)).astype(np.uint8)
        out[str(family)] = float(preds.mean()) if len(preds) else float("nan")
    return out


def _write_training_log(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_run(
    *,
    run_id: str,
    model_name: str,
    merged_model_config: dict[str, Any],
    train_arrays: Any,
    calib_arrays: Any,
    test_arrays: Any,
    train_window_df: pd.DataFrame,
    calib_window_df: pd.DataFrame,
    test_window_df: pd.DataFrame,
    feature_names: list[str],
    data_manifest: dict[str, Any],
    data_config: dict[str, Any],
    seed: int,
    device: torch.device,
    epochs: int,
    smoke: bool,
) -> dict[str, Any]:
    set_seed(int(seed))
    train_starts = train_window_df["split_start_offset"].to_numpy(dtype=np.int64)
    calib_starts = calib_window_df["split_start_offset"].to_numpy(dtype=np.int64)
    test_starts = test_window_df["split_start_offset"].to_numpy(dtype=np.int64)
    train_labels = train_window_df["label"].to_numpy(dtype=np.uint8)
    test_labels = test_window_df["label"].to_numpy(dtype=np.uint8)

    training = train_model(
        merged_model_config,
        feature_dim=int(train_arrays.features.shape[1]),
        window_size=int(data_config["window_size"]),
        train_features=train_arrays.features,
        train_starts=train_starts,
        train_labels=train_labels,
        device=device,
        epochs=int(epochs),
    )
    infer_t0 = time.perf_counter()
    calib_scores = score_windows(
        training.discriminator,
        features=calib_arrays.features,
        starts=calib_starts,
        ref_features=train_arrays.features,
        ref_starts=train_starts,
        window_size=int(data_config["window_size"]),
        batch_size=int(merged_model_config["eval_batch_size"]),
        alpha=float(merged_model_config["alpha"]),
        device=device,
    )
    test_scores = score_windows(
        training.discriminator,
        features=test_arrays.features,
        starts=test_starts,
        ref_features=train_arrays.features,
        ref_starts=train_starts,
        window_size=int(data_config["window_size"]),
        batch_size=int(merged_model_config["eval_batch_size"]),
        alpha=float(merged_model_config["alpha"]),
        device=device,
    )
    infer_seconds = float(time.perf_counter() - infer_t0)

    threshold = threshold_from_benign_fpr(calib_scores["fused_score"], float(data_config["target_fpr"]))
    auc, ap = compute_auc_ap(test_labels, test_scores["fused_score"])
    cls = metrics_at_threshold(test_labels, test_scores["fused_score"], threshold)
    family_recall = _attack_family_recall(test_window_df, test_scores["fused_score"], threshold)
    nan_or_inf = any(
        not np.isfinite(v).all()
        for v in [
            calib_scores["fused_score"],
            test_scores["fused_score"],
            calib_scores["SD_raw"],
            test_scores["SD_raw"],
        ]
    ) or (not math.isfinite(float(threshold)))

    scores_dir = ROOT / "journal_rebuild" / "runs" / "scores" / run_id
    metrics_dir = ROOT / "journal_rebuild" / "runs" / "metrics" / run_id
    ckpt_dir = ROOT / "journal_rebuild" / "runs" / "checkpoints" / run_id
    logs_dir = ROOT / "journal_rebuild" / "runs" / "logs" / run_id
    for folder in [scores_dir, metrics_dir, ckpt_dir, logs_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    calib_export = _score_export_frame(calib_window_df, calib_scores)
    test_export = _score_export_frame(test_window_df, test_scores)
    calib_export.to_csv(scores_dir / "scores_calibration.csv", index=False)
    test_export.to_csv(scores_dir / "scores_test.csv", index=False)
    _write_training_log(logs_dir / "training_log.csv", training.training_log)

    checkpoint_path = ckpt_dir / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model_name=model_name,
        loss_type=str(merged_model_config["loss_type"]),
        seed=int(seed),
        epoch=int(epochs),
        model_config=merged_model_config,
        data_config=data_config,
        feature_names=feature_names,
        scaler_hash=str(data_manifest["scaler_hash"]),
        manifest_hash=str(data_manifest["manifest_hash"]),
        source_file_hashes=dict(data_manifest["source_file_hashes"]),
        git_commit=git_commit(ROOT),
        timestamp=utc_timestamp(),
        generator=training.generator,
        discriminator=training.discriminator,
        g_optimizer=training.g_optimizer_state,
        d_optimizer=training.d_optimizer_state,
    )
    load_checkpoint_validated(
        checkpoint_path,
        expected_model_name=model_name,
        expected_loss_type=str(merged_model_config["loss_type"]),
        expected_manifest_hash=str(data_manifest["manifest_hash"]),
        expected_scaler_hash=str(data_manifest["scaler_hash"]),
    )

    metrics = {
        "run_id": run_id,
        "model_name": model_name,
        "loss_type": str(merged_model_config["loss_type"]),
        "seed": int(seed),
        "smoke": bool(smoke),
        "epochs": int(epochs),
        "threshold": float(threshold),
        "calibration_empirical_fpr": float(np.mean(calib_scores["fused_score"] >= float(threshold))),
        "auc": float(auc),
        "ap": float(ap),
        "precision": float(cls["precision"]),
        "recall": float(cls["recall"]),
        "f1": float(cls["f1"]),
        "test_benign_fpr": float(cls["fpr"]),
        "tp": int(cls["tp"]),
        "fp": int(cls["fp"]),
        "tn": int(cls["tn"]),
        "fn": int(cls["fn"]),
        "parameter_count": int(count_parameters(training.generator) + count_parameters(training.discriminator)),
        "training_time_seconds": float(training.train_seconds),
        "inference_time_seconds": float(infer_seconds),
        "nan_or_inf_detected": bool(nan_or_inf),
        "score_std_calibration": float(np.std(calib_scores["fused_score"])),
        "score_std_test": float(np.std(test_scores["fused_score"])),
        "family_recall": family_recall,
        "manifest_hash": str(data_manifest["manifest_hash"]),
        "scaler_hash": str(data_manifest["scaler_hash"]),
        "device": str(device),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "scores_calibration_csv": str((scores_dir / "scores_calibration.csv").relative_to(ROOT)),
        "scores_test_csv": str((scores_dir / "scores_test.csv").relative_to(ROOT)),
        "training_log_csv": str((logs_dir / "training_log.csv").relative_to(ROOT)),
        "training_failed_or_degenerate": bool(training.failed_nan or nan_or_inf or np.std(test_scores["fused_score"]) < 1e-8),
    }
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def _write_pilot_report(results: list[dict[str, Any]], smoke_results: list[dict[str, Any]]) -> None:
    out_path = ROOT / "journal_rebuild" / "reports" / "model_selection" / "pilot_seed0_report.md"
    smoke_lines = [
        f"- `{item['model_name']}` smoke: `f1={item['f1']:.4f}`, `auc={item['auc']:.4f}`, `nan={item['nan_or_inf_detected']}`"
        for item in smoke_results
    ]
    formal_lines = [
        f"- `{item['model_name']}` formal: `f1={item['f1']:.4f}`, `auc={item['auc']:.4f}`, `fpr={item['test_benign_fpr']:.4f}`, `threshold={item['threshold']:.6f}`"
        for item in results
    ]
    lines = [
        "# Pilot Seed-0 Report",
        "",
        "## Smoke test",
        "",
        *smoke_lines,
        "",
        "## Formal seed-0",
        "",
        *formal_lines,
        "",
        "## Notes",
        "",
        "- `tcn_gan` and `tcn_wgan_gp` share the same generator, discriminator backbone, hidden channels, optimizer, batch size, epochs, score definition, calibration rule, and evaluation path.",
        "- The only intended training difference is adversarial loss: BCE vs Wasserstein + gradient penalty.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_build_report(data_manifest: dict[str, Any], smoke_results: list[dict[str, Any]], formal_results: list[dict[str, Any]]) -> None:
    out_path = ROOT / "journal_rebuild" / "reports" / "reproducibility" / "build_report.md"
    lines = [
        "# Journal Rebuild Build Report",
        "",
        "## Data chain",
        "",
        f"- Manifest: `{data_manifest['manifest_path']}`",
        f"- Manifest hash: `{data_manifest['manifest_hash']}`",
        f"- Scaler: `{data_manifest['scaler_path']}`",
        f"- Scaler hash: `{data_manifest['scaler_hash']}`",
        f"- Train windows: `{data_manifest['split_window_counts']['model_train_benign']}`",
        f"- Calibration windows: `{data_manifest['split_window_counts']['independent_calibration_benign']}`",
        f"- Test windows: `{data_manifest['split_window_counts']['test']}`",
        "",
        "## Smoke test status",
        "",
        *[
            f"- `{row['model_name']}`: `degenerate={row['training_failed_or_degenerate']}`, `nan={row['nan_or_inf_detected']}`"
            for row in smoke_results
        ],
        "",
        "## Formal seed-0 status",
        "",
        *[
            f"- `{row['model_name']}`: `f1={row['f1']:.4f}`, `auc={row['auc']:.4f}`, `checkpoint={row['checkpoint']}`"
            for row in formal_results
        ],
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_omp_ok()
    data_config = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "data" / "cicids2017.yaml")
    exp_config = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "experiments" / "pilot_seed0.yaml")
    data_config = merge_dicts(data_config, {"target_fpr": exp_config["target_fpr"]})

    data_manifest_path = ROOT / str(data_config["data_manifest_path"])
    if not data_manifest_path.exists():
        build_canonical_artifacts(ROOT, data_config)
    data_manifest = load_data_manifest(ROOT, data_config["data_manifest_path"])
    manifest_df = load_window_manifest(ROOT, data_manifest)
    train_arrays = load_split_arrays(ROOT, data_manifest, "model_train_benign")
    calib_arrays = load_split_arrays(ROOT, data_manifest, "independent_calibration_benign")
    test_arrays = load_split_arrays(ROOT, data_manifest, "test")
    feature_names = json.loads((ROOT / str(data_manifest["feature_path"])).read_text(encoding="utf-8"))

    train_window_df = _window_frame(manifest_df, "model_train_benign")
    calib_window_df = _window_frame(manifest_df, "independent_calibration_benign")
    test_window_df = _window_frame(manifest_df, "test")
    device = select_device(str(exp_config["device"]))

    gan_cfg = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "models" / "tcn_gan.yaml")
    wgan_cfg = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "models" / "tcn_wgan_gp.yaml")
    same, diffs = compare_model_configs_except_loss(gan_cfg, wgan_cfg)
    if not same:
        raise ValueError(f"Model configs diverge beyond loss-specific keys: {diffs}")

    smoke_results: list[dict[str, Any]] = []
    formal_results: list[dict[str, Any]] = []
    for model_name in exp_config["models"]:
        model_cfg = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "models" / f"{model_name}.yaml")
        merged = merge_dicts(model_cfg, exp_config)
        smoke_cfg = exp_config["smoke_test"]
        if bool(smoke_cfg["enabled"]):
            smoke_results.append(
                _evaluate_run(
                    run_id=_run_id("smoke", model_name, exp_config["seed"]),
                    model_name=model_name,
                    merged_model_config=merged,
                    train_arrays=train_arrays,
                    calib_arrays=calib_arrays,
                    test_arrays=test_arrays,
                    train_window_df=_subset_manifest(train_window_df, smoke_cfg["max_train_windows"]),
                    calib_window_df=_subset_manifest(calib_window_df, smoke_cfg["max_calibration_windows"]),
                    test_window_df=_subset_manifest(test_window_df, smoke_cfg["max_test_windows"]),
                    feature_names=feature_names,
                    data_manifest=data_manifest,
                    data_config=data_config,
                    seed=int(exp_config["seed"]),
                    device=device,
                    epochs=int(smoke_cfg["epochs"]),
                    smoke=True,
                )
            )
        if bool(exp_config["formal_run"]["enabled"]):
            formal_results.append(
                _evaluate_run(
                    run_id=_run_id("pilot", model_name, exp_config["seed"]),
                    model_name=model_name,
                    merged_model_config=merged,
                    train_arrays=train_arrays,
                    calib_arrays=calib_arrays,
                    test_arrays=test_arrays,
                    train_window_df=train_window_df,
                    calib_window_df=calib_window_df,
                    test_window_df=test_window_df,
                    feature_names=feature_names,
                    data_manifest=data_manifest,
                    data_config=data_config,
                    seed=int(exp_config["seed"]),
                    device=device,
                    epochs=int(exp_config["epochs"]),
                    smoke=False,
                )
            )

    summary_path = ROOT / "journal_rebuild" / "runs" / "metrics" / "pilot_seed0_summary.csv"
    pd.DataFrame(formal_results).to_csv(summary_path, index=False)
    _write_pilot_report(formal_results, smoke_results)
    _write_build_report(data_manifest, smoke_results, formal_results)
    print(summary_path)


if __name__ == "__main__":
    main()
