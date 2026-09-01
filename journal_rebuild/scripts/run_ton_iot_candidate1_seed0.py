#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from journal_rebuild.src.datasets.ton_iot_candidate1 import (  # noqa: E402
    build_candidate1_artifacts,
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
from journal_rebuild.src.utils.hashing import sha256_json  # noqa: E402
from journal_rebuild.src.utils.repro import ensure_omp_ok, git_commit, select_device, set_seed, utc_timestamp  # noqa: E402


REPORT_ROOT = ROOT / "journal_rebuild" / "reports" / "ton_iot_candidate1"


def _run_id(prefix: str, model_name: str, seed: int) -> str:
    return f"ton_iot_candidate1_{prefix}_{model_name}_seed{int(seed)}"


def _window_frame(manifest_df: pd.DataFrame, role: str) -> pd.DataFrame:
    return manifest_df[manifest_df["split_role"] == role].reset_index(drop=True).copy()


def _subset_manifest(sub: pd.DataFrame, max_windows: int | None) -> pd.DataFrame:
    if max_windows is None:
        return sub.reset_index(drop=True)
    return sub.iloc[: int(max_windows)].reset_index(drop=True)


def _subset_test_manifest_balanced(sub: pd.DataFrame, max_windows: int | None) -> pd.DataFrame:
    if max_windows is None or len(sub) <= int(max_windows):
        return sub.reset_index(drop=True)
    max_windows = int(max_windows)
    benign = sub[sub["label"] == 0]
    attack = sub[sub["label"] == 1]
    if len(benign) == 0 or len(attack) == 0:
        return _subset_manifest(sub, max_windows)
    half = max_windows // 2
    benign_take = min(len(benign), max(1, half))
    attack_take = min(len(attack), max(1, max_windows - benign_take))
    chosen = pd.concat([benign.iloc[:benign_take], attack.iloc[:attack_take]], ignore_index=False)
    if len(chosen) < max_windows:
        remainder = sub.drop(index=chosen.index)
        chosen = pd.concat([chosen, remainder.iloc[: max_windows - len(chosen)]], ignore_index=False)
    return chosen.sort_index().reset_index(drop=True)


def _score_export_frame(window_df: pd.DataFrame, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    keep = [
        "window_id",
        "split_role",
        "source_file",
        "start_row",
        "end_row",
        "start_ts",
        "end_ts",
        "source_rows",
        "label",
        "attack_type",
        "attack_family",
        "benign_flag",
        "attack_ratio",
        "benign_ratio",
        "mixed_label_window",
        "src_entity_summary",
        "dst_entity_summary",
    ]
    out = window_df.loc[:, keep].copy()
    for key, values in scores.items():
        out[key] = np.asarray(values, dtype=np.float32)
    return out


def _write_training_log(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _attack_family_recall(window_df: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict[str, float]:
    out: dict[str, float] = {}
    attack_rows = window_df[window_df["label"] == 1]
    for family, sub in attack_rows.groupby("attack_type"):
        preds = (scores[sub.index.to_numpy(dtype=np.int64)] >= float(threshold)).astype(np.uint8)
        out[str(family)] = float(preds.mean()) if len(preds) else float("nan")
    return out


def _write_resolved_config(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _evaluate_run(
    *,
    run_id: str,
    model_name: str,
    merged_model_config: dict[str, Any],
    data_config: dict[str, Any],
    exp_config: dict[str, Any],
    data_manifest: dict[str, Any],
    feature_names: list[str],
    train_arrays: Any,
    reference_arrays: Any,
    calib_arrays: Any,
    test_arrays: Any,
    train_window_df: pd.DataFrame,
    reference_window_df: pd.DataFrame,
    calib_window_df: pd.DataFrame,
    test_window_df: pd.DataFrame,
    device: torch.device,
    seed: int,
    epochs: int,
    smoke: bool,
) -> dict[str, Any]:
    set_seed(int(seed))
    train_starts = train_window_df["split_start_offset"].to_numpy(dtype=np.int64)
    ref_starts = reference_window_df["split_start_offset"].to_numpy(dtype=np.int64)
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
        ref_features=reference_arrays.features,
        ref_starts=ref_starts,
        window_size=int(data_config["window_size"]),
        batch_size=int(merged_model_config["eval_batch_size"]),
        alpha=float(merged_model_config["alpha"]),
        device=device,
    )
    test_scores = score_windows(
        training.discriminator,
        features=test_arrays.features,
        starts=test_starts,
        ref_features=reference_arrays.features,
        ref_starts=ref_starts,
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
        for v in [calib_scores["fused_score"], test_scores["fused_score"], calib_scores["SD_raw"], test_scores["SD_raw"]]
    ) or (not math.isfinite(float(threshold)))

    ckpt_dir = ROOT / "journal_rebuild" / "runs" / "checkpoints" / run_id
    scores_dir = ROOT / "journal_rebuild" / "runs" / "scores" / run_id
    metrics_dir = ROOT / "journal_rebuild" / "runs" / "metrics" / run_id
    logs_dir = ROOT / "journal_rebuild" / "runs" / "logs" / run_id
    for folder in [ckpt_dir, scores_dir, metrics_dir, logs_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    research_config = {
        "dataset_name": str(data_config["dataset_name"]),
        "manifest_hash": str(data_manifest["manifest_hash"]),
        "scaler_hash": str(data_manifest["scaler_hash"]),
        "window_manifest_hash": str(data_manifest["window_manifest_hash"]),
        "window_size": int(data_config["window_size"]),
        "stride": int(data_config["stride"]),
        "target_fpr": float(data_config["target_fpr"]),
        "model_name": model_name,
        "loss_type": str(merged_model_config["loss_type"]),
        "seed": int(seed),
        "epochs": int(epochs),
        "alpha": float(merged_model_config["alpha"]),
        "feature_names": feature_names,
    }
    runtime_metadata = {
        "run_id": run_id,
        "device": str(device),
        "smoke": bool(smoke),
        "timestamp": utc_timestamp(),
        "git_commit": git_commit(ROOT),
    }
    resolved = {
        "research_config": research_config,
        "runtime_metadata": runtime_metadata,
        "research_config_hash": sha256_json(research_config),
        "runtime_metadata_hash": sha256_json(runtime_metadata),
        "data_config": data_config,
        "experiment_config": exp_config,
        "model_config": merged_model_config,
    }
    _write_resolved_config(ckpt_dir / "resolved_config.yaml", resolved)

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

    benign_mask = test_labels == 0
    attack_mask = test_labels == 1
    score_mean_benign = float(test_scores["fused_score"][benign_mask].mean()) if np.any(benign_mask) else float("nan")
    score_mean_attack = float(test_scores["fused_score"][attack_mask].mean()) if np.any(attack_mask) else float("nan")
    preds = (test_scores["fused_score"] >= threshold).astype(np.uint8)
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
        "score_mean_benign": score_mean_benign,
        "score_mean_attack": score_mean_attack,
        "all_same_prediction": bool(np.unique(preds).size <= 1),
        "family_recall": family_recall,
        "backdoor_recall": float(family_recall.get("backdoor", float("nan"))),
        "mitm_recall": float(family_recall.get("mitm", float("nan"))),
        "manifest_hash": str(data_manifest["manifest_hash"]),
        "scaler_hash": str(data_manifest["scaler_hash"]),
        "window_manifest_hash": str(data_manifest["window_manifest_hash"]),
        "research_config_hash": resolved["research_config_hash"],
        "runtime_metadata_hash": resolved["runtime_metadata_hash"],
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "resolved_config": str((ckpt_dir / "resolved_config.yaml").relative_to(ROOT)),
        "scores_calibration_csv": str((scores_dir / "scores_calibration.csv").relative_to(ROOT)),
        "scores_test_csv": str((scores_dir / "scores_test.csv").relative_to(ROOT)),
        "training_log_csv": str((logs_dir / "training_log.csv").relative_to(ROOT)),
        "training_failed_or_degenerate": bool(training.failed_nan or nan_or_inf or np.std(test_scores["fused_score"]) < 1e-8),
    }
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (metrics_dir / "run_metadata.json").write_text(json.dumps(runtime_metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def _write_report(data_manifest: dict[str, Any], smoke_results: list[dict[str, Any]], formal_results: list[dict[str, Any]], window_manifest: pd.DataFrame) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    test_sub = window_manifest[window_manifest["split_role"] == "test"]
    test_benign_windows = int((test_sub["label"] == 0).sum())
    suspicious = any(row["auc"] > 0.995 or row["f1"] > 0.995 for row in formal_results)
    checked_rows = formal_results if formal_results else smoke_results
    lines = [
        "# TON_IoT Candidate 1 Pilot Seed-0 Report",
        "",
        "## Split / window sanity",
        "",
        f"- Train windows: `{data_manifest['split_window_counts']['model_train_benign']}`",
        f"- Reference windows: `{data_manifest['split_window_counts']['reference_benign']}`",
        f"- Calibration windows: `{data_manifest['split_window_counts']['independent_calibration_benign']}`",
        f"- Test windows: `{data_manifest['split_window_counts']['test']}`",
        f"- Test benign windows: `{test_benign_windows}`",
        "",
        "## Smoke test",
        "",
        *[
            f"- `{row['model_name']}`: `degenerate={row['training_failed_or_degenerate']}`, `nan={row['nan_or_inf_detected']}`, `threshold={row['threshold']:.6f}`"
            for row in smoke_results
        ],
        "",
        "## Formal seed-0",
        "",
        *(
            [
                f"- `{row['model_name']}`: `AUC={row['auc']:.4f}`, `AP={row['ap']:.4f}`, `F1={row['f1']:.4f}`, `test_FPR={row['test_benign_fpr']:.4f}`, `BackdoorRecall={row['backdoor_recall']:.4f}`, `MITMRecall={row['mitm_recall']:.4f}`"
                for row in formal_results
            ]
            if formal_results
            else ["- Not completed in the current harness because only CPU is available here; use the prepared runner locally for formal seed-0."]
        ),
        "",
        "## Validity checks",
        "",
        f"- Test benign window count enough for FPR estimation: `{test_benign_windows >= 100}`",
        f"- Any degenerate run in completed runs: `{any(row['training_failed_or_degenerate'] for row in checked_rows)}`",
        f"- Any all-same prediction in completed runs: `{any(row['all_same_prediction'] for row in checked_rows)}`",
        f"- Any non-finite threshold in completed runs: `{any(not math.isfinite(float(row['threshold'])) for row in checked_rows)}`",
        f"- Suspiciously high performance suggesting residual leakage: `{suspicious}`",
        f"- Score direction consistent with anomaly semantics (`attack_mean > benign_mean`) for completed runs: `{all(row['score_mean_attack'] > row['score_mean_benign'] for row in checked_rows if math.isfinite(row['score_mean_attack']) and math.isfinite(row['score_mean_benign']))}`",
    ]
    (REPORT_ROOT / "pilot_seed0_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--formal-only", action="store_true")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.smoke_only and args.formal_only:
        raise ValueError("Cannot set both --smoke-only and --formal-only")

    ensure_omp_ok()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    data_config = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "data" / "ton_iot_candidate1.yaml")
    exp_config = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "experiments" / "ton_iot_candidate1_seed0.yaml")
    if args.models:
        exp_config["models"] = list(args.models)
    if args.device is not None:
        exp_config["device"] = str(args.device)
    if args.smoke_only:
        exp_config["formal_run"]["enabled"] = False
        exp_config["smoke_test"]["enabled"] = True
    if args.formal_only:
        exp_config["smoke_test"]["enabled"] = False
        exp_config["formal_run"]["enabled"] = True
    data_config = merge_dicts(data_config, {"target_fpr": exp_config["target_fpr"]})

    data_manifest_path = ROOT / str(data_config["data_manifest_path"])
    if not data_manifest_path.exists():
        build_candidate1_artifacts(ROOT, data_config)
    data_manifest = load_data_manifest(ROOT, data_config["data_manifest_path"])
    manifest_df = load_window_manifest(ROOT, data_manifest)
    train_arrays = load_split_arrays(ROOT, data_manifest, "model_train_benign")
    reference_arrays = load_split_arrays(ROOT, data_manifest, "reference_benign")
    calib_arrays = load_split_arrays(ROOT, data_manifest, "independent_calibration_benign")
    test_arrays = load_split_arrays(ROOT, data_manifest, "test")
    feature_names = json.loads((ROOT / str(data_manifest["feature_path"])).read_text(encoding="utf-8"))

    train_window_df = _window_frame(manifest_df, "model_train_benign")
    reference_window_df = _window_frame(manifest_df, "reference_benign")
    calib_window_df = _window_frame(manifest_df, "independent_calibration_benign")
    test_window_df = _window_frame(manifest_df, "test")

    gan_cfg = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "models" / "tcn_gan.yaml")
    wgan_cfg = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "models" / "tcn_wgan_gp.yaml")
    same, diffs = compare_model_configs_except_loss(gan_cfg, wgan_cfg)
    if not same:
        raise ValueError(f"Model configs diverge beyond loss-specific keys: {diffs}")

    device = select_device(str(exp_config["device"]))
    smoke_results: list[dict[str, Any]] = []
    formal_results: list[dict[str, Any]] = []
    for model_name in exp_config["models"]:
        model_cfg = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "models" / f"{model_name}.yaml")
        merged = merge_dicts(model_cfg, exp_config)
        smoke_cfg = exp_config["smoke_test"]
        if bool(smoke_cfg["enabled"]):
            smoke_run_id = _run_id("smoke", model_name, exp_config["seed"])
            if args.skip_existing and (ROOT / "journal_rebuild" / "runs" / "metrics" / smoke_run_id / "metrics.json").exists():
                smoke_results.append(json.loads((ROOT / "journal_rebuild" / "runs" / "metrics" / smoke_run_id / "metrics.json").read_text(encoding="utf-8")))
            else:
                smoke_results.append(
                    _evaluate_run(
                        run_id=smoke_run_id,
                        model_name=model_name,
                        merged_model_config=merged,
                        data_config=data_config,
                        exp_config=exp_config,
                        data_manifest=data_manifest,
                        feature_names=feature_names,
                        train_arrays=train_arrays,
                        reference_arrays=reference_arrays,
                        calib_arrays=calib_arrays,
                        test_arrays=test_arrays,
                        train_window_df=_subset_manifest(train_window_df, smoke_cfg["max_train_windows"]),
                        reference_window_df=_subset_manifest(reference_window_df, smoke_cfg["max_reference_windows"]),
                        calib_window_df=_subset_manifest(calib_window_df, smoke_cfg["max_calibration_windows"]),
                        test_window_df=_subset_test_manifest_balanced(test_window_df, smoke_cfg["max_test_windows"]),
                        device=device,
                        seed=int(exp_config["seed"]),
                        epochs=int(smoke_cfg["epochs"]),
                        smoke=True,
                    )
                )
        if bool(exp_config["formal_run"]["enabled"]):
            formal_run_id = _run_id("pilot", model_name, exp_config["seed"])
            if args.skip_existing and (ROOT / "journal_rebuild" / "runs" / "metrics" / formal_run_id / "metrics.json").exists():
                formal_results.append(json.loads((ROOT / "journal_rebuild" / "runs" / "metrics" / formal_run_id / "metrics.json").read_text(encoding="utf-8")))
            else:
                formal_results.append(
                    _evaluate_run(
                        run_id=formal_run_id,
                        model_name=model_name,
                        merged_model_config=merged,
                        data_config=data_config,
                        exp_config=exp_config,
                        data_manifest=data_manifest,
                        feature_names=feature_names,
                        train_arrays=train_arrays,
                        reference_arrays=reference_arrays,
                        calib_arrays=calib_arrays,
                        test_arrays=test_arrays,
                        train_window_df=train_window_df,
                        reference_window_df=reference_window_df,
                        calib_window_df=calib_window_df,
                        test_window_df=test_window_df,
                        device=device,
                        seed=int(exp_config["seed"]),
                        epochs=int(exp_config["epochs"]),
                        smoke=False,
                    )
                )

    summary_rows = []
    for row in formal_results:
        summary_rows.append(
            {
                "model_name": row["model_name"],
                "seed": row["seed"],
                "threshold": row["threshold"],
                "auc": row["auc"],
                "ap": row["ap"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "test_benign_fpr": row["test_benign_fpr"],
                "tp": row["tp"],
                "fp": row["fp"],
                "tn": row["tn"],
                "fn": row["fn"],
                "backdoor_recall": row["backdoor_recall"],
                "mitm_recall": row["mitm_recall"],
                "parameter_count": row["parameter_count"],
                "training_time_seconds": row["training_time_seconds"],
                "inference_time_seconds": row["inference_time_seconds"],
                "score_std_calibration": row["score_std_calibration"],
                "score_std_test": row["score_std_test"],
                "training_failed_or_degenerate": row["training_failed_or_degenerate"],
                "all_same_prediction": row["all_same_prediction"],
                "score_mean_benign": row["score_mean_benign"],
                "score_mean_attack": row["score_mean_attack"],
            }
        )
    summary_path = ROOT / "journal_rebuild" / "runs" / "metrics" / "ton_iot_candidate1_seed0_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    _write_report(data_manifest, smoke_results, formal_results, manifest_df)
    print(summary_path)


if __name__ == "__main__":
    main()
