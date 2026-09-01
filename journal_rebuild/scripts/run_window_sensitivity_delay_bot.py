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
from journal_rebuild.src.datasets.cicids2017 import (  # noqa: E402
    build_canonical_artifacts,
    load_data_manifest,
    load_split_arrays,
    load_window_manifest,
)
from journal_rebuild.src.evaluation.metrics import compute_auc_ap, metrics_at_threshold  # noqa: E402
from journal_rebuild.src.models.tcn_adversarial import count_parameters, save_checkpoint, train_model  # noqa: E402
from journal_rebuild.src.scoring.gan_scores import score_windows  # noqa: E402
from journal_rebuild.src.utils.config import load_yaml_like, merge_dicts  # noqa: E402
from journal_rebuild.src.utils.repro import ensure_omp_ok, git_commit, select_device, set_seed, utc_timestamp  # noqa: E402


TARGET_WINDOWS = [64, 256]
BASELINE_WINDOW = 128
SEED = 1
MODEL_NAME = "tcn_wgan_gp"
RUN_PREFIX = "window_sensitivity"


def role_window_frame(manifest_df: pd.DataFrame, role: str) -> pd.DataFrame:
    return manifest_df[manifest_df["split_role"] == role].reset_index(drop=True).copy()


def run_paths(run_id: str) -> dict[str, Path]:
    return {
        "checkpoint_dir": ROOT / "journal_rebuild" / "runs" / "checkpoints" / run_id,
        "scores_dir": ROOT / "journal_rebuild" / "runs" / "scores" / run_id,
        "logs_dir": ROOT / "journal_rebuild" / "runs" / "logs" / run_id,
        "metrics_dir": ROOT / "journal_rebuild" / "runs" / "metrics" / run_id,
    }


def write_training_log(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def score_export_frame(window_df: pd.DataFrame, scores: dict[str, np.ndarray], threshold: float) -> pd.DataFrame:
    out = window_df.loc[
        :,
        [
            "window_id",
            "split_role",
            "source_file",
            "source_day",
            "window_index_in_split",
            "start_row",
            "end_row",
            "split_start_offset",
            "split_end_offset",
            "label",
            "attack_family",
            "benign_flag",
            "attack_ratio",
        ],
    ].copy()
    for key, values in scores.items():
        out[key] = np.asarray(values, dtype=np.float32)
    out["threshold"] = float(threshold)
    out["prediction"] = (out["fused_score"].to_numpy(dtype=np.float32) >= float(threshold)).astype(np.uint8)
    out["threshold_margin"] = out["fused_score"] - float(threshold)
    return out


def attack_family_recall(window_df: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict[str, float]:
    out: dict[str, float] = {}
    attack_rows = window_df[window_df["label"] == 1]
    for family, sub in attack_rows.groupby("attack_family"):
        preds = (scores[sub.index.to_numpy(dtype=np.int64)] >= float(threshold)).astype(np.uint8)
        out[str(family)] = float(preds.mean()) if len(preds) else float("nan")
    return out


def build_window_config(window_size: int, target_fpr: float) -> dict[str, Any]:
    base = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "data" / "cicids2017.yaml")
    cfg = dict(base)
    cfg["window_size"] = int(window_size)
    cfg["stride"] = 16
    cfg["target_fpr"] = float(target_fpr)
    cfg["manifest_path"] = f"journal_rebuild/data/window_sensitivity/w{window_size}/canonical_split_manifest.csv"
    cfg["data_manifest_path"] = f"journal_rebuild/data/window_sensitivity/w{window_size}/data_manifest.json"
    cfg["scaler_path"] = f"journal_rebuild/data/window_sensitivity/w{window_size}/scaler.pkl"
    cfg["processed_dir"] = f"journal_rebuild/data/window_sensitivity/w{window_size}/processed"
    return cfg


def run_training_for_window(window_size: int, device: torch.device, target_fpr: float) -> dict[str, Any]:
    data_config = build_window_config(window_size, target_fpr)
    (ROOT / str(data_config["manifest_path"])).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / str(data_config["data_manifest_path"])).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / str(data_config["scaler_path"])).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / str(data_config["processed_dir"])).mkdir(parents=True, exist_ok=True)
    data_manifest = build_canonical_artifacts(ROOT, data_config)
    manifest_df = load_window_manifest(ROOT, data_manifest)
    train_arrays = load_split_arrays(ROOT, data_manifest, "model_train_benign")
    calib_arrays = load_split_arrays(ROOT, data_manifest, "independent_calibration_benign")
    test_arrays = load_split_arrays(ROOT, data_manifest, "test")
    feature_names = json.loads((ROOT / str(data_manifest["feature_path"])).read_text(encoding="utf-8"))
    train_window_df = role_window_frame(manifest_df, "model_train_benign")
    calib_window_df = role_window_frame(manifest_df, "independent_calibration_benign")
    test_window_df = role_window_frame(manifest_df, "test")

    model_config = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "models" / "tcn_wgan_gp.yaml")
    set_seed(SEED)
    training = train_model(
        model_config,
        feature_dim=int(train_arrays.features.shape[1]),
        window_size=int(window_size),
        train_features=train_arrays.features,
        train_starts=train_window_df["split_start_offset"].to_numpy(dtype=np.int64),
        train_labels=train_window_df["label"].to_numpy(dtype=np.uint8),
        device=device,
        epochs=int(model_config["epochs"]),
    )
    infer_t0 = time.perf_counter()
    calib_scores = score_windows(
        training.discriminator,
        features=calib_arrays.features,
        starts=calib_window_df["split_start_offset"].to_numpy(dtype=np.int64),
        ref_features=train_arrays.features,
        ref_starts=train_window_df["split_start_offset"].to_numpy(dtype=np.int64),
        window_size=int(window_size),
        batch_size=int(model_config["eval_batch_size"]),
        alpha=float(model_config["alpha"]),
        device=device,
    )
    test_scores = score_windows(
        training.discriminator,
        features=test_arrays.features,
        starts=test_window_df["split_start_offset"].to_numpy(dtype=np.int64),
        ref_features=train_arrays.features,
        ref_starts=train_window_df["split_start_offset"].to_numpy(dtype=np.int64),
        window_size=int(window_size),
        batch_size=int(model_config["eval_batch_size"]),
        alpha=float(model_config["alpha"]),
        device=device,
    )
    inference_time = float(time.perf_counter() - infer_t0)

    threshold = threshold_from_benign_fpr(calib_scores["fused_score"], float(target_fpr))
    auc, ap = compute_auc_ap(test_window_df["label"].to_numpy(dtype=np.uint8), test_scores["fused_score"])
    cls = metrics_at_threshold(test_window_df["label"].to_numpy(dtype=np.uint8), test_scores["fused_score"], threshold)
    fam_recall = attack_family_recall(test_window_df, test_scores["fused_score"], threshold)

    run_id = f"{RUN_PREFIX}_w{window_size}_{MODEL_NAME}_seed{SEED}"
    paths = run_paths(run_id)
    for folder in paths.values():
        folder.mkdir(parents=True, exist_ok=True)

    calib_export = score_export_frame(calib_window_df, calib_scores, threshold)
    test_export = score_export_frame(test_window_df, test_scores, threshold)
    calib_export.to_csv(paths["scores_dir"] / "scores_calibration.csv", index=False)
    test_export.to_csv(paths["scores_dir"] / "scores_test.csv", index=False)
    write_training_log(paths["logs_dir"] / "training_log.csv", training.training_log)

    checkpoint_path = paths["checkpoint_dir"] / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model_name=MODEL_NAME,
        loss_type=str(model_config["loss_type"]),
        seed=int(SEED),
        epoch=int(model_config["epochs"]),
        model_config=model_config,
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

    metrics = {
        "run_id": run_id,
        "model_name": MODEL_NAME,
        "seed": int(SEED),
        "window_size": int(window_size),
        "stride": int(data_config["stride"]),
        "target_fpr": float(target_fpr),
        "threshold": float(threshold),
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
        "train_windows": int(len(train_window_df)),
        "calibration_windows": int(len(calib_window_df)),
        "test_windows": int(len(test_window_df)),
        "parameter_count": int(count_parameters(training.generator) + count_parameters(training.discriminator)),
        "training_time_seconds": float(training.train_seconds),
        "inference_time_seconds": float(inference_time),
        "family_recall": fam_recall,
        "manifest_hash": str(data_manifest["manifest_hash"]),
        "scaler_hash": str(data_manifest["scaler_hash"]),
        "scores_test_csv": str((paths["scores_dir"] / "scores_test.csv").relative_to(ROOT)),
        "scores_calibration_csv": str((paths["scores_dir"] / "scores_calibration.csv").relative_to(ROOT)),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }
    (paths["metrics_dir"] / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def load_existing_w128_metrics() -> tuple[dict[str, Any], pd.DataFrame]:
    metrics_path = ROOT / "journal_rebuild" / "runs" / "metrics" / "formal_tcn_wgan_gp_seed1" / "metrics.json"
    scores_path = ROOT / "journal_rebuild" / "runs" / "scores" / "formal_tcn_wgan_gp_seed1" / "scores_test.csv"
    if not metrics_path.exists() or not scores_path.exists():
        raise FileNotFoundError("Missing existing formal_tcn_wgan_gp_seed1 artifacts for W=128 baseline")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    scores_df = pd.read_csv(scores_path)
    return metrics, scores_df


def detect_attack_segments(scores_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    test_df = scores_df.copy()
    if "prediction" not in test_df.columns:
        test_df["prediction"] = (test_df["fused_score"].to_numpy(dtype=np.float32) >= float(threshold)).astype(np.uint8)
    order_cols = ["source_file"]
    if "window_index_in_split" in test_df.columns:
        order_cols.append("window_index_in_split")
    elif "start_row" in test_df.columns:
        order_cols.append("start_row")
    else:
        order_cols.append("window_id")
    test_df = test_df.sort_values(order_cols).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    seg_id = 0
    for source_file, sub in test_df.groupby("source_file", sort=False):
        sub = sub.reset_index(drop=True)
        in_seg = False
        start = 0
        current_family = None
        for idx, row in sub.iterrows():
            label = int(row["label"])
            family = str(row["attack_family"])
            if label == 1 and not in_seg:
                in_seg = True
                start = idx
                current_family = family
            elif label == 1 and in_seg:
                continue
            elif label == 0 and in_seg:
                seg = sub.iloc[start:idx].copy()
                detected_idx = seg.index[seg["prediction"].astype(int) == 1]
                first_detect = int(detected_idx[0] - start) if len(detected_idx) else math.nan
                rows.append(
                    {
                        "segment_id": seg_id,
                        "source_file": source_file,
                        "attack_family": current_family,
                        "start_window_id": str(seg.iloc[0]["window_id"]),
                        "end_window_id": str(seg.iloc[-1]["window_id"]),
                        "segment_length_windows": int(len(seg)),
                        "detected": int(len(detected_idx) > 0),
                        "delay_windows": first_detect,
                        "segment_max_score": float(seg["fused_score"].max()),
                        "segment_mean_score": float(seg["fused_score"].mean()),
                    }
                )
                seg_id += 1
                in_seg = False
                current_family = None
        if in_seg:
            seg = sub.iloc[start:].copy()
            detected_idx = seg.index[seg["prediction"].astype(int) == 1]
            first_detect = int(detected_idx[0] - start) if len(detected_idx) else math.nan
            rows.append(
                {
                    "segment_id": seg_id,
                    "source_file": source_file,
                    "attack_family": current_family,
                    "start_window_id": str(seg.iloc[0]["window_id"]),
                    "end_window_id": str(seg.iloc[-1]["window_id"]),
                    "segment_length_windows": int(len(seg)),
                    "detected": int(len(detected_idx) > 0),
                    "delay_windows": first_detect,
                    "segment_max_score": float(seg["fused_score"].max()),
                    "segment_mean_score": float(seg["fused_score"].mean()),
                }
            )
            seg_id += 1
    return pd.DataFrame(rows)


def summarize_delay(window_size: int, scores_df: pd.DataFrame, threshold: float) -> tuple[dict[str, Any], pd.DataFrame]:
    seg_df = detect_attack_segments(scores_df, threshold)
    detected = seg_df[seg_df["detected"] == 1]
    summary = {
        "window_size": int(window_size),
        "attack_segments": int(len(seg_df)),
        "detected_segments": int(len(detected)),
        "segment_detection_rate": float(len(detected) / len(seg_df)) if len(seg_df) else math.nan,
        "avg_delay_windows_detected_only": float(detected["delay_windows"].mean()) if len(detected) else math.nan,
        "median_delay_windows_detected_only": float(detected["delay_windows"].median()) if len(detected) else math.nan,
        "avg_delay_windows_all_segments_with_miss_as_nan": float(seg_df["delay_windows"].mean()) if len(seg_df) else math.nan,
    }
    return summary, seg_df


def bot_failure_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for run_id in [
        "pilot_tcn_wgan_gp_seed0",
        "formal_tcn_wgan_gp_seed1",
        "formal_tcn_wgan_gp_seed2",
        "formal_tcn_wgan_gp_seed3",
        "formal_tcn_wgan_gp_seed4",
    ]:
        metrics_path = ROOT / "journal_rebuild" / "runs" / "metrics" / run_id / "metrics.json"
        scores_path = ROOT / "journal_rebuild" / "runs" / "scores" / run_id / "scores_test.csv"
        if not metrics_path.exists() or not scores_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        df = pd.read_csv(scores_path)
        threshold = float(metrics["threshold"])
        bot = df[df["attack_family"].astype(str).str.lower() == "bot"].copy()
        benign = df[df["label"] == 0].copy()
        if bot.empty:
            continue
        rows.append(
            {
                "run_id": run_id,
                "seed": int(metrics["seed"]),
                "threshold": threshold,
                "bot_support": int(len(bot)),
                "bot_recall": float((bot["prediction"].astype(int) == 1).mean()),
                "bot_score_mean": float(bot["fused_score"].mean()),
                "bot_score_median": float(bot["fused_score"].median()),
                "bot_margin_mean": float((bot["fused_score"] - threshold).mean()),
                "benign_score_mean": float(benign["fused_score"].mean()),
                "benign_score_median": float(benign["fused_score"].median()),
            }
        )
        score_rows.extend(
            [
                {
                    "run_id": run_id,
                    "seed": int(metrics["seed"]),
                    "group": "bot",
                    "mean_score": float(bot["fused_score"].mean()),
                    "median_score": float(bot["fused_score"].median()),
                    "std_score": float(bot["fused_score"].std()),
                    "q05": float(bot["fused_score"].quantile(0.05)),
                    "q25": float(bot["fused_score"].quantile(0.25)),
                    "q75": float(bot["fused_score"].quantile(0.75)),
                    "q95": float(bot["fused_score"].quantile(0.95)),
                    "mean_margin": float((bot["fused_score"] - threshold).mean()),
                },
                {
                    "run_id": run_id,
                    "seed": int(metrics["seed"]),
                    "group": "benign",
                    "mean_score": float(benign["fused_score"].mean()),
                    "median_score": float(benign["fused_score"].median()),
                    "std_score": float(benign["fused_score"].std()),
                    "q05": float(benign["fused_score"].quantile(0.05)),
                    "q25": float(benign["fused_score"].quantile(0.25)),
                    "q75": float(benign["fused_score"].quantile(0.75)),
                    "q95": float(benign["fused_score"].quantile(0.95)),
                    "mean_margin": float((benign["fused_score"] - threshold).mean()),
                },
            ]
        )
    return pd.DataFrame(rows), pd.DataFrame(score_rows)


def main() -> None:
    ensure_omp_ok()
    parser = argparse.ArgumentParser(description="Run window-size sensitivity, delay, and Bot-boundary analyses")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    exp_config = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "experiments" / "pilot_seed0.yaml")
    target_fpr = float(exp_config["target_fpr"])
    device = select_device(str(args.device))

    out_dir = ROOT / "journal_rebuild" / "reports" / "window_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    delay_rows: list[dict[str, Any]] = []

    # Existing 128 baseline
    metrics_128, scores_128 = load_existing_w128_metrics()
    delay_summary_128, seg_df_128 = summarize_delay(BASELINE_WINDOW, scores_128, float(metrics_128["threshold"]))
    summary_rows.append(
        {
            "window_size": 128,
            "seed": int(metrics_128["seed"]),
            "source": "existing_formal_run",
            "auc": float(metrics_128["auc"]),
            "ap": float(metrics_128["ap"]),
            "precision": float(metrics_128["precision"]),
            "recall": float(metrics_128["recall"]),
            "f1": float(metrics_128["f1"]),
            "test_benign_fpr": float(metrics_128["test_benign_fpr"]),
            "threshold": float(metrics_128["threshold"]),
            "train_windows": int(metrics_128.get("train_windows", 66366)),
            "calibration_windows": int(metrics_128.get("calibration_windows", 16591)),
            "test_windows": int(metrics_128.get("test_windows", 43897)),
            "training_time_seconds": float(metrics_128["training_time_seconds"]),
            "inference_time_seconds": float(metrics_128["inference_time_seconds"]),
            **delay_summary_128,
        }
    )
    seg_df_128.assign(window_size=128).to_csv(out_dir / "delay_segments_w128.csv", index=False)
    delay_rows.append({"window_size": 128, **delay_summary_128})

    # New 64/256 runs
    for window_size in TARGET_WINDOWS:
        metrics = run_training_for_window(window_size, device=device, target_fpr=target_fpr)
        scores_path = ROOT / metrics["scores_test_csv"]
        scores_df = pd.read_csv(scores_path)
        delay_summary, seg_df = summarize_delay(window_size, scores_df, float(metrics["threshold"]))
        summary_rows.append({**metrics, "source": "new_window_run", **delay_summary})
        delay_rows.append({"window_size": window_size, **delay_summary})
        seg_df.assign(window_size=window_size).to_csv(out_dir / f"delay_segments_w{window_size}.csv", index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values("window_size").reset_index(drop=True)
    summary_df.to_csv(out_dir / "window_sensitivity_summary.csv", index=False)
    pd.DataFrame(delay_rows).sort_values("window_size").to_csv(out_dir / "delay_summary.csv", index=False)

    bot_df, bot_score_df = bot_failure_summary()
    bot_df.to_csv(out_dir / "bot_failure_summary.csv", index=False)
    bot_score_df.to_csv(out_dir / "bot_score_statistics.csv", index=False)

    lines = [
        "# Window Sensitivity, Detection Delay, and Bot Boundary",
        "",
        "## Window-size trade-off",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `W={int(row['window_size'])}`: AUC `{row['auc']:.4f}`, AP `{row['ap']:.4f}`, "
            f"F1 `{row['f1']:.4f}`, test FPR `{row['test_benign_fpr']:.4f}`, "
            f"avg detected delay `{row['avg_delay_windows_detected_only']:.2f}` windows, "
            f"segment detection rate `{row['segment_detection_rate']:.4f}`."
        )
    if not bot_df.empty:
        lines.extend(
            [
                "",
                "## Bot failure boundary",
                "",
                f"- Mean Bot recall across available WGAN-GP seeds: `{bot_df['bot_recall'].mean():.4f}`.",
                f"- Mean Bot score margin to threshold: `{bot_df['bot_margin_mean'].mean():.4f}`.",
                f"- Mean benign score: `{bot_df['benign_score_mean'].mean():.4f}`; mean Bot score: `{bot_df['bot_score_mean'].mean():.4f}`.",
            ]
        )
    (out_dir / "window_delay_bot_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(out_dir / "window_sensitivity_summary.csv")
    print(out_dir / "delay_summary.csv")
    print(out_dir / "bot_failure_summary.csv")


if __name__ == "__main__":
    main()
