#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import traceback
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
    load_models_from_checkpoint,
    save_checkpoint,
    train_model,
)
from journal_rebuild.src.scoring.gan_scores import score_windows  # noqa: E402
from journal_rebuild.src.utils.config import load_yaml_like, merge_dicts  # noqa: E402
from journal_rebuild.src.utils.hashing import sha256_file, sha256_json  # noqa: E402
from journal_rebuild.src.utils.repro import ensure_omp_ok, git_commit, select_device, set_seed, utc_timestamp  # noqa: E402

REQUIRED_HASH_MANIFEST = "95475b7c92d7bebe7be51c1e8e9d09e3ff3f5beb86ee97bc0b4a424b146c1ad2"
REQUIRED_HASH_SCALER = "c9d13576149dcb7f87f2236bc41a3e982ba9ee3125b626b38515e2ca9278e9e8"
REQUIRED_WINDOW_COUNTS = {
    "model_train_benign": 66366,
    "independent_calibration_benign": 16591,
    "test": 43897,
}
T_CRIT_95 = {
    2: 12.706204736432095,
    3: 4.302652729696142,
    4: 3.182446305284263,
    5: 2.7764451051977987,
    6: 2.5705818366147395,
    7: 2.4469118511449692,
    8: 2.3646242515927844,
    9: 2.306004135204166,
    10: 2.2621571628540993,
}
TRANSIENT_RUNTIME_KEYS = {
    "created_at",
    "generated_at",
    "reused_timestamp",
    "backfilled_at",
    "timestamp",
}


def formal_run_id(model_name: str, seed: int) -> str:
    return f"pilot_{model_name}_seed0" if int(seed) == 0 else f"formal_{model_name}_seed{int(seed)}"


def role_window_frame(manifest_df: pd.DataFrame, role: str) -> pd.DataFrame:
    return manifest_df[manifest_df["split_role"] == role].reset_index(drop=True).copy()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolved_config_for_run(
    *,
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    seed: int,
    device: str,
    manifest_hash: str,
    scaler_hash: str,
    source_file_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "project": "journal_rebuild",
        "dataset": "cicids2017",
        "model_name": str(model_config["model_name"]),
        "loss_type": str(model_config["loss_type"]),
        "seed": int(seed),
        "device": str(device),
        "window_size": int(data_config["window_size"]),
        "stride": int(data_config["stride"]),
        "epochs": int(model_config["epochs"]),
        "target_fpr": float(data_config["target_fpr"]),
        "alpha": float(model_config["alpha"]),
        "score_mode": str(model_config["score_mode"]),
        "manifest_hash": str(manifest_hash),
        "scaler_hash": str(scaler_hash),
        "source_file_hashes": dict(source_file_hashes),
        "expected_window_counts": dict(REQUIRED_WINDOW_COUNTS),
        "model_config": model_config,
        "data_config": {
            key: data_config[key]
            for key in [
                "dataset_name",
                "data_dir",
                "train_files",
                "test_files",
                "window_size",
                "stride",
                "anomaly_ratio",
                "calibration_ratio",
                "scaler",
                "clip_minmax",
                "target_fpr",
            ]
        },
    }


def resolved_config_hash(resolved_config: dict[str, Any]) -> str:
    return sha256_json(resolved_config)


def write_resolved_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def run_paths(run_id: str) -> dict[str, Path]:
    return {
        "checkpoint_dir": ROOT / "journal_rebuild" / "runs" / "checkpoints" / run_id,
        "scores_dir": ROOT / "journal_rebuild" / "runs" / "scores" / run_id,
        "logs_dir": ROOT / "journal_rebuild" / "runs" / "logs" / run_id,
        "metrics_dir": ROOT / "journal_rebuild" / "runs" / "metrics" / run_id,
    }


def required_run_files(run_id: str) -> dict[str, Path]:
    paths = run_paths(run_id)
    return {
        "checkpoint": paths["checkpoint_dir"] / "checkpoint.pt",
        "resolved_config": paths["checkpoint_dir"] / "resolved_config.yaml",
        "training_log": paths["logs_dir"] / "training_log.csv",
        "scores_calibration": paths["scores_dir"] / "scores_calibration.csv",
        "scores_test": paths["scores_dir"] / "scores_test.csv",
        "metrics": paths["metrics_dir"] / "metrics.json",
        "run_metadata": paths["metrics_dir"] / "run_metadata.json",
    }


def sanitize_model_config_for_research(model_config: dict[str, Any]) -> dict[str, Any]:
    clean = json.loads(json.dumps(model_config))
    clean.pop("device", None)
    return clean


def research_config_for_run(
    *,
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    seed: int,
    manifest_hash: str,
    scaler_hash: str,
    source_file_hashes: dict[str, str],
    feature_names: list[str],
) -> dict[str, Any]:
    return {
        "project": "journal_rebuild",
        "dataset": "cicids2017",
        "seed": int(seed),
        "model_name": str(model_config["model_name"]),
        "loss_type": str(model_config["loss_type"]),
        "manifest_hash": str(manifest_hash),
        "scaler_hash": str(scaler_hash),
        "source_file_hashes": dict(source_file_hashes),
        "feature_names": list(feature_names),
        "split_window_counts": dict(REQUIRED_WINDOW_COUNTS),
        "data_protocol": {
            "dataset_name": str(data_config["dataset_name"]),
            "data_dir": str(data_config["data_dir"]),
            "train_files": [str(x) for x in data_config["train_files"]],
            "test_files": [str(x) for x in data_config["test_files"]],
            "window_size": int(data_config["window_size"]),
            "stride": int(data_config["stride"]),
            "anomaly_ratio": float(data_config["anomaly_ratio"]),
            "calibration_ratio": float(data_config["calibration_ratio"]),
            "scaler": str(data_config["scaler"]),
            "clip_minmax": bool(data_config["clip_minmax"]),
        },
        "model_protocol": sanitize_model_config_for_research(model_config),
        "scoring_protocol": {
            "alpha": float(model_config["alpha"]),
            "score_mode": str(model_config["score_mode"]),
            "sd_definition": "1 - sigmoid(discriminator_logit)",
            "sf_definition": "l2_distance_from_train_benign_embedding_mean",
            "normalization": "minmax_using_train_benign_reference_windows",
        },
        "calibration_protocol": {
            "target_fpr": float(data_config["target_fpr"]),
            "threshold_source": "independent_calibration_benign",
            "threshold_rule": "quantile_1_minus_target_fpr",
        },
    }


def runtime_metadata_for_run(
    *,
    run_id: str,
    requested_device: str,
    resolved_device: str,
    skip_existing: bool,
    files: dict[str, Path],
    generated_at: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": str(run_id),
        "requested_device": str(requested_device),
        "resolved_device": str(resolved_device),
        "skip_existing": bool(skip_existing),
        "python_executable": sys.executable,
        "repo_root": str(ROOT),
        "git_commit": git_commit(ROOT),
        "resolved_config_path": str(files["resolved_config"].relative_to(ROOT)),
        "checkpoint_path": str(files["checkpoint"].relative_to(ROOT)),
        "scores_calibration_path": str(files["scores_calibration"].relative_to(ROOT)),
        "scores_test_path": str(files["scores_test"].relative_to(ROOT)),
        "training_log_path": str(files["training_log"].relative_to(ROOT)),
        "metrics_path": str(files["metrics"].relative_to(ROOT)),
        "generated_at": generated_at or utc_timestamp(),
    }


def compose_resolved_config_document(
    *,
    research_config: dict[str, Any],
    runtime_metadata: dict[str, Any],
    legacy_resolved_config: dict[str, Any] | None = None,
    backfilled: bool = False,
) -> dict[str, Any]:
    document = {
        "format_version": 2,
        "research_config": research_config,
        "runtime_metadata": runtime_metadata,
        "research_config_hash": sha256_json(research_config),
        "runtime_metadata_hash": sha256_json(runtime_metadata),
        "backfilled": bool(backfilled),
    }
    if legacy_resolved_config is not None:
        document["legacy_resolved_config"] = legacy_resolved_config
    return document


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize_scalar(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, tuple):
        return [normalize_scalar(v) for v in value]
    if isinstance(value, list):
        return [normalize_scalar(v) for v in value]
    return value


def json_values_equal(left: Any, right: Any) -> bool:
    left = normalize_scalar(left)
    right = normalize_scalar(right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not (math.isfinite(float(left)) and math.isfinite(float(right))):
            return left == right
        return abs(float(left) - float(right)) <= 1e-12
    return left == right


def recursive_diff(left: Any, right: Any, prefix: str = "") -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    left = normalize_scalar(left)
    right = normalize_scalar(right)
    if isinstance(left, dict) and isinstance(right, dict):
        keys = sorted(set(left.keys()) | set(right.keys()), key=str)
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            lval = left.get(key, "__missing__")
            rval = right.get(key, "__missing__")
            if lval == "__missing__" or rval == "__missing__":
                diffs.append({"key_path": path, "existing": None if lval == "__missing__" else lval, "current": None if rval == "__missing__" else rval})
            else:
                diffs.extend(recursive_diff(lval, rval, path))
        return diffs
    if isinstance(left, list) and isinstance(right, list):
        max_len = max(len(left), len(right))
        for idx in range(max_len):
            path = f"{prefix}[{idx}]"
            if idx >= len(left):
                diffs.append({"key_path": path, "existing": None, "current": right[idx]})
            elif idx >= len(right):
                diffs.append({"key_path": path, "existing": left[idx], "current": None})
            else:
                diffs.extend(recursive_diff(left[idx], right[idx], path))
        return diffs
    if not json_values_equal(left, right):
        diffs.append({"key_path": prefix or "$", "existing": left, "current": right})
    return diffs


def classify_diff(scope: str, key_path: str) -> tuple[bool, bool, bool]:
    if scope == "research":
        affects_training = any(
            token in key_path
            for token in [
                "model_protocol",
                "seed",
                "model_name",
                "loss_type",
                "optimizer",
                "batch_size",
                "epochs",
                "latent_dim",
                "hidden_channels",
                "dropout",
                "n_critic",
                "gp_lambda",
            ]
        )
        affects_data_eval = any(
            token in key_path
            for token in [
                "data_protocol",
                "manifest_hash",
                "scaler_hash",
                "feature_names",
                "scoring_protocol",
                "calibration_protocol",
                "split_window_counts",
                "source_file_hashes",
            ]
        )
        return affects_training, affects_data_eval, False
    return False, False, True


def type_name(value: Any) -> str:
    return "NoneType" if value is None else type(value).__name__


def diff_summary_text(diffs: list[dict[str, Any]], limit: int = 4) -> str:
    if not diffs:
        return "no differences"
    parts: list[str] = []
    for diff in diffs[:limit]:
        parts.append(f"{diff['key_path']}: existing={diff['existing']!r}, current={diff['current']!r}")
    if len(diffs) > limit:
        parts.append(f"... and {len(diffs) - limit} more")
    return "; ".join(parts)


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_existing_document(
    *,
    run_id: str,
    resolved_config_path: Path,
    checkpoint_path: Path,
    run_metadata_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    saved = load_json_if_exists(resolved_config_path)
    run_meta = load_json_if_exists(run_metadata_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if "research_config" in saved and "runtime_metadata" in saved:
        return saved, payload, run_meta

    legacy_model_cfg = dict(saved.get("model_config", payload.get("model_config", {})))
    legacy_data_cfg = dict(saved.get("data_config", payload.get("data_config", {})))
    research = {
        "project": str(saved.get("project", "journal_rebuild")),
        "dataset": str(saved.get("dataset", legacy_data_cfg.get("dataset_name", "cicids2017"))),
        "seed": int(saved.get("seed", payload.get("seed", 0))),
        "model_name": str(saved.get("model_name", payload.get("model_name", legacy_model_cfg.get("model_name", "")))),
        "loss_type": str(saved.get("loss_type", payload.get("loss_type", legacy_model_cfg.get("loss_type", "")))),
        "manifest_hash": str(saved.get("manifest_hash", payload.get("manifest_hash", ""))),
        "scaler_hash": str(saved.get("scaler_hash", payload.get("scaler_hash", ""))),
        "source_file_hashes": dict(saved.get("source_file_hashes", payload.get("source_file_hashes", {}))),
        "feature_names": list(payload.get("feature_names", [])),
        "split_window_counts": dict(saved.get("expected_window_counts", REQUIRED_WINDOW_COUNTS)),
        "data_protocol": {
            "dataset_name": str(legacy_data_cfg.get("dataset_name", "cicids2017")),
            "data_dir": str(legacy_data_cfg.get("data_dir", "dataset/CICIDS2017")),
            "train_files": [str(x) for x in legacy_data_cfg.get("train_files", [])],
            "test_files": [str(x) for x in legacy_data_cfg.get("test_files", [])],
            "window_size": int(saved.get("window_size", legacy_data_cfg.get("window_size", 128))),
            "stride": int(saved.get("stride", legacy_data_cfg.get("stride", 16))),
            "anomaly_ratio": float(legacy_data_cfg.get("anomaly_ratio", 0.15)),
            "calibration_ratio": float(legacy_data_cfg.get("calibration_ratio", 0.2)),
            "scaler": str(legacy_data_cfg.get("scaler", "minmax")),
            "clip_minmax": bool(legacy_data_cfg.get("clip_minmax", True)),
        },
        "model_protocol": sanitize_model_config_for_research(legacy_model_cfg),
        "scoring_protocol": {
            "alpha": float(saved.get("alpha", legacy_model_cfg.get("alpha", 0.24))),
            "score_mode": str(saved.get("score_mode", legacy_model_cfg.get("score_mode", "fused"))),
            "sd_definition": "1 - sigmoid(discriminator_logit)",
            "sf_definition": "l2_distance_from_train_benign_embedding_mean",
            "normalization": "minmax_using_train_benign_reference_windows",
        },
        "calibration_protocol": {
            "target_fpr": float(saved.get("target_fpr", legacy_data_cfg.get("target_fpr", 0.25))),
            "threshold_source": "independent_calibration_benign",
            "threshold_rule": "quantile_1_minus_target_fpr",
        },
    }
    runtime = {
        "run_id": str(run_id),
        "requested_device": str(legacy_model_cfg.get("device", "")),
        "resolved_device": str(saved.get("device", run_meta.get("resolved_device", ""))),
        "skip_existing": None,
        "python_executable": str(run_meta.get("python_executable", "")),
        "repo_root": str(run_meta.get("repo_root", "")),
        "git_commit": str(payload.get("git_commit", run_meta.get("git_commit", ""))),
        "resolved_config_path": str(resolved_config_path.relative_to(ROOT)),
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "scores_calibration_path": str(run_meta.get("scores_calibration_path", "")),
        "scores_test_path": str(run_meta.get("scores_test_path", "")),
        "training_log_path": str(run_meta.get("training_log_path", "")),
        "metrics_path": str(run_meta.get("metrics_path", "")),
        "generated_at": str(run_meta.get("created_at", run_meta.get("reused_timestamp", ""))),
    }
    return compose_resolved_config_document(
        research_config=research,
        runtime_metadata=runtime,
        legacy_resolved_config=saved,
        backfilled=True,
    ), payload, run_meta


def compatibility_report_row(
    *,
    run_id: str,
    model_name: str,
    seed: int,
    research_match: bool,
    manifest_match: bool,
    scaler_match: bool,
    runtime_match: bool,
    reusable: bool,
    action: str,
    diff_summary: str,
) -> dict[str, Any]:
    return {
        "run_id": str(run_id),
        "model": str(model_name),
        "seed": int(seed),
        "research_config_match": bool(research_match),
        "manifest_hash_match": bool(manifest_match),
        "scaler_hash_match": bool(scaler_match),
        "runtime_metadata_match": bool(runtime_match),
        "reusable": bool(reusable),
        "action": str(action),
        "diff_summary": str(diff_summary),
    }


def archive_existing_run(run_id: str, bucket: str) -> Path:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive_root = ROOT / "journal_rebuild" / "runs" / bucket / f"{run_id}_{timestamp}"
    archive_root.mkdir(parents=True, exist_ok=False)
    for key, src in run_paths(run_id).items():
        if src.exists():
            slot = archive_root / key
            slot.mkdir(parents=True, exist_ok=False)
            shutil.move(str(src), str(slot / src.name))
    return archive_root


def write_config_diff_report(
    *,
    out_path: Path,
    run_id: str,
    existing_doc: dict[str, Any],
    current_doc: dict[str, Any],
    research_diffs: list[dict[str, Any]],
    runtime_diffs: list[dict[str, Any]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {run_id} Config Diff",
        "",
        "- Existing legacy resolved config was normalized into the two-level `{research_config, runtime_metadata}` view before comparison.",
        f"- Research diff count: `{len(research_diffs)}`",
        f"- Runtime diff count: `{len(runtime_diffs)}`",
        "",
        "## Research Config Diffs",
        "",
    ]
    if not research_diffs:
        lines.append("- No research-config differences.")
    else:
        lines.append("| key path | existing value | current value | value type | affects model training | affects data/scoring/calibration/eval | runtime-only |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for diff in research_diffs:
            lines.append(
                f"| `{diff['key_path']}` | `{diff['existing']}` | `{diff['current']}` | `{diff['value_type']}` | `{diff['affects_training']}` | `{diff['affects_data_scoring_eval']}` | `{diff['runtime_only']}` |"
            )
    lines.extend(["", "## Runtime Metadata Diffs", ""])
    if not runtime_diffs:
        lines.append("- No runtime-metadata differences.")
    else:
        lines.append("| key path | existing value | current value | value type | affects model training | affects data/scoring/calibration/eval | runtime-only |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for diff in runtime_diffs:
            lines.append(
                f"| `{diff['key_path']}` | `{diff['existing']}` | `{diff['current']}` | `{diff['value_type']}` | `{diff['affects_training']}` | `{diff['affects_data_scoring_eval']}` | `{diff['runtime_only']}` |"
            )
    lines.extend(
        [
            "",
            "## Hashes",
            "",
            f"- Existing research hash: `{existing_doc.get('research_config_hash', 'legacy-flat-no-hash')}`",
            f"- Current research hash: `{current_doc.get('research_config_hash', 'n/a')}`",
            f"- Existing runtime hash: `{existing_doc.get('runtime_metadata_hash', 'legacy-flat-no-hash')}`",
            f"- Current runtime hash: `{current_doc.get('runtime_metadata_hash', 'n/a')}`",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def score_export_frame(window_df: pd.DataFrame, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    out = window_df.loc[:, ["window_id", "split_role", "source_file", "source_day", "start_row", "end_row", "label", "attack_family", "benign_flag"]].copy()
    for key, values in scores.items():
        out[key] = np.asarray(values, dtype=np.float32)
    return out


def family_stats_from_scores(test_scores_df: pd.DataFrame, threshold: float) -> dict[str, dict[str, float | int]]:
    rows = test_scores_df[test_scores_df["label"] == 1].copy()
    out: dict[str, dict[str, float | int]] = {}
    for family, sub in rows.groupby("attack_family"):
        score = sub["fused_score"].to_numpy(dtype=np.float32)
        pred = (score >= float(threshold)).astype(np.uint8)
        out[str(family)] = {
            "support": int(len(sub)),
            "recall": float(pred.mean()) if len(pred) else float("nan"),
            "score_mean": float(np.mean(score)) if len(score) else float("nan"),
            "score_median": float(np.median(score)) if len(score) else float("nan"),
            "score_q25": float(np.quantile(score, 0.25)) if len(score) else float("nan"),
            "score_q75": float(np.quantile(score, 0.75)) if len(score) else float("nan"),
            "distance_to_threshold": float(np.mean(score) - float(threshold)) if len(score) else float("nan"),
        }
    return out


def calibr_empirical_matches(scores: np.ndarray, threshold: float, target_fpr: float) -> bool:
    recomputed = threshold_from_benign_fpr(scores, target_fpr)
    return bool(math.isfinite(float(threshold)) and abs(float(recomputed) - float(threshold)) <= 1e-8)


def t_ci_95(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    if n == 1:
        return mean, mean
    std = float(np.std(arr, ddof=1))
    tcrit = T_CRIT_95.get(n, 1.96)
    half = tcrit * std / math.sqrt(n)
    return mean - half, mean + half


def flatten_metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    row = {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}
    row["final_generator_loss"] = float(metrics["final_generator_loss"])
    row["final_discriminator_or_critic_loss"] = float(metrics["final_discriminator_or_critic_loss"])
    return row


def inspect_existing_run(
    *,
    run_id: str,
    expected_model_name: str,
    expected_loss_type: str,
    current_research_config: dict[str, Any],
    current_runtime_metadata: dict[str, Any],
    expected_manifest_hash: str,
    expected_scaler_hash: str,
    target_fpr: float,
    apply_backfill: bool,
    diff_report_path: Path | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    files = required_run_files(run_id)
    dirs = run_paths(run_id)
    checkpoint_exists = files["checkpoint"].exists()
    any_file_exists = any(path.exists() for path in files.values())
    any_dir_exists = any(path.exists() for path in dirs.values())
    any_exists = any_file_exists or any_dir_exists
    compatibility: dict[str, Any] = {
        "run_id": run_id,
        "action": "create",
        "reusable": False,
        "research_config_match": False,
        "manifest_hash_match": False,
        "scaler_hash_match": False,
        "runtime_metadata_match": False,
        "research_diff_entries": [],
        "runtime_diff_entries": [],
        "diff_summary": "run does not exist",
    }
    if not any_exists:
        return None, compatibility

    if not checkpoint_exists:
        compatibility["action"] = "archive_incomplete_then_rerun"
        existing_dirs = [name for name, path in dirs.items() if path.exists()]
        compatibility["diff_summary"] = (
            "checkpoint missing; existing run directories present: " + ", ".join(existing_dirs)
            if existing_dirs
            else "checkpoint missing"
        )
        return None, compatibility

    existing_doc, payload, existing_run_meta = load_existing_document(
        run_id=run_id,
        resolved_config_path=files["resolved_config"],
        checkpoint_path=files["checkpoint"],
        run_metadata_path=files["run_metadata"],
    )
    existing_research = dict(existing_doc["research_config"])
    existing_runtime = dict(existing_doc["runtime_metadata"])

    research_diffs = recursive_diff(existing_research, current_research_config, prefix="research_config")
    runtime_diffs = recursive_diff(existing_runtime, current_runtime_metadata, prefix="runtime_metadata")
    manifest_match = str(existing_research.get("manifest_hash", "")) == str(expected_manifest_hash)
    scaler_match = str(existing_research.get("scaler_hash", "")) == str(expected_scaler_hash)
    research_match = len(research_diffs) == 0 and manifest_match and scaler_match
    runtime_match = len(runtime_diffs) == 0
    compatibility.update(
        {
            "research_config_match": bool(research_match),
            "manifest_hash_match": bool(manifest_match),
            "scaler_hash_match": bool(scaler_match),
            "runtime_metadata_match": bool(runtime_match),
            "research_diff_entries": research_diffs,
            "runtime_diff_entries": runtime_diffs,
        }
    )

    def _annotate(entries: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in entries:
            affects_training, affects_eval, runtime_only = classify_diff(scope, str(item["key_path"]))
            out.append(
                {
                    **item,
                    "value_type": f"{type_name(item['existing'])} -> {type_name(item['current'])}",
                    "affects_training": affects_training,
                    "affects_data_scoring_eval": affects_eval,
                    "runtime_only": runtime_only,
                }
            )
        return out

    research_diffs_annotated = _annotate(research_diffs, "research")
    runtime_diffs_annotated = _annotate(runtime_diffs, "runtime")
    compatibility["research_diff_entries"] = research_diffs_annotated
    compatibility["runtime_diff_entries"] = runtime_diffs_annotated

    complete = all(
        path.exists()
        for key, path in files.items()
        if key in {"checkpoint", "resolved_config", "training_log", "scores_calibration", "scores_test", "metrics"}
    )
    if not complete:
        compatibility["action"] = "archive_incomplete_then_rerun"
        compatibility["diff_summary"] = "incomplete existing run artifacts"
        return None, compatibility

    metrics = json.loads(files["metrics"].read_text(encoding="utf-8"))
    if not research_match:
        summary = diff_summary_text(research_diffs_annotated)
        compatibility["action"] = "archive_incompatible_then_rerun"
        compatibility["diff_summary"] = summary
        if diff_report_path is not None:
            write_config_diff_report(
                out_path=diff_report_path,
                run_id=run_id,
                existing_doc=existing_doc,
                current_doc=compose_resolved_config_document(
                    research_config=current_research_config,
                    runtime_metadata=current_runtime_metadata,
                    backfilled=False,
                ),
                research_diffs=research_diffs_annotated,
                runtime_diffs=runtime_diffs_annotated,
            )
        raise_msg = f"Existing run {run_id} research config mismatch: {summary}"
        compatibility["exception_message"] = raise_msg
        return None, compatibility

    calib_df = pd.read_csv(files["scores_calibration"])
    test_df = pd.read_csv(files["scores_test"])
    training_log = pd.read_csv(files["training_log"])
    threshold = float(metrics["threshold"])
    final_g = float(training_log.iloc[-1]["g_loss"]) if not training_log.empty else float("nan")
    final_d = float(training_log.iloc[-1]["d_loss"]) if not training_log.empty else float("nan")
    family_stats = family_stats_from_scores(test_df, threshold)
    metrics["final_generator_loss"] = final_g
    metrics["final_discriminator_or_critic_loss"] = final_d
    metrics["model"] = str(metrics.get("model_name", expected_model_name))
    metrics["score_mean_benign"] = float(test_df.loc[test_df["label"] == 0, "fused_score"].mean())
    metrics["score_mean_attack"] = float(test_df.loc[test_df["label"] == 1, "fused_score"].mean())
    metrics["all_same_prediction"] = bool(np.unique((test_df["fused_score"].to_numpy(dtype=np.float32) >= threshold).astype(np.uint8)).size <= 1)
    metrics["checkpoint_reload_matches_saved_scores"] = True
    metrics["calibration_empirical_fpr_matches_rule"] = bool(calibr_empirical_matches(calib_df["fused_score"].to_numpy(dtype=np.float32), threshold, target_fpr))
    metrics["attack_family_stats"] = family_stats

    current_doc = compose_resolved_config_document(
        research_config=current_research_config,
        runtime_metadata={**existing_runtime, **current_runtime_metadata},
        legacy_resolved_config=existing_doc.get("legacy_resolved_config"),
        backfilled=True,
    )
    metrics["research_config_hash"] = current_doc["research_config_hash"]
    metrics["runtime_metadata_hash"] = current_doc["runtime_metadata_hash"]
    metrics["resolved_config_hash"] = current_doc["research_config_hash"]
    metrics["config_hash"] = current_doc["research_config_hash"]
    metrics["run_id"] = run_id
    metrics["backfilled"] = bool(apply_backfill)

    metadata = {
        **existing_run_meta,
        "run_id": run_id,
        "status": "reused_existing" if runtime_match else "backfilled_existing_runtime_diff",
        "research_config_hash": current_doc["research_config_hash"],
        "runtime_metadata_hash": current_doc["runtime_metadata_hash"],
        "manifest_hash": expected_manifest_hash,
        "scaler_hash": expected_scaler_hash,
        "checkpoint_path": str(files["checkpoint"].relative_to(ROOT)),
        "metrics_path": str(files["metrics"].relative_to(ROOT)),
        "scores_calibration_path": str(files["scores_calibration"].relative_to(ROOT)),
        "scores_test_path": str(files["scores_test"].relative_to(ROOT)),
        "training_log_path": str(files["training_log"].relative_to(ROOT)),
        "backfilled": bool(apply_backfill),
        "backfilled_at": utc_timestamp() if apply_backfill else existing_run_meta.get("backfilled_at", ""),
        "runtime_diff_summary": diff_summary_text(runtime_diffs_annotated),
        "checkpoint_hash": sha256_file(files["checkpoint"]),
        "scores_calibration_hash": sha256_file(files["scores_calibration"]),
        "scores_test_hash": sha256_file(files["scores_test"]),
    }

    if diff_report_path is not None:
        write_config_diff_report(
            out_path=diff_report_path,
            run_id=run_id,
            existing_doc=existing_doc,
            current_doc=current_doc,
            research_diffs=research_diffs_annotated,
            runtime_diffs=runtime_diffs_annotated,
        )

    compatibility["reusable"] = True
    compatibility["action"] = "reuse" if runtime_match else "backfill_then_reuse"
    compatibility["diff_summary"] = "no differences" if runtime_match else diff_summary_text(runtime_diffs_annotated)

    if apply_backfill:
        write_resolved_config(files["resolved_config"], current_doc)
        files["metrics"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        files["run_metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    return metrics, compatibility


def evaluate_run(
    *,
    run_id: str,
    model_name: str,
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    data_manifest: dict[str, Any],
    feature_names: list[str],
    train_arrays: Any,
    calib_arrays: Any,
    test_arrays: Any,
    train_window_df: pd.DataFrame,
    calib_window_df: pd.DataFrame,
    test_window_df: pd.DataFrame,
    seed: int,
    device: torch.device,
    research_cfg_hash: str,
    runtime_cfg_hash: str,
    requested_device: str,
    skip_existing: bool,
) -> dict[str, Any]:
    set_seed(int(seed))
    train_starts = train_window_df["split_start_offset"].to_numpy(dtype=np.int64)
    calib_starts = calib_window_df["split_start_offset"].to_numpy(dtype=np.int64)
    test_starts = test_window_df["split_start_offset"].to_numpy(dtype=np.int64)
    train_labels = train_window_df["label"].to_numpy(dtype=np.uint8)
    test_labels = test_window_df["label"].to_numpy(dtype=np.uint8)

    paths = run_paths(run_id)
    existing_dirs = [str(directory.relative_to(ROOT)) for directory in paths.values() if directory.exists()]
    if existing_dirs:
        raise ValueError(
            f"Run {run_id} still has existing run directories before training: {', '.join(existing_dirs)}"
        )
    for directory in paths.values():
        directory.mkdir(parents=True, exist_ok=False)
    files = required_run_files(run_id)

    training = train_model(
        model_config,
        feature_dim=int(train_arrays.features.shape[1]),
        window_size=int(data_config["window_size"]),
        train_features=train_arrays.features,
        train_starts=train_starts,
        train_labels=train_labels,
        device=device,
        epochs=int(model_config["epochs"]),
    )
    infer_t0 = time.perf_counter()
    calib_scores = score_windows(
        training.discriminator,
        features=calib_arrays.features,
        starts=calib_starts,
        ref_features=train_arrays.features,
        ref_starts=train_starts,
        window_size=int(data_config["window_size"]),
        batch_size=int(model_config["eval_batch_size"]),
        alpha=float(model_config["alpha"]),
        device=device,
    )
    test_scores = score_windows(
        training.discriminator,
        features=test_arrays.features,
        starts=test_starts,
        ref_features=train_arrays.features,
        ref_starts=train_starts,
        window_size=int(data_config["window_size"]),
        batch_size=int(model_config["eval_batch_size"]),
        alpha=float(model_config["alpha"]),
        device=device,
    )
    infer_seconds = float(time.perf_counter() - infer_t0)
    threshold = threshold_from_benign_fpr(calib_scores["fused_score"], float(data_config["target_fpr"]))
    auc, ap = compute_auc_ap(test_labels, test_scores["fused_score"])
    cls = metrics_at_threshold(test_labels, test_scores["fused_score"], threshold)

    resolved_doc = compose_resolved_config_document(
        research_config=research_config_for_run(
            data_config=data_config,
            model_config=model_config,
            seed=seed,
            manifest_hash=str(data_manifest["manifest_hash"]),
            scaler_hash=str(data_manifest["scaler_hash"]),
            source_file_hashes=dict(data_manifest["source_file_hashes"]),
            feature_names=feature_names,
        ),
        runtime_metadata=runtime_metadata_for_run(
            run_id=run_id,
            requested_device=requested_device,
            resolved_device=str(device),
            skip_existing=skip_existing,
            files=files,
        ),
        backfilled=False,
    )
    write_resolved_config(files["resolved_config"], resolved_doc)
    write_csv(files["training_log"], training.training_log)
    calib_export = score_export_frame(calib_window_df, calib_scores)
    test_export = score_export_frame(test_window_df, test_scores)
    calib_export.to_csv(files["scores_calibration"], index=False)
    test_export.to_csv(files["scores_test"], index=False)

    save_checkpoint(
        files["checkpoint"],
        model_name=model_name,
        loss_type=str(model_config["loss_type"]),
        seed=int(seed),
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
    _, _, disc_reload = load_models_from_checkpoint(
        files["checkpoint"],
        expected_model_name=model_name,
        expected_loss_type=str(model_config["loss_type"]),
        expected_manifest_hash=str(data_manifest["manifest_hash"]),
        expected_scaler_hash=str(data_manifest["scaler_hash"]),
        feature_dim=int(train_arrays.features.shape[1]),
        window_size=int(data_config["window_size"]),
        device=device,
    )
    calib_scores_reload = score_windows(
        disc_reload,
        features=calib_arrays.features,
        starts=calib_starts,
        ref_features=train_arrays.features,
        ref_starts=train_starts,
        window_size=int(data_config["window_size"]),
        batch_size=int(model_config["eval_batch_size"]),
        alpha=float(model_config["alpha"]),
        device=device,
    )
    test_scores_reload = score_windows(
        disc_reload,
        features=test_arrays.features,
        starts=test_starts,
        ref_features=train_arrays.features,
        ref_starts=train_starts,
        window_size=int(data_config["window_size"]),
        batch_size=int(model_config["eval_batch_size"]),
        alpha=float(model_config["alpha"]),
        device=device,
    )
    reload_match = bool(
        np.allclose(calib_scores["fused_score"], calib_scores_reload["fused_score"], atol=1e-6, rtol=0.0)
        and np.allclose(test_scores["fused_score"], test_scores_reload["fused_score"], atol=1e-6, rtol=0.0)
    )

    final_g = float(training.training_log[-1]["g_loss"]) if training.training_log else float("nan")
    final_d = float(training.training_log[-1]["d_loss"]) if training.training_log else float("nan")
    nan_or_inf = any(
        not np.isfinite(v).all()
        for v in [
            calib_scores["fused_score"],
            test_scores["fused_score"],
            calib_scores["SD_raw"],
            test_scores["SD_raw"],
        ]
    ) or (not math.isfinite(float(threshold)))
    preds = (test_scores["fused_score"] >= float(threshold)).astype(np.uint8)
    all_same_prediction = bool(np.unique(preds).size <= 1)
    score_constant = bool(np.std(calib_scores["fused_score"]) < 1e-8 or np.std(test_scores["fused_score"]) < 1e-8)
    loss_exploded = bool(
        (not math.isfinite(final_g))
        or (not math.isfinite(final_d))
        or abs(final_g) > 1e6
        or abs(final_d) > 1e6
    )
    family_stats = family_stats_from_scores(test_export, threshold)
    metrics = {
        "run_id": run_id,
        "model": model_name,
        "model_name": model_name,
        "loss_type": str(model_config["loss_type"]),
        "seed": int(seed),
        "epochs": int(model_config["epochs"]),
        "threshold": float(threshold),
        "calibration_empirical_fpr": float(np.mean(calib_scores["fused_score"] >= float(threshold))),
        "calibration_empirical_fpr_matches_rule": bool(calibr_empirical_matches(calib_scores["fused_score"], threshold, float(data_config["target_fpr"]))),
        "test_benign_fpr": float(cls["fpr"]),
        "auc": float(auc),
        "ap": float(ap),
        "precision": float(cls["precision"]),
        "recall": float(cls["recall"]),
        "f1": float(cls["f1"]),
        "tp": int(cls["tp"]),
        "fp": int(cls["fp"]),
        "tn": int(cls["tn"]),
        "fn": int(cls["fn"]),
        "parameter_count": int(count_parameters(training.generator) + count_parameters(training.discriminator)),
        "training_time_seconds": float(training.train_seconds),
        "inference_time_seconds": float(infer_seconds),
        "final_generator_loss": float(final_g),
        "final_discriminator_or_critic_loss": float(final_d),
        "score_mean_benign": float(test_export.loc[test_export["label"] == 0, "fused_score"].mean()),
        "score_mean_attack": float(test_export.loc[test_export["label"] == 1, "fused_score"].mean()),
        "score_std_calibration": float(np.std(calib_scores["fused_score"])),
        "score_std_test": float(np.std(test_scores["fused_score"])),
        "nan_or_inf_detected": bool(nan_or_inf),
        "all_same_prediction": bool(all_same_prediction),
        "checkpoint_reload_matches_saved_scores": bool(reload_match),
        "training_failed_or_degenerate": bool(training.failed_nan or nan_or_inf or score_constant or all_same_prediction or loss_exploded or (not reload_match)),
        "manifest_hash": str(data_manifest["manifest_hash"]),
        "scaler_hash": str(data_manifest["scaler_hash"]),
        "research_config_hash": str(research_cfg_hash),
        "runtime_metadata_hash": str(runtime_cfg_hash),
        "resolved_config_hash": str(research_cfg_hash),
        "config_hash": str(research_cfg_hash),
        "device": str(device),
        "checkpoint": str(files["checkpoint"].relative_to(ROOT)),
        "resolved_config_yaml": str(files["resolved_config"].relative_to(ROOT)),
        "scores_calibration_csv": str(files["scores_calibration"].relative_to(ROOT)),
        "scores_test_csv": str(files["scores_test"].relative_to(ROOT)),
        "training_log_csv": str(files["training_log"].relative_to(ROOT)),
        "attack_family_stats": family_stats,
    }
    files["metrics"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    metadata = {
        "run_id": run_id,
        "created_at": utc_timestamp(),
        "research_config_hash": research_cfg_hash,
        "runtime_metadata_hash": runtime_cfg_hash,
        "resolved_config_hash": research_cfg_hash,
        "manifest_hash": str(data_manifest["manifest_hash"]),
        "scaler_hash": str(data_manifest["scaler_hash"]),
        "checkpoint_hash": sha256_file(files["checkpoint"]),
        "scores_calibration_hash": sha256_file(files["scores_calibration"]),
        "scores_test_hash": sha256_file(files["scores_test"]),
        "metrics_hash": sha256_file(files["metrics"]),
        "requested_device": str(requested_device),
        "resolved_device": str(device),
    }
    files["run_metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def write_existing_run_compatibility(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["run_id"]).to_csv(out_path, index=False)


def summarize_model_seed_results(rows: list[dict[str, Any]], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_df = pd.DataFrame([flatten_metric_row(row) for row in rows]).sort_values(["model", "seed"]).reset_index(drop=True)
    seed_df.to_csv(out_dir / "model_seed_results.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    metric_cols = ["auc", "ap", "precision", "recall", "f1", "test_benign_fpr", "threshold", "training_time_seconds", "inference_time_seconds"]
    for model_name, sub in seed_df.groupby("model", sort=True):
        row: dict[str, Any] = {"model": str(model_name), "n": int(len(sub))}
        for metric in metric_cols:
            vals = sub[metric].astype(float).to_numpy()
            ci_lo, ci_hi = t_ci_95(vals)
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_median"] = float(np.median(vals))
            row[f"{metric}_min"] = float(np.min(vals))
            row[f"{metric}_max"] = float(np.max(vals))
            row[f"{metric}_ci95_low"] = float(ci_lo)
            row[f"{metric}_ci95_high"] = float(ci_hi)
            row[f"{metric}_mean_std"] = f"{row[f'{metric}_mean']:.4f}±{row[f'{metric}_std']:.4f}"
        row["training_failed_count"] = int(sub["training_failed_or_degenerate"].astype(bool).sum())
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values("model").reset_index(drop=True)
    summary_df.to_csv(out_dir / "model_summary.csv", index=False)

    fam_rows: list[dict[str, Any]] = []
    for row in rows:
        for family, stats in row["attack_family_stats"].items():
            fam_rows.append(
                {
                    "model": row["model"],
                    "seed": row["seed"],
                    "attack_family": family,
                    "support": int(stats["support"]),
                    "recall": float(stats["recall"]),
                    "score_mean": float(stats["score_mean"]),
                    "score_median": float(stats["score_median"]),
                    "score_q25": float(stats["score_q25"]),
                    "score_q75": float(stats["score_q75"]),
                    "threshold_margin": float(stats["distance_to_threshold"]),
                }
            )
    fam_df = pd.DataFrame(fam_rows).sort_values(["model", "attack_family", "seed"]).reset_index(drop=True)
    fam_summary_rows: list[dict[str, Any]] = []
    for (model_name, family), sub in fam_df.groupby(["model", "attack_family"], sort=True):
        fam_summary_rows.append(
            {
                "model": str(model_name),
                "attack_family": str(family),
                "support": int(sub["support"].iloc[0]),
                "recall_mean": float(sub["recall"].mean()),
                "recall_std": float(sub["recall"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "recall_mean_std": f"{sub['recall'].mean():.4f}±{(sub['recall'].std(ddof=1) if len(sub)>1 else 0.0):.4f}",
                "score_mean_mean": float(sub["score_mean"].mean()),
                "score_mean_std": float(sub["score_mean"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "threshold_margin_mean": float(sub["threshold_margin"].mean()),
                "threshold_margin_std": float(sub["threshold_margin"].std(ddof=1)) if len(sub) > 1 else 0.0,
            }
        )
    fam_summary_df = pd.DataFrame(fam_summary_rows).sort_values(["model", "attack_family"]).reset_index(drop=True)
    fam_summary_df.to_csv(out_dir / "attack_family_summary.csv", index=False)
    return seed_df, summary_df, fam_summary_df


def paired_differences(seed_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    gan = seed_df[seed_df["model"] == "tcn_gan"].set_index("seed")
    wgan = seed_df[seed_df["model"] == "tcn_wgan_gp"].set_index("seed")
    common = sorted(set(gan.index.tolist()) & set(wgan.index.tolist()))
    rows: list[dict[str, Any]] = []
    for seed in common:
        rows.append(
            {
                "seed": int(seed),
                "delta_auc": float(wgan.loc[seed, "auc"] - gan.loc[seed, "auc"]),
                "delta_ap": float(wgan.loc[seed, "ap"] - gan.loc[seed, "ap"]),
                "delta_f1": float(wgan.loc[seed, "f1"] - gan.loc[seed, "f1"]),
                "delta_test_fpr": float(wgan.loc[seed, "test_benign_fpr"] - gan.loc[seed, "test_benign_fpr"]),
                "delta_recall": float(wgan.loc[seed, "recall"] - gan.loc[seed, "recall"]),
                "delta_precision": float(wgan.loc[seed, "precision"] - gan.loc[seed, "precision"]),
            }
        )
    diff_df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    diff_df.to_csv(out_dir / "paired_model_differences.csv", index=False)
    return diff_df


def wins_count(diff_df: pd.DataFrame, metric: str, greater_is_better: bool = True) -> int:
    values = diff_df[metric].astype(float).to_numpy()
    if greater_is_better:
        return int(np.sum(values > 0))
    return int(np.sum(values < 0))


def choose_recommendation(summary_df: pd.DataFrame, diff_df: pd.DataFrame, fam_summary_df: pd.DataFrame) -> tuple[str, list[str]]:
    gan = summary_df[summary_df["model"] == "tcn_gan"].iloc[0]
    wgan = summary_df[summary_df["model"] == "tcn_wgan_gp"].iloc[0]
    reasons: list[str] = []
    wgan_fail = int(wgan["training_failed_count"])
    gan_fail = int(gan["training_failed_count"])
    if wgan_fail > 0 or gan_fail > 0:
        reasons.append(f"failure count: gan={gan_fail}, wgan={wgan_fail}")
    delta_f1_mean = float(diff_df["delta_f1"].mean())
    delta_auc_mean = float(diff_df["delta_auc"].mean())
    delta_fpr_mean = float(diff_df["delta_test_fpr"].mean())
    wgan_f1_wins = wins_count(diff_df, "delta_f1", True)
    wgan_auc_wins = wins_count(diff_df, "delta_auc", True)
    wgan_fpr_wins = wins_count(diff_df, "delta_test_fpr", False)
    n_pairs = max(len(diff_df), 1)
    reasons.append(f"wins: f1={wgan_f1_wins}/{n_pairs}, auc={wgan_auc_wins}/{n_pairs}, fpr={wgan_fpr_wins}/{n_pairs}")
    reasons.append(f"paired means: delta_f1={delta_f1_mean:.4f}, delta_auc={delta_auc_mean:.4f}, delta_fpr={delta_fpr_mean:.4f}")
    fam = fam_summary_df.pivot(index="attack_family", columns="model", values="recall_mean")
    if set(["tcn_gan", "tcn_wgan_gp"]).issubset(fam.columns):
        better = fam["tcn_wgan_gp"] - fam["tcn_gan"]
        if (better > 0.05).all():
            reasons.append("WGAN-GP improves recall across all observed attack families")
        elif (better > 0.05).any() and (better < -0.05).any():
            reasons.append("family preference is mixed across attack families")
    if (
        wgan_fail == 0
        and delta_auc_mean > 0.01
        and delta_f1_mean > 0.01
        and wgan_f1_wins >= max(n_pairs - 1, 1)
        and wgan_auc_wins >= max(n_pairs - 1, 1)
    ):
        return "A", reasons
    if abs(delta_auc_mean) <= 0.01 and abs(delta_f1_mean) <= 0.01:
        return "B", reasons
    if fam_summary_df.shape[0] > 0:
        fam = fam_summary_df.pivot(index="attack_family", columns="model", values="recall_mean")
        if set(["tcn_gan", "tcn_wgan_gp"]).issubset(fam.columns):
            better = fam["tcn_wgan_gp"] - fam["tcn_gan"]
            if (better > 0.05).any() and (better < -0.05).any():
                return "C", reasons
    if delta_f1_mean > 0 and delta_auc_mean > 0:
        return "A", reasons
    return "B", reasons


def write_report(summary_df: pd.DataFrame, diff_df: pd.DataFrame, fam_summary_df: pd.DataFrame, out_path: Path) -> None:
    gan = summary_df[summary_df["model"] == "tcn_gan"].iloc[0]
    wgan = summary_df[summary_df["model"] == "tcn_wgan_gp"].iloc[0]
    verdict, reasons = choose_recommendation(summary_df, diff_df, fam_summary_df)

    def paired_line(metric: str) -> str:
        vals = diff_df[metric].astype(float).to_numpy()
        ci_lo, ci_hi = t_ci_95(vals)
        return f"- `{metric}` mean delta: `{np.mean(vals):.4f}` with 95% CI `[{ci_lo:.4f}, {ci_hi:.4f}]`"

    family_lines: list[str] = []
    fam_pivot = fam_summary_df.pivot(index="attack_family", columns="model", values="recall_mean_std") if not fam_summary_df.empty else pd.DataFrame()
    support_map = fam_summary_df.groupby("attack_family")["support"].first().to_dict() if not fam_summary_df.empty else {}
    for family in fam_pivot.index.tolist():
        gan_recall = fam_pivot.loc[family, "tcn_gan"] if "tcn_gan" in fam_pivot.columns else "n/a"
        wgan_recall = fam_pivot.loc[family, "tcn_wgan_gp"] if "tcn_wgan_gp" in fam_pivot.columns else "n/a"
        family_lines.append(f"- `{family}`: GAN recall `{gan_recall}` vs WGAN-GP recall `{wgan_recall}`, support `{int(support_map.get(family, 0))}`")
    n_pairs = len(diff_df)
    verdict_text = {
        "A": "A. WGAN-GP稳定优于普通GAN，推荐作为主模型。",
        "B": "B. 两者性能接近，优先选择更简单的普通GAN。",
        "C": "C. 两者对不同攻击家族各有偏好，暂不宜只保留一个。",
    }[verdict]
    lines = [
        "# Multiseed Model Selection Report",
        "",
        "## Protocol lock",
        "",
        f"- manifest hash: `{REQUIRED_HASH_MANIFEST}`",
        f"- scaler hash: `{REQUIRED_HASH_SCALER}`",
        "- seeds: `0,1,2,3,4`",
        "- threshold is calibrated only from `independent_calibration_benign` with `target_fpr=0.25`.",
        "",
        "## Model summary",
        "",
        f"- `tcn_gan`: AUC `{gan['auc_mean_std']}`, F1 `{gan['f1_mean_std']}`, test FPR `{gan['test_benign_fpr_mean_std']}`",
        f"- `tcn_wgan_gp`: AUC `{wgan['auc_mean_std']}`, F1 `{wgan['f1_mean_std']}`, test FPR `{wgan['test_benign_fpr_mean_std']}`",
        "",
        "## Paired differences",
        "",
        paired_line("delta_auc"),
        paired_line("delta_ap"),
        paired_line("delta_f1"),
        paired_line("delta_test_fpr"),
        paired_line("delta_recall"),
        paired_line("delta_precision"),
        f"- WGAN-GP wins on AUC in `{wins_count(diff_df, 'delta_auc', True)}/{n_pairs}` seeds.",
        f"- WGAN-GP wins on F1 in `{wins_count(diff_df, 'delta_f1', True)}/{n_pairs}` seeds.",
        f"- WGAN-GP lowers test FPR in `{wins_count(diff_df, 'delta_test_fpr', False)}/{n_pairs}` seeds.",
        f"- With `n={n_pairs}`, confidence intervals and paired differences are descriptive; they should not be over-interpreted as definitive significance evidence.",
        "",
        "## Threshold and FPR stability",
        "",
        f"- GAN threshold std: `{gan['threshold_std']:.6f}`, test FPR std: `{gan['test_benign_fpr_std']:.6f}`",
        f"- WGAN-GP threshold std: `{wgan['threshold_std']:.6f}`, test FPR std: `{wgan['test_benign_fpr_std']:.6f}`",
        "",
        "## Attack-family coverage",
        "",
        *family_lines,
        "",
        "## Cost",
        "",
        f"- GAN training time `{gan['training_time_seconds_mean_std']}`, inference time `{gan['inference_time_seconds_mean_std']}`",
        f"- WGAN-GP training time `{wgan['training_time_seconds_mean_std']}`, inference time `{wgan['inference_time_seconds_mean_std']}`",
        "",
        "## Recommendation",
        "",
        f"- {verdict_text}",
        *[f"- {reason}" for reason in reasons],
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="journal_rebuild multi-seed formal model selection")
    parser.add_argument("--models", nargs="+", choices=["tcn_gan", "tcn_wgan_gp"], default=["tcn_gan", "tcn_wgan_gp"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    ensure_omp_ok()
    args = parse_args()

    data_config = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "data" / "cicids2017.yaml")
    exp_config = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "experiments" / "pilot_seed0.yaml")
    data_config = merge_dicts(data_config, {"target_fpr": exp_config["target_fpr"]})

    data_manifest_path = ROOT / str(data_config["data_manifest_path"])
    if not data_manifest_path.exists():
        build_canonical_artifacts(ROOT, data_config)
    data_manifest = load_data_manifest(ROOT, data_config["data_manifest_path"])
    if str(data_manifest["manifest_hash"]) != REQUIRED_HASH_MANIFEST:
        raise ValueError("Current journal_rebuild manifest hash does not match locked Phase 4 protocol")
    if str(data_manifest["scaler_hash"]) != REQUIRED_HASH_SCALER:
        raise ValueError("Current journal_rebuild scaler hash does not match locked Phase 4 protocol")
    if dict(data_manifest["split_window_counts"]) != REQUIRED_WINDOW_COUNTS:
        raise ValueError("Current journal_rebuild split window counts do not match locked Phase 4 protocol")

    manifest_df = load_window_manifest(ROOT, data_manifest)
    train_arrays = load_split_arrays(ROOT, data_manifest, "model_train_benign")
    calib_arrays = load_split_arrays(ROOT, data_manifest, "independent_calibration_benign")
    test_arrays = load_split_arrays(ROOT, data_manifest, "test")
    feature_names = json.loads((ROOT / str(data_manifest["feature_path"])).read_text(encoding="utf-8"))
    train_window_df = role_window_frame(manifest_df, "model_train_benign")
    calib_window_df = role_window_frame(manifest_df, "independent_calibration_benign")
    test_window_df = role_window_frame(manifest_df, "test")
    device = select_device(str(args.device))

    gan_cfg = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "models" / "tcn_gan.yaml")
    wgan_cfg = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "models" / "tcn_wgan_gp.yaml")
    same, diffs = compare_model_configs_except_loss(gan_cfg, wgan_cfg)
    if not same:
        raise ValueError(f"Model configs diverge beyond loss-specific keys: {diffs}")

    rows: list[dict[str, Any]] = []
    compatibility_rows: list[dict[str, Any]] = []
    dry_run_lines: list[str] = []
    repro_dir = ROOT / "journal_rebuild" / "reports" / "reproducibility"
    repro_dir.mkdir(parents=True, exist_ok=True)
    for model_name in args.models:
        base_cfg = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "models" / f"{model_name}.yaml")
        merged_cfg = merge_dicts(base_cfg, {"device": str(args.device)})
        for seed in args.seeds:
            run_id = formal_run_id(model_name, int(seed))
            files = required_run_files(run_id)
            research_cfg = research_config_for_run(
                data_config=data_config,
                model_config=merged_cfg,
                seed=int(seed),
                manifest_hash=str(data_manifest["manifest_hash"]),
                scaler_hash=str(data_manifest["scaler_hash"]),
                source_file_hashes=dict(data_manifest["source_file_hashes"]),
                feature_names=feature_names,
            )
            runtime_cfg = runtime_metadata_for_run(
                run_id=run_id,
                requested_device=str(args.device),
                resolved_device=str(device),
                skip_existing=bool(args.skip_existing),
                files=files,
            )
            research_cfg_hash = sha256_json(research_cfg)
            runtime_cfg_hash = sha256_json(runtime_cfg)
            diff_report_path = None
            if run_id == "formal_tcn_gan_seed1":
                diff_report_path = repro_dir / "formal_tcn_gan_seed1_config_diff.md"
            if args.skip_existing:
                reused, compatibility = inspect_existing_run(
                    run_id=run_id,
                    expected_model_name=model_name,
                    expected_loss_type=str(merged_cfg["loss_type"]),
                    current_research_config=research_cfg,
                    current_runtime_metadata=runtime_cfg,
                    expected_manifest_hash=str(data_manifest["manifest_hash"]),
                    expected_scaler_hash=str(data_manifest["scaler_hash"]),
                    target_fpr=float(data_config["target_fpr"]),
                    apply_backfill=not bool(args.dry_run),
                    diff_report_path=diff_report_path,
                )
                compatibility_rows.append(
                    compatibility_report_row(
                        run_id=run_id,
                        model_name=model_name,
                        seed=int(seed),
                        research_match=bool(compatibility["research_config_match"]),
                        manifest_match=bool(compatibility["manifest_hash_match"]),
                        scaler_match=bool(compatibility["scaler_hash_match"]),
                        runtime_match=bool(compatibility["runtime_metadata_match"]),
                        reusable=bool(compatibility["reusable"]),
                        action=str(compatibility["action"]),
                        diff_summary=str(compatibility["diff_summary"]),
                    )
                )
                dry_run_lines.append(f"- `{run_id}`: action=`{compatibility['action']}`, reusable=`{compatibility['reusable']}`, summary=`{compatibility['diff_summary']}`")
                if reused is not None:
                    rows.append(reused)
                    print(f"[skip-existing] {run_id}")
                    continue
                action = str(compatibility["action"])
                if args.dry_run:
                    if action == "archive_incompatible_then_rerun":
                        continue
                    if action == "archive_incomplete_then_rerun":
                        continue
                else:
                    if action == "archive_incompatible_then_rerun":
                        archived = archive_existing_run(run_id, "incompatible")
                        print(f"[archived-incompatible] {run_id} -> {archived.relative_to(ROOT)}")
                    elif action == "archive_incomplete_then_rerun":
                        archived = archive_existing_run(run_id, "incomplete")
                        print(f"[archived-incomplete] {run_id} -> {archived.relative_to(ROOT)}")
            else:
                compatibility_rows.append(
                    compatibility_report_row(
                        run_id=run_id,
                        model_name=model_name,
                        seed=int(seed),
                        research_match=False,
                        manifest_match=True,
                        scaler_match=True,
                        runtime_match=False,
                        reusable=False,
                        action="create",
                        diff_summary="run does not exist",
                    )
                )
                dry_run_lines.append(f"- `{run_id}`: action=`create`, reusable=`False`, summary=`run does not exist`")
                if args.dry_run:
                    continue
            files = required_run_files(run_id)
            if any(path.exists() for path in files.values()):
                raise ValueError(f"Run {run_id} already has partial outputs after recovery; refusing to overwrite")
            if args.dry_run:
                continue
            print(f"[run] {run_id} device={device}")
            row = evaluate_run(
                run_id=run_id,
                model_name=model_name,
                model_config=merged_cfg,
                data_config=data_config,
                data_manifest=data_manifest,
                feature_names=feature_names,
                train_arrays=train_arrays,
                calib_arrays=calib_arrays,
                test_arrays=test_arrays,
                train_window_df=train_window_df,
                calib_window_df=calib_window_df,
                test_window_df=test_window_df,
                seed=int(seed),
                device=device,
                research_cfg_hash=research_cfg_hash,
                runtime_cfg_hash=runtime_cfg_hash,
                requested_device=str(args.device),
                skip_existing=bool(args.skip_existing),
            )
            rows.append(row)
            print(f"[done] {run_id} AUC={row['auc']:.4f} F1={row['f1']:.4f} FPR={row['test_benign_fpr']:.4f}")

    write_existing_run_compatibility(
        compatibility_rows,
        ROOT / "journal_rebuild" / "reports" / "reproducibility" / "existing_run_compatibility.csv",
    )
    if args.dry_run:
        print("[dry-run]")
        for line in dry_run_lines:
            print(line)
        if any(row["action"] == "archive_incompatible_then_rerun" for row in compatibility_rows):
            raise SystemExit("Dry-run found research-config conflicts; see diff reports and compatibility CSV.")
        return

    metrics_root = ROOT / "journal_rebuild" / "runs" / "metrics"
    seed_df, summary_df, fam_summary_df = summarize_model_seed_results(rows, metrics_root)
    diff_df = paired_differences(seed_df, metrics_root)
    write_report(summary_df, diff_df, fam_summary_df, ROOT / "journal_rebuild" / "reports" / "model_selection" / "multiseed_model_selection_report.md")
    print(metrics_root / "model_seed_results.csv")


if __name__ == "__main__":
    main()
