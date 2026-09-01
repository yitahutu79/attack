#!/usr/bin/env python3
"""Minimal TCN-GAN runner for cross-dataset smoke validation.

This intentionally avoids the large experiment CLI so the second-dataset
bring-up path stays simple and robust.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TCN_GAN_DISABLE_WEIGHT_NORM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ATTACK = ROOT
if str(ATTACK) not in sys.path:
    sys.path.insert(0, str(ATTACK))

from dataset_loaders import (  # noqa: E402
    WindowedDataset,
    _fit_scaler,
    _frame_to_numeric_and_labels,
    _read_csv,
    _resolve_files,
    build_sequences,
    load_windowed_chrono_unsupervised_split,
    load_windowed_dataset,
    load_windowed_mixed_split,
    load_windowed_unsupervised_split,
    normalize_dataset_name,
)
from models.tcn_gan_experiment import (  # noqa: E402
    TCNDiscriminator,
    TCNGenerator,
    SequenceDataset,
    compute_anomaly_scores,
    metrics_at_threshold,
    set_seed,
    threshold_from_benign_fpr,
    train_one_epoch,
)


def split_windows_for_independent_calibration(
    windows: np.ndarray, calib_ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    n = int(len(windows))
    if n < 2:
        raise ValueError("independent calibration requires at least 2 train benign windows")
    ratio = float(calib_ratio)
    if not (0.0 < ratio < 1.0):
        raise ValueError("--calib-ratio must be in (0,1)")
    calib_n = int(n * ratio)
    calib_n = max(calib_n, 1)
    calib_n = min(calib_n, n - 1)
    train_n = n - calib_n
    return np.asarray(windows[:train_n]), np.asarray(windows[train_n:])


def split_records_for_independent_calibration(
    total_records: int, calib_ratio: float
) -> tuple[int, int]:
    n = int(total_records)
    if n < 2:
        raise ValueError("independent calibration requires at least 2 benign records")
    ratio = float(calib_ratio)
    if not (0.0 < ratio < 1.0):
        raise ValueError("--calib-ratio must be in (0,1)")
    calib_n = int(n * ratio)
    calib_n = max(calib_n, 1)
    calib_n = min(calib_n, n - 1)
    train_n = n - calib_n
    return int(train_n), int(calib_n)


def load_windowed_dataset_independent_calibration_strict(
    *,
    dataset: str,
    data_dir: str,
    train_files: list[str],
    test_files: list[str],
    window_size: int,
    stride: int,
    anomaly_ratio: float,
    calib_ratio: float,
    scaler: str = "minmax",
    clip_minmax: bool = True,
    progress: bool = False,
) -> tuple[WindowedDataset, str]:
    """Strict no-leakage loader for independent calibration.

    The default generic loader fits scaler before independent-calibration split.
    In this strict path, we split train benign records first and fit scaler only
    on model-train benign subset to avoid calibration leakage.
    """
    if not train_files or not test_files:
        raise ValueError(
            "strict no-leakage mode currently requires explicit --train-files and --test-files"
        )
    ds_name = normalize_dataset_name(dataset)
    train_paths = _resolve_files(data_dir, train_files)
    test_paths = _resolve_files(data_dir, test_files)

    train_frames_raw = []
    train_labels_raw: list[np.ndarray] = []
    test_frames = []
    test_labels: list[np.ndarray] = []

    for path in train_paths:
        numeric, y = _frame_to_numeric_and_labels(_read_csv(path), ds_name)
        train_frames_raw.append(numeric)
        train_labels_raw.append(y)
    for path in test_paths:
        numeric, y = _frame_to_numeric_and_labels(_read_csv(path), ds_name)
        test_frames.append(numeric)
        test_labels.append(y)

    common_cols = set(train_frames_raw[0].columns)
    for frame in train_frames_raw[1:] + test_frames:
        common_cols &= set(frame.columns)
    feature_names = [c for c in train_frames_raw[0].columns if c in common_cols]
    if not feature_names:
        raise ValueError("训练/测试文件之间没有共同数值特征列")

    train_frames_raw = [f.loc[:, feature_names].reset_index(drop=True) for f in train_frames_raw]
    test_frames = [f.loc[:, feature_names].reset_index(drop=True) for f in test_frames]

    benign_frames = []
    for frame, y in zip(train_frames_raw, train_labels_raw, strict=True):
        mask = (np.asarray(y) == 0)
        if np.any(mask):
            benign_frames.append(frame.loc[mask].reset_index(drop=True))
    if not benign_frames:
        raise ValueError("strict no-leakage requires benign records in train files")

    total_benign_records = int(sum(len(f) for f in benign_frames))
    train_record_target, _ = split_records_for_independent_calibration(total_benign_records, calib_ratio)

    model_train_frames = []
    calib_frames = []
    remain_train = int(train_record_target)
    for frame in benign_frames:
        n = int(len(frame))
        if remain_train <= 0:
            calib_frames.append(frame)
            continue
        if remain_train >= n:
            model_train_frames.append(frame)
            remain_train -= n
            continue
        model_train_frames.append(frame.iloc[:remain_train].reset_index(drop=True))
        calib_frames.append(frame.iloc[remain_train:].reset_index(drop=True))
        remain_train = 0

    model_train_frames = [f for f in model_train_frames if len(f) > 0]
    calib_frames = [f for f in calib_frames if len(f) > 0]
    if not model_train_frames or not calib_frames:
        raise ValueError("strict no-leakage split failed: empty train or calibration subset")

    model_train_labels = [np.zeros(len(f), dtype=np.uint8) for f in model_train_frames]
    scaler_obj = _fit_scaler(model_train_frames, model_train_labels, benign_only=True, scaler=scaler)

    def transform(frame):
        x = scaler_obj.transform(frame.astype(np.float32)).astype(np.float32)
        if clip_minmax and str(scaler).lower().strip() == "minmax":
            x = np.clip(x, 0.0, 1.0)
        return x

    train_xs: list[np.ndarray] = []
    train_ys: list[np.ndarray] = []
    for idx, frame in enumerate(model_train_frames, start=1):
        if len(frame) < int(window_size):
            continue
        labels = np.zeros(len(frame), dtype=np.uint8)
        xw, yw = build_sequences(
            transform(frame),
            labels,
            window_size,
            stride,
            anomaly_ratio,
            progress=progress,
            desc=f"strict train windows {idx}/{len(model_train_frames)}",
        )
        keep = (yw == 0)
        if np.any(keep):
            train_xs.append(xw[keep])
            train_ys.append(yw[keep])
    if not train_xs:
        raise ValueError("strict no-leakage train windows are empty")

    calib_xs: list[np.ndarray] = []
    calib_ys: list[np.ndarray] = []
    for idx, frame in enumerate(calib_frames, start=1):
        if len(frame) < int(window_size):
            continue
        labels = np.zeros(len(frame), dtype=np.uint8)
        xw, yw = build_sequences(
            transform(frame),
            labels,
            window_size,
            stride,
            anomaly_ratio,
            progress=progress,
            desc=f"strict calib windows {idx}/{len(calib_frames)}",
        )
        keep = (yw == 0)
        if np.any(keep):
            calib_xs.append(xw[keep])
            calib_ys.append(yw[keep])
    if not calib_xs:
        raise ValueError("strict no-leakage calibration windows are empty")

    test_xs: list[np.ndarray] = []
    test_ys: list[np.ndarray] = []
    for idx, (frame, y) in enumerate(zip(test_frames, test_labels, strict=True), start=1):
        if len(frame) < int(window_size):
            continue
        xw, yw = build_sequences(
            transform(frame),
            y,
            window_size,
            stride,
            anomaly_ratio,
            progress=progress,
            desc=f"strict test windows {idx}/{len(test_frames)}",
        )
        test_xs.append(xw)
        test_ys.append(yw)
    if not test_xs:
        raise ValueError("strict no-leakage test windows are empty")

    dataset_obj = WindowedDataset(
        x_train=np.concatenate(train_xs, axis=0),
        y_train=np.concatenate(train_ys, axis=0),
        x_test=np.concatenate(test_xs, axis=0),
        y_test=np.concatenate(test_ys, axis=0),
        x_calib=np.concatenate(calib_xs, axis=0),
        y_calib=np.concatenate(calib_ys, axis=0),
        feature_names=[str(c) for c in feature_names],
        train_files=[str(p.relative_to(Path(data_dir))) if Path(data_dir) in p.parents else str(p) for p in train_paths],
        test_files=[str(p.relative_to(Path(data_dir))) if Path(data_dir) in p.parents else str(p) for p in test_paths],
        calib_files=["independent_calibration_from_train_benign_records"],
        dataset=ds_name,
        scaler=str(scaler),
    )
    return dataset_obj, "model_train_benign only"


EPOCH_LOG_FIELDS = [
    "epoch",
    "total_epochs",
    "d_loss",
    "g_loss",
    "seconds",
    "selected_device",
    "dataset",
    "window_size",
    "stride",
    "anomaly_ratio_threshold",
    "score_mode",
    "alpha",
    "target_fpr",
    "strict_no_leakage",
    "independent_calibration",
    "calib_ratio",
]


def select_device(device_arg: str) -> torch.device:
    req = str(device_arg).lower().strip()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if req == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if req == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("--device mps requested but MPS is not available")
        return torch.device("mps")
    if req == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported --device: {device_arg}")


def append_epoch_log_row(out_csv: Path, row: dict[str, object]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not out_csv.exists()) or out_csv.stat().st_size == 0
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EPOCH_LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in EPOCH_LOG_FIELDS})
        f.flush()
        os.fsync(f.fileno())


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_epoch_checkpoint(
    *,
    checkpoint_dir: Path,
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    protocol_check: dict[str, object],
    selected_device: str,
    history: list[dict[str, float]] | None = None,
) -> tuple[Path, Path]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_epoch": int(epoch),
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "g_optimizer_state_dict": g_optimizer.state_dict(),
        "d_optimizer_state_dict": d_optimizer.state_dict(),
        "seed": int(args.seed),
        "torch_rng_state": torch.get_rng_state(),
        "args": vars(args),
        "protocol_check": dict(protocol_check),
        "selected_device": str(selected_device),
        "saved_at_unix": float(time.time()),
    }
    if torch.cuda.is_available():
        try:
            payload["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    if history is not None:
        payload["history"] = list(history)
    epoch_path = checkpoint_dir / f"epoch_{int(epoch):03d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(payload, epoch_path)
    torch.save(payload, latest_path)
    return epoch_path, latest_path


def _write_final_summary_md(payload: dict, out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    calibrated = payload.get("calibrated", {}) if isinstance(payload.get("calibrated"), dict) else {}
    lines: list[str] = []
    lines.append("# Proposed Run Summary")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Dataset | {payload.get('dataset')} |")
    lines.append(f"| Selected Device | {payload.get('selected_device')} |")
    lines.append(f"| Window Size | {payload.get('window_size')} |")
    lines.append(f"| Stride | {payload.get('stride')} |")
    lines.append(f"| Anomaly Ratio Threshold | {payload.get('anomaly_ratio')} |")
    lines.append(f"| Score Mode | {payload.get('score_mode')} |")
    lines.append(f"| Alpha | {payload.get('score_alpha')} |")
    lines.append(f"| Target FPR | {calibrated.get('target_fpr')} |")
    lines.append(f"| AUC | {metrics.get('auc')} |")
    lines.append(f"| AP | {metrics.get('ap')} |")
    lines.append(f"| F1 | {calibrated.get('f1')} |")
    lines.append(f"| Precision | {calibrated.get('precision')} |")
    lines.append(f"| Recall | {calibrated.get('recall')} |")
    lines.append(f"| Threshold | {calibrated.get('threshold')} |")
    lines.append(f"| Observed Test Benign FPR | {calibrated.get('test_benign_fpr')} |")
    lines.append(f"| FP/Test Benign | {calibrated.get('fp_over_test_benign')} |")
    lines.append(f"| Independent Calibration | {payload.get('independent_calibration')} |")
    lines.append(f"| Calib Ratio | {payload.get('calib_ratio')} |")
    lines.append(f"| Strict No Leakage | {payload.get('strict_no_leakage')} |")
    lines.append(f"| Checkpoint Dir | {payload.get('checkpoint_dir')} |")
    lines.append(f"| Resume From | {payload.get('resume_from')} |")
    protocol = payload.get("protocol_check", {}) if isinstance(payload.get("protocol_check"), dict) else {}
    lines.append(f"| scaler fit uses | {protocol.get('scaler_fit_uses')} |")
    lines.append(f"| model training uses | {protocol.get('model_training_uses')} |")
    lines.append(f"| threshold calibration uses | {protocol.get('threshold_calibration_uses')} |")
    lines.append(f"| TAD reference uses | {protocol.get('tad_reference_uses')} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal cross-dataset TCN-GAN runner")
    ap.add_argument("--dataset", default="swat", choices=["cicids2017", "swat", "ton_iot", "unsw_nb15", "generic"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-files", nargs="*", default=[])
    ap.add_argument("--test-files", nargs="*", default=[])
    ap.add_argument("--supervised-mixed-split", action="store_true")
    ap.add_argument("--mixed-files", nargs="*", default=[])
    ap.add_argument("--mixed-train-fraction", type=float, default=0.5)
    ap.add_argument("--unsupervised-formal-split", action="store_true")
    ap.add_argument("--unsup-benign-file", default="")
    ap.add_argument("--unsup-attack-file", default="")
    ap.add_argument("--unsup-train-fraction", type=float, default=0.6)
    ap.add_argument("--unsup-calib-fraction", type=float, default=0.2)
    ap.add_argument("--chrono-unsupervised-split", action="store_true")
    ap.add_argument("--chrono-file", default="")
    ap.add_argument("--chrono-train-fraction", type=float, default=0.6)
    ap.add_argument("--chrono-calib-fraction", type=float, default=0.1)
    ap.add_argument("--independent-calibration", action="store_true")
    ap.add_argument("--calib-ratio", type=float, default=0.2)
    ap.add_argument("--strict-no-leakage", action="store_true")
    ap.add_argument("--fit-scaler-on-model-train-only", action="store_true")
    ap.add_argument("--dry-run-protocol-check", action="store_true")
    ap.add_argument("--window-size", type=int, default=128)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--anomaly-ratio", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--test-batch-size", type=int, default=256)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--hidden-channels", nargs="+", type=int, default=[128, 128])
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gan-loss", choices=["vanilla", "wgan-gp"], default="vanilla")
    ap.add_argument("--gp-lambda", type=float, default=10.0)
    ap.add_argument("--n-critic", type=int, default=5)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--disc-pooling", choices=["attn", "mean"], default="attn")
    ap.add_argument("--score-mode", choices=["prob", "fused", "feat_l2", "feat_mahal"], default="prob")
    ap.add_argument("--score-alpha", type=float, default=0.24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--save-best", default="")
    ap.add_argument("--epoch-log-csv", default="", help="Optional per-epoch loss log CSV path.")
    ap.add_argument("--summary-md", default="", help="Optional final summary markdown path.")
    ap.add_argument("--checkpoint-dir", default="", help="Directory to save per-epoch checkpoints.")
    ap.add_argument("--resume-from", default="", help="Resume training from a checkpoint file.")
    args = ap.parse_args()

    out_json = Path(args.out_json).resolve()
    epoch_log_csv = (
        Path(args.epoch_log_csv).resolve()
        if str(args.epoch_log_csv).strip()
        else out_json.with_name(f"{out_json.stem}_epoch_log.csv")
    )
    summary_md = (
        Path(args.summary_md).resolve()
        if str(args.summary_md).strip()
        else out_json.with_name(f"{out_json.stem}_summary.md")
    )
    checkpoint_dir = (
        Path(args.checkpoint_dir).resolve()
        if str(args.checkpoint_dir).strip()
        else out_json.with_name(f"{out_json.stem}_checkpoints")
    )

    t0 = time.perf_counter()
    set_seed(int(args.seed))
    torch.set_num_threads(1)
    device = select_device(args.device)
    selected_device = str(device)
    print(f"Selected device: {selected_device}", flush=True)
    strict_no_leakage = bool(args.strict_no_leakage or args.fit_scaler_on_model_train_only)
    scaler_fit_uses = "train benign (default loader behavior)"

    if args.unsupervised_formal_split:
        ds = load_windowed_unsupervised_split(
            dataset=args.dataset,
            data_dir=args.data_dir,
            benign_file=args.unsup_benign_file or None,
            attack_file=args.unsup_attack_file or None,
            window_size=args.window_size,
            stride=args.stride,
            anomaly_ratio=args.anomaly_ratio,
            benign_train_fraction=args.unsup_train_fraction,
            benign_calib_fraction=args.unsup_calib_fraction,
            scaler="minmax",
            clip_minmax=True,
            progress=True,
        )
        x_train_benign = np.asarray(ds.x_train)
        x_calib_benign = np.asarray(ds.x_calib) if ds.x_calib is not None else None
        scaler_fit_uses = "unsupervised_formal_train_benign only"
    elif args.chrono_unsupervised_split:
        ds = load_windowed_chrono_unsupervised_split(
            dataset=args.dataset,
            data_dir=args.data_dir,
            mixed_file=args.chrono_file or None,
            window_size=args.window_size,
            stride=args.stride,
            anomaly_ratio=args.anomaly_ratio,
            train_fraction=args.chrono_train_fraction,
            calib_fraction=args.chrono_calib_fraction,
            scaler="minmax",
            clip_minmax=True,
            progress=True,
        )
        x_train_benign = np.asarray(ds.x_train)
        x_calib_benign = np.asarray(ds.x_calib) if ds.x_calib is not None else None
        scaler_fit_uses = "chrono_train_benign only"
    elif args.supervised_mixed_split:
        mixed = load_windowed_mixed_split(
            dataset=args.dataset,
            data_dir=args.data_dir,
            files=args.mixed_files,
            window_size=args.window_size,
            stride=args.stride,
            anomaly_ratio=args.anomaly_ratio,
            train_fraction=args.mixed_train_fraction,
            scaler="standard",
            clip_minmax=False,
            progress=True,
        )
        benign_mask = np.asarray(mixed.y_train) == 0
        if not np.any(benign_mask):
            raise ValueError("mixed split 训练集里没有 benign 窗口，无法训练 TCN 异常检测器")
        ds = mixed
        x_train_benign = np.asarray(mixed.x_train)[benign_mask]
        x_calib_benign = None
        scaler_fit_uses = "mixed_train records (supervised split)"
    else:
        if strict_no_leakage and args.independent_calibration:
            ds, scaler_fit_uses = load_windowed_dataset_independent_calibration_strict(
                dataset=args.dataset,
                data_dir=args.data_dir,
                train_files=list(args.train_files),
                test_files=list(args.test_files),
                window_size=args.window_size,
                stride=args.stride,
                anomaly_ratio=args.anomaly_ratio,
                calib_ratio=float(args.calib_ratio),
                scaler="minmax",
                clip_minmax=True,
                progress=True,
            )
            x_train_benign = np.asarray(ds.x_train)
            x_calib_benign = np.asarray(ds.x_calib) if ds.x_calib is not None else None
        else:
            ds = load_windowed_dataset(
                dataset=args.dataset,
                data_dir=args.data_dir,
                train_files=args.train_files,
                test_files=args.test_files,
                window_size=args.window_size,
                stride=args.stride,
                anomaly_ratio=args.anomaly_ratio,
                train_benign_only=True,
                scaler="minmax",
                clip_minmax=True,
                progress=True,
            )
            x_train_benign = np.asarray(ds.x_train)
            x_calib_benign = None

    if args.independent_calibration:
        if x_calib_benign is not None and len(x_calib_benign):
            print(
                "independent calibration requested but an explicit calibration split already exists; "
                "keeping existing calibration split.",
                flush=True,
            )
        else:
            x_train_benign, x_calib_benign = split_windows_for_independent_calibration(
                x_train_benign, float(args.calib_ratio)
            )
            print(
                f"Applied independent calibration on train benign windows: "
                f"train={len(x_train_benign)}, calib={len(x_calib_benign)}, ratio={float(args.calib_ratio):.2f}",
                flush=True,
            )

    model_training_uses = "model_train_benign"
    threshold_calibration_uses = (
        "independent_calibration_benign"
        if x_calib_benign is not None and len(x_calib_benign)
        else "train_benign_reuse"
    )
    tad_reference_uses = (
        "model_train_benign"
        if str(args.score_mode).lower().strip() != "prob"
        else "not_used_prob_mode"
    )
    protocol_meta = {
        "scaler_fit_uses": scaler_fit_uses,
        "model_training_uses": model_training_uses,
        "threshold_calibration_uses": threshold_calibration_uses,
        "tad_reference_uses": tad_reference_uses,
    }

    print(
        f"Loaded {args.dataset}: train_windows={len(ds.x_train)}, "
        f"test_windows={len(ds.x_test)}, test_anom_ratio={float(ds.y_test.mean()):.4f}, "
        f"features={len(ds.feature_names)}",
        flush=True,
    )
    if args.supervised_mixed_split:
        print(
            f"Using mixed split: train_anom_ratio={float(ds.y_train.mean()):.4f}, "
            f"benign_train_windows={len(x_train_benign)}, train_fraction={float(args.mixed_train_fraction):.2f}",
            flush=True,
        )
    elif args.unsupervised_formal_split:
        print(
            f"Using unsupervised formal split: benign_train_windows={len(x_train_benign)}, "
            f"benign_calib_windows={0 if x_calib_benign is None else len(x_calib_benign)}, "
            f"train_fraction={float(args.unsup_train_fraction):.2f}, calib_fraction={float(args.unsup_calib_fraction):.2f}",
            flush=True,
        )
    elif args.chrono_unsupervised_split:
        print(
            f"Using chrono unsupervised split: benign_train_windows={len(x_train_benign)}, "
            f"benign_calib_windows={0 if x_calib_benign is None else len(x_calib_benign)}, "
            f"train_fraction={float(args.chrono_train_fraction):.2f}, calib_fraction={float(args.chrono_calib_fraction):.2f}",
            flush=True,
        )
    if args.dry_run_protocol_check:
        payload = {
            "task": "cross_dataset_minimal_tcn_protocol_check",
            "dataset": args.dataset,
            "selected_device": selected_device,
            "window_size": int(args.window_size),
            "stride": int(args.stride),
            "anomaly_ratio": float(args.anomaly_ratio),
            "anomaly_ratio_threshold": float(args.anomaly_ratio),
            "target_fpr": float(args.target_fpr),
            "score_mode": str(args.score_mode),
            "score_alpha": float(args.score_alpha),
            "alpha": float(args.score_alpha),
            "independent_calibration": bool(args.independent_calibration),
            "calib_ratio": float(args.calib_ratio),
            "strict_no_leakage": bool(strict_no_leakage),
            "protocol_check": protocol_meta,
            "n_train_windows": int(len(x_train_benign)),
            "n_calib_windows": 0 if x_calib_benign is None else int(len(x_calib_benign)),
            "n_test_windows": int(len(ds.x_test)),
            "train_files": ds.train_files,
            "calib_files": ds.calib_files,
            "test_files": ds.test_files,
            "checkpoint_dir": str(checkpoint_dir),
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved protocol-check JSON (dry-run, no training): {out_json}", flush=True)
        return

    train_loader = DataLoader(
        SequenceDataset(x_train_benign), batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    generator = TCNGenerator(
        args.window_size, x_train_benign.shape[2], args.latent_dim, args.hidden_channels, args.dropout
    ).to(device)
    discriminator = TCNDiscriminator(
        args.window_size, x_train_benign.shape[2], args.hidden_channels, args.dropout, pooling=args.disc_pooling
    ).to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    history: list[dict[str, float]] = []
    start_epoch = 1
    resumed_from: str | None = None
    if str(args.resume_from).strip():
        resume_path = Path(args.resume_from).resolve()
        ckpt = torch.load(resume_path, map_location="cpu")
        g_state = ckpt.get("generator_state_dict", ckpt.get("generator"))
        d_state = ckpt.get("discriminator_state_dict", ckpt.get("discriminator"))
        if g_state is None or d_state is None:
            raise ValueError(f"Invalid checkpoint (missing model states): {resume_path}")
        generator.load_state_dict(g_state)
        discriminator.load_state_dict(d_state)
        if ckpt.get("g_optimizer_state_dict") is not None:
            try:
                g_optimizer.load_state_dict(ckpt["g_optimizer_state_dict"])
                move_optimizer_state_to_device(g_optimizer, device)
            except Exception as e:
                print(f"Warning: failed to load generator optimizer state: {e}", flush=True)
        if ckpt.get("d_optimizer_state_dict") is not None:
            try:
                d_optimizer.load_state_dict(ckpt["d_optimizer_state_dict"])
                move_optimizer_state_to_device(d_optimizer, device)
            except Exception as e:
                print(f"Warning: failed to load discriminator optimizer state: {e}", flush=True)
        resumed_epoch = int(ckpt.get("checkpoint_epoch", 0))
        start_epoch = resumed_epoch + 1
        resumed_from = str(resume_path)
        old_history = ckpt.get("history")
        if isinstance(old_history, list):
            history.extend(old_history)
        print(
            f"Resumed from checkpoint: {resume_path} (completed_epoch={resumed_epoch}, next_epoch={start_epoch})",
            flush=True,
        )

    final_target_epoch = int(args.epochs)
    if start_epoch > final_target_epoch:
        print(
            f"Resume checkpoint already reached target epochs (next_epoch={start_epoch}, target={final_target_epoch}); "
            "skip training and run final evaluation only.",
            flush=True,
        )

    for epoch in range(start_epoch, final_target_epoch + 1):
        e0 = time.perf_counter()
        d_loss, g_loss = train_one_epoch(
            generator,
            discriminator,
            train_loader,
            device,
            g_optimizer,
            d_optimizer,
            criterion,
            args.latent_dim,
            gan_loss=str(args.gan_loss),
            gp_lambda=float(args.gp_lambda),
            n_critic=int(args.n_critic),
            progress=True,
            desc=f"train ep{epoch}/{args.epochs}",
        )
        seconds = time.perf_counter() - e0
        history_row = {"epoch": float(epoch), "d_loss": float(d_loss), "g_loss": float(g_loss), "seconds": seconds}
        history.append(history_row)
        print(
            f"epoch {epoch}/{final_target_epoch}: d_loss={d_loss:.4f}, g_loss={g_loss:.4f}, seconds={seconds:.2f}",
            flush=True,
        )
        append_epoch_log_row(
            epoch_log_csv,
            {
                "epoch": int(epoch),
                "total_epochs": int(final_target_epoch),
                "d_loss": float(d_loss),
                "g_loss": float(g_loss),
                "seconds": float(seconds),
                "selected_device": selected_device,
                "dataset": str(args.dataset),
                "window_size": int(args.window_size),
                "stride": int(args.stride),
                "anomaly_ratio_threshold": float(args.anomaly_ratio),
                "score_mode": str(args.score_mode),
                "alpha": float(args.score_alpha),
                "target_fpr": float(args.target_fpr),
                "strict_no_leakage": bool(strict_no_leakage),
                "independent_calibration": bool(args.independent_calibration),
                "calib_ratio": float(args.calib_ratio),
            },
        )
        epoch_ckpt, latest_ckpt = save_epoch_checkpoint(
            checkpoint_dir=checkpoint_dir,
            epoch=int(epoch),
            generator=generator,
            discriminator=discriminator,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            args=args,
            protocol_check=protocol_meta,
            selected_device=selected_device,
            history=history,
        )
        print(f"Saved epoch checkpoint: {epoch_ckpt}", flush=True)
        print(f"Updated latest checkpoint: {latest_ckpt}", flush=True)

    y_true = np.asarray(ds.y_test, dtype=np.int64)
    y_score = compute_anomaly_scores(
        discriminator,
        ds.x_test,
        device,
        args.test_batch_size,
        score_mode=args.score_mode,
        score_alpha=args.score_alpha,
        x_ref_benign=x_train_benign if str(args.score_mode).lower().strip() != "prob" else None,
    )
    calib_source = x_calib_benign if x_calib_benign is not None and len(x_calib_benign) else x_train_benign
    train_scores = compute_anomaly_scores(
        discriminator,
        calib_source,
        device,
        args.test_batch_size,
        score_mode=args.score_mode,
        score_alpha=args.score_alpha,
        x_ref_benign=x_train_benign if str(args.score_mode).lower().strip() != "prob" else None,
    )
    threshold = threshold_from_benign_fpr(train_scores, args.target_fpr)
    calibrated = metrics_at_threshold(y_true, y_score, threshold)
    train_benign_fpr = float(np.mean(train_scores >= threshold))
    test_benign_count = int((y_true == 0).sum())
    fp_count = int(round(float(calibrated.get("fp", 0.0))))
    fp_over_test_benign = f"{fp_count}/{test_benign_count}" if test_benign_count > 0 else f"{fp_count}/0"

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        auc = float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size > 1 else None
        ap_score = float(average_precision_score(y_true, y_score)) if float(y_true.sum()) > 0 else 0.0
    except Exception:
        auc = None
        ap_score = 0.0

    payload = {
        "task": "cross_dataset_minimal_tcn",
        "dataset": args.dataset,
        "selected_device": selected_device,
        "data_dir": args.data_dir,
        "train_files": ds.train_files,
        "test_files": ds.test_files,
        "calib_files": ds.calib_files,
        "checkpoint_dir": str(checkpoint_dir),
        "resume_from": resumed_from,
        "supervised_mixed_split": bool(args.supervised_mixed_split),
        "mixed_train_fraction": float(args.mixed_train_fraction) if args.supervised_mixed_split else None,
        "unsupervised_formal_split": bool(args.unsupervised_formal_split),
        "unsup_train_fraction": float(args.unsup_train_fraction) if args.unsupervised_formal_split else None,
        "unsup_calib_fraction": float(args.unsup_calib_fraction) if args.unsupervised_formal_split else None,
        "chrono_unsupervised_split": bool(args.chrono_unsupervised_split),
        "chrono_train_fraction": float(args.chrono_train_fraction) if args.chrono_unsupervised_split else None,
        "chrono_calib_fraction": float(args.chrono_calib_fraction) if args.chrono_unsupervised_split else None,
        "independent_calibration": bool(args.independent_calibration),
        "calib_ratio": float(args.calib_ratio),
        "strict_no_leakage": bool(strict_no_leakage),
        "protocol_check": protocol_meta,
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "anomaly_ratio": float(args.anomaly_ratio),
        "anomaly_ratio_threshold": float(args.anomaly_ratio),
        "target_fpr": float(args.target_fpr),
        "feature_dim": int(x_train_benign.shape[2]),
        "n_train_windows": int(len(x_train_benign)),
        "n_train_windows_raw": int(len(ds.x_train)),
        "n_calib_windows": 0 if x_calib_benign is None else int(len(x_calib_benign)),
        "n_test_windows": int(len(ds.x_test)),
        "train_anomaly_ratio": float(ds.y_train.mean()) if len(ds.y_train) else 0.0,
        "test_anomaly_ratio": float(y_true.mean()) if len(y_true) else 0.0,
        "model": {
            "disc_pooling": str(args.disc_pooling),
            "gan_loss": str(args.gan_loss),
            "gp_lambda": float(args.gp_lambda),
            "n_critic": int(args.n_critic),
            "weight_norm_disabled": True,
        },
        "score_mode": str(args.score_mode),
        "score_alpha": float(args.score_alpha),
        "alpha": float(args.score_alpha),
        "history": history,
        "metrics": {
            "auc": auc,
            "ap": ap_score,
        },
        "calibrated": {
            **calibrated,
            "target_fpr": float(args.target_fpr),
            "train_benign_fpr": train_benign_fpr,
            "test_benign_fpr": None if int((y_true == 0).sum()) == 0 else float(calibrated["fpr"]),
            "observed_test_benign_fpr": None if int((y_true == 0).sum()) == 0 else float(calibrated["fpr"]),
            "test_benign_count": test_benign_count,
            "fp_over_test_benign": fp_over_test_benign,
        },
        "timing": {
            "total_seconds": float(time.perf_counter() - t0),
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_final_summary_md(payload, summary_md)
    if args.save_best:
        torch.save(
            {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "args": vars(args),
                "payload": payload,
            },
            args.save_best,
        )
    print(f"Saved JSON: {out_json}", flush=True)
    print(f"Saved epoch log CSV: {epoch_log_csv}", flush=True)
    print(f"Saved summary MD: {summary_md}", flush=True)
    print(
        "Calibrated: F1={f1:.4f}, recall={recall:.4f}, precision={precision:.4f}, train_FPR={train_benign_fpr:.4f}".format(
            **payload["calibrated"]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
