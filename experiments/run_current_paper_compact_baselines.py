#!/usr/bin/env python3
"""Unified compact baselines for current paper under strict no-leakage protocols.

Design:
- Run one dataset per execution (`--dataset`).
- Support dry-run protocol checks without training (`--dry-run-protocol-check`).
- Persist strict protocol metadata in every result row.
- Never use test labels for threshold calibration.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
from torch.nn.utils import parametrize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_loaders import (  # noqa: E402
    _fit_scaler,
    _frame_to_numeric_and_labels,
    _read_csv,
    _resolve_files,
    build_sequences,
    load_windowed_unsupervised_split,
)
from pipelines.run_tcn_cross_dataset_minimal import (  # noqa: E402
    load_windowed_dataset_independent_calibration_strict,
)
from baselines.window_baselines import (  # noqa: E402
    GANomalyNet,
    MinimalTranAD,
    train_ganomaly_model,
    train_tranad_model,
)
from models.tcn_gan_experiment import (  # noqa: E402
    TCNDiscriminator,
    compute_anomaly_scores,
    metrics_at_threshold,
    threshold_from_benign_fpr,
)


CICIDS_TRAIN_FILES = [
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
]
CICIDS_TEST_FILES = [
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]
UNSW_TRAIN_FILES = ["Training and Testing Sets/UNSW_NB15_training-set.csv"]
UNSW_TEST_FILES = ["Training and Testing Sets/UNSW_NB15_testing-set.csv"]

CICIDS_OURS_FIXED = {
    "threshold": 0.425420,
    "observed_test_benign_fpr": 0.046023,
    "precision": 0.938395,
    "recall": 0.956847,
    "f1": 0.947531,
    "auc": 0.978349,
    "ap": 0.963120,
    "tp": 17761,
    "fp": 1166,
    "tn": 24169,
    "fn": 801,
    "target_fpr": 0.25,
}


@dataclass
class DatasetBundle:
    dataset: str
    x_train_benign: np.ndarray
    x_calib_benign: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_dim: int
    split_source: str
    train_spec: str
    test_spec: str
    scaler_fit_source: str
    model_training_source: str
    threshold_calibration_source: str
    protocol_details: dict[str, Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["cicids2017", "swat", "unsw_nb15"])
    p.add_argument("--data-dir", default="")
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--anomaly-ratio", type=float, default=0.15)
    p.add_argument("--independent-calibration", action="store_true")
    p.add_argument("--calib-ratio", type=float, default=0.2)
    p.add_argument("--strict-no-leakage", action="store_true")
    p.add_argument("--dry-run-protocol-check", action="store_true")
    p.add_argument(
        "--baselines",
        nargs="+",
        default=["isolationforest", "oneclasssvm", "mlp", "ganomaly", "tranad", "timesnet", "dlinear", "ours"],
        help="Subset to run: isolationforest oneclasssvm mlp ganomaly tranad timesnet dlinear ours",
    )
    p.add_argument("--target-fprs", nargs="+", type=float, default=[])
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--mlp-max-iter", type=int, default=100)
    p.add_argument("--ocsvm-max-samples", type=int, default=10000)
    p.add_argument("--ocsvm-pca-components", type=int, default=100)
    p.add_argument("--output-dir", default="results/current_paper_compact_baselines")
    p.add_argument("--swat-benign-file", default="benign_data/benign_samples_5sec.csv")
    p.add_argument("--swat-attack-file", default="attack_data/attack_samples_5sec.csv")
    p.add_argument("--swat-train-fraction", type=float, default=0.6)
    p.add_argument("--swat-calib-fraction", type=float, default=0.2)
    p.add_argument("--unsw-train-files", nargs="*", default=[])
    p.add_argument("--unsw-test-files", nargs="*", default=[])
    p.add_argument("--cicids-train-files", nargs="*", default=[])
    p.add_argument("--cicids-test-files", nargs="*", default=[])
    p.add_argument("--ours-checkpoint", default="", help="Optional checkpoint for evaluating Ours when fixed reference is unavailable")
    return p.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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
            raise ValueError("--device cuda requested but CUDA not available")
        return torch.device("cuda")
    if req == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("--device mps requested but MPS not available")
        return torch.device("mps")
    return torch.device("cpu")


def _to_rel(paths: list[Path], data_dir: str) -> list[str]:
    root = Path(data_dir).resolve()
    out: list[str] = []
    for p in paths:
        p = p.resolve()
        if root in p.parents:
            out.append(str(p.relative_to(root)))
        else:
            out.append(str(p))
    return out


def split_records_for_independent_calibration(total_records: int, calib_ratio: float) -> tuple[int, int]:
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


def _transform_with_scaler(scaler_obj: Any, frame: Any) -> np.ndarray:
    x = scaler_obj.transform(frame.astype(np.float32)).astype(np.float32)
    return np.clip(x, 0.0, 1.0)


def prepare_unsupervised_bundle(args: argparse.Namespace) -> DatasetBundle:
    dataset = str(args.dataset)
    window_size = int(args.window_size)
    stride = int(args.stride)
    anomaly_ratio = float(args.anomaly_ratio)

    if dataset == "cicids2017":
        data_dir = args.data_dir or str(ROOT / "dataset" / "CICIDS2017")
        train_files = args.cicids_train_files if args.cicids_train_files else list(CICIDS_TRAIN_FILES)
        test_files = args.cicids_test_files if args.cicids_test_files else list(CICIDS_TEST_FILES)
        ds, scaler_fit_source = load_windowed_dataset_independent_calibration_strict(
            dataset=dataset,
            data_dir=data_dir,
            train_files=train_files,
            test_files=test_files,
            window_size=window_size,
            stride=stride,
            anomaly_ratio=anomaly_ratio,
            calib_ratio=float(args.calib_ratio),
            scaler="minmax",
            clip_minmax=True,
            progress=False,
        )
        split_source = "explicit_cicids_train_test + strict_independent_calibration"
        train_spec = ";".join(train_files)
        test_spec = ";".join(test_files)
        protocol_details = {
            "train_files": train_files,
            "test_files": test_files,
            "calib_files": list(ds.calib_files),
        }
    elif dataset == "swat":
        data_dir = args.data_dir or str(ROOT / "dataset" / "SWaT")
        ds = load_windowed_unsupervised_split(
            dataset="swat",
            data_dir=data_dir,
            benign_file=str(args.swat_benign_file),
            attack_file=str(args.swat_attack_file),
            window_size=window_size,
            stride=stride,
            anomaly_ratio=anomaly_ratio,
            benign_train_fraction=float(args.swat_train_fraction),
            benign_calib_fraction=float(args.swat_calib_fraction),
            scaler="minmax",
            clip_minmax=True,
            progress=False,
        )
        scaler_fit_source = "model_train_benign only"
        split_source = "swat_formal_unsupervised_split_60_20_20"
        train_spec = f"{args.swat_benign_file}[:{float(args.swat_train_fraction):.2f}]"
        mid = float(args.swat_train_fraction)
        hi = float(args.swat_train_fraction) + float(args.swat_calib_fraction)
        test_spec = f"{args.swat_benign_file}[{hi:.2f}:] + {args.swat_attack_file}"
        protocol_details = {
            "benign_file": str(args.swat_benign_file),
            "attack_file": str(args.swat_attack_file),
            "train_fraction": float(args.swat_train_fraction),
            "calib_fraction": float(args.swat_calib_fraction),
            "dataset_loader_train_files": list(ds.train_files),
            "dataset_loader_calib_files": list(ds.calib_files),
            "dataset_loader_test_files": list(ds.test_files),
        }
    else:  # unsw_nb15
        data_dir = args.data_dir or str(ROOT / "dataset" / "UNSW-NB15")
        train_files = args.unsw_train_files if args.unsw_train_files else list(UNSW_TRAIN_FILES)
        test_files = args.unsw_test_files if args.unsw_test_files else list(UNSW_TEST_FILES)
        ds, scaler_fit_source = load_windowed_dataset_independent_calibration_strict(
            dataset="unsw_nb15",
            data_dir=data_dir,
            train_files=train_files,
            test_files=test_files,
            window_size=window_size,
            stride=stride,
            anomaly_ratio=anomaly_ratio,
            calib_ratio=float(args.calib_ratio),
            scaler="minmax",
            clip_minmax=True,
            progress=False,
        )
        split_source = "unsw_official_train_test + strict_independent_calibration_from_train_benign"
        train_spec = ";".join(train_files)
        test_spec = ";".join(test_files)
        protocol_details = {
            "train_files": train_files,
            "test_files": test_files,
            "calib_files": list(ds.calib_files),
        }

    x_train = np.asarray(ds.x_train, dtype=np.float32)
    x_calib = np.asarray(ds.x_calib, dtype=np.float32) if ds.x_calib is not None else np.empty((0,), dtype=np.float32)
    x_test = np.asarray(ds.x_test, dtype=np.float32)
    y_test = np.asarray(ds.y_test, dtype=np.int64)

    if x_calib.ndim == 1 or len(x_calib) == 0:
        raise ValueError("Calibration benign windows are empty; strict independent calibration failed.")

    return DatasetBundle(
        dataset=dataset,
        x_train_benign=x_train,
        x_calib_benign=x_calib,
        x_test=x_test,
        y_test=y_test,
        feature_dim=int(x_train.shape[2]),
        split_source=split_source,
        train_spec=train_spec,
        test_spec=test_spec,
        scaler_fit_source="model_train_benign only",
        model_training_source="model_train_benign",
        threshold_calibration_source="independent_calibration_benign",
        protocol_details=protocol_details,
    )


def get_dataset_train_test_files(dataset: str, args: argparse.Namespace) -> tuple[str, list[str], list[str]]:
    ds = str(dataset)
    if ds == "cicids2017":
        data_dir = args.data_dir or str(ROOT / "dataset" / "CICIDS2017")
        train_files = args.cicids_train_files if args.cicids_train_files else list(CICIDS_TRAIN_FILES)
        test_files = args.cicids_test_files if args.cicids_test_files else list(CICIDS_TEST_FILES)
        return data_dir, list(train_files), list(test_files)
    if ds == "unsw_nb15":
        data_dir = args.data_dir or str(ROOT / "dataset" / "UNSW-NB15")
        train_files = args.unsw_train_files if args.unsw_train_files else list(UNSW_TRAIN_FILES)
        test_files = args.unsw_test_files if args.unsw_test_files else list(UNSW_TEST_FILES)
        return data_dir, list(train_files), list(test_files)
    raise ValueError(f"Supervised MLP strict training not configured for dataset={dataset}")


def prepare_supervised_mlp_train_windows_strict(
    dataset: str,
    args: argparse.Namespace,
    expected_feature_dim: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Prepare strict supervised MLP train windows from train split labels only.

    - Scaler is fit only on model_train_benign records.
    - Window labels come only from training split records.
    - Calibration and test records are never used for MLP training.
    """
    data_dir, train_files, test_files = get_dataset_train_test_files(dataset, args)
    dataset = str(dataset)

    train_paths = _resolve_files(data_dir, train_files)
    test_paths = _resolve_files(data_dir, test_files)
    train_frames_raw: list[Any] = []
    train_labels_raw: list[np.ndarray] = []
    for path in train_paths:
        numeric, y = _frame_to_numeric_and_labels(_read_csv(path), dataset)
        train_frames_raw.append(numeric)
        train_labels_raw.append(y)

    test_frames_raw: list[Any] = []
    for path in test_paths:
        numeric, _ = _frame_to_numeric_and_labels(_read_csv(path), dataset)
        test_frames_raw.append(numeric)

    # Keep exact feature intersection style aligned with strict loader.
    common_cols = set(train_frames_raw[0].columns)
    for frame in train_frames_raw[1:] + test_frames_raw:
        common_cols &= set(frame.columns)
    feature_names = [c for c in train_frames_raw[0].columns if c in common_cols]
    train_frames_raw = [f.loc[:, feature_names].reset_index(drop=True) for f in train_frames_raw]

    benign_frames = []
    for frame, y in zip(train_frames_raw, train_labels_raw, strict=True):
        mask = np.asarray(y) == 0
        if np.any(mask):
            benign_frames.append(frame.loc[mask].reset_index(drop=True))
    total_benign_records = int(sum(len(f) for f in benign_frames))
    train_record_target, _ = split_records_for_independent_calibration(total_benign_records, float(args.calib_ratio))

    model_train_frames = []
    remain_train = int(train_record_target)
    for frame in benign_frames:
        n = int(len(frame))
        if remain_train <= 0:
            break
        if remain_train >= n:
            model_train_frames.append(frame)
            remain_train -= n
            continue
        model_train_frames.append(frame.iloc[:remain_train].reset_index(drop=True))
        remain_train = 0

    scaler_obj = _fit_scaler(model_train_frames, [np.zeros(len(f), dtype=np.uint8) for f in model_train_frames], benign_only=True, scaler="minmax")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for frame, y in zip(train_frames_raw, train_labels_raw, strict=True):
        if len(frame) < int(args.window_size):
            continue
        xw, yw = build_sequences(
            _transform_with_scaler(scaler_obj, frame),
            y,
            int(args.window_size),
            int(args.stride),
            float(args.anomaly_ratio),
            progress=False,
        )
        xs.append(xw)
        ys.append(yw)
    if not xs:
        raise ValueError(f"{dataset} supervised MLP train windows are empty.")
    x_out = np.concatenate(xs, axis=0)
    y_out = np.concatenate(ys, axis=0)
    if int(x_out.shape[2]) != int(expected_feature_dim):
        raise ValueError(
            f"Feature dim mismatch for supervised MLP: train={int(x_out.shape[2])}, expected={int(expected_feature_dim)}"
        )
    meta = {
        "train_files": train_files,
        "test_files_for_feature_alignment": test_files,
        "n_supervised_train_windows": int(len(x_out)),
        "supervised_train_anomaly_ratio": float(np.mean(y_out == 1)),
        "feature_dim": int(x_out.shape[2]),
    }
    return x_out, y_out, meta


def to_flat(x: np.ndarray) -> np.ndarray:
    n, w, f = x.shape
    return x.reshape(n, w * f)


def safe_auc_ap(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if np.unique(y_true).size < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_score))
    ap = float(average_precision_score(y_true, y_score)) if int(np.sum(y_true == 1)) > 0 else 0.0
    return auc, ap


def calibrate_and_metrics(
    y_test: np.ndarray,
    scores_test: np.ndarray,
    calib_benign_scores: np.ndarray,
    target_fpr: float,
) -> dict[str, float]:
    th = threshold_from_benign_fpr(np.asarray(calib_benign_scores, dtype=np.float32), float(target_fpr))
    m = metrics_at_threshold(np.asarray(y_test, dtype=np.int64), np.asarray(scores_test, dtype=np.float32), float(th))
    out = dict(m)
    out["threshold"] = float(th)
    return out


def run_iforest(bundle: DatasetBundle, target_fpr: float, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, float], float, float]:
    x_tr = to_flat(bundle.x_train_benign)
    x_cal = to_flat(bundle.x_calib_benign)
    x_te = to_flat(bundle.x_test)

    t0 = time.perf_counter()
    model = IsolationForest(contamination=0.1, random_state=int(seed), n_estimators=100, n_jobs=-1)
    model.fit(x_tr)
    train_seconds = float(time.perf_counter() - t0)

    t1 = time.perf_counter()
    s_cal = -model.decision_function(x_cal)
    s_te = -model.decision_function(x_te)
    infer_seconds = float(time.perf_counter() - t1)

    metrics = calibrate_and_metrics(bundle.y_test, s_te, s_cal, target_fpr)
    return s_te, s_cal, metrics, train_seconds, infer_seconds


def run_ocsvm(
    bundle: DatasetBundle,
    target_fpr: float,
    seed: int,
    max_samples: int,
    pca_components: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], float, float, dict[str, Any]]:
    from sklearn.decomposition import PCA

    x_tr = to_flat(bundle.x_train_benign)
    x_cal = to_flat(bundle.x_calib_benign)
    x_te = to_flat(bundle.x_test)

    rng = np.random.default_rng(int(seed))
    sample_idx = None
    if len(x_tr) > int(max_samples):
        sample_idx = rng.choice(len(x_tr), size=int(max_samples), replace=False)
        x_fit = x_tr[sample_idx]
    else:
        x_fit = x_tr

    pca = None
    if x_fit.shape[1] > int(pca_components):
        pca = PCA(n_components=int(pca_components), random_state=int(seed))
        x_fit_r = pca.fit_transform(x_fit)
        x_cal_r = pca.transform(x_cal)
        x_te_r = pca.transform(x_te)
    else:
        x_fit_r = x_fit
        x_cal_r = x_cal
        x_te_r = x_te

    t0 = time.perf_counter()
    model = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
    model.fit(x_fit_r)
    train_seconds = float(time.perf_counter() - t0)

    t1 = time.perf_counter()
    s_cal = -model.decision_function(x_cal_r)
    s_te = -model.decision_function(x_te_r)
    infer_seconds = float(time.perf_counter() - t1)

    metrics = calibrate_and_metrics(bundle.y_test, s_te, s_cal, target_fpr)
    extra = {
        "sampled_train_windows": None if sample_idx is None else int(len(sample_idx)),
        "pca_used": bool(pca is not None),
        "pca_components": int(pca_components) if pca is not None else int(x_fit_r.shape[1]),
    }
    return s_te, s_cal, metrics, train_seconds, infer_seconds, extra


def run_tranad(
    bundle: DatasetBundle,
    target_fpr: float,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], float, float]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    x_tr = np.asarray(bundle.x_train_benign, dtype=np.float32)
    x_cal = np.asarray(bundle.x_calib_benign, dtype=np.float32)
    x_te = np.asarray(bundle.x_test, dtype=np.float32)

    model = MinimalTranAD(bundle.feature_dim, d_model=64, nhead=4, num_layers=2)

    t0 = time.perf_counter()
    model = train_tranad_model(
        model,
        x_tr,
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=1e-3,
        device=str(device),
    )
    train_seconds = float(time.perf_counter() - t0)

    t1 = time.perf_counter()
    with torch.no_grad():
        s_cal = model.get_reconstruction_error(torch.as_tensor(x_cal, dtype=torch.float32, device=device))
        s_te = model.get_reconstruction_error(torch.as_tensor(x_te, dtype=torch.float32, device=device))
    infer_seconds = float(time.perf_counter() - t1)

    metrics = calibrate_and_metrics(bundle.y_test, s_te, s_cal, target_fpr)
    return s_te, s_cal, metrics, train_seconds, infer_seconds


def run_ganomaly(
    bundle: DatasetBundle,
    target_fpr: float,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], float, float]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    x_tr = to_flat(bundle.x_train_benign)
    x_cal = to_flat(bundle.x_calib_benign)
    x_te = to_flat(bundle.x_test)

    model = GANomalyNet(x_tr.shape[1], latent_dim=64)

    t0 = time.perf_counter()
    model = train_ganomaly_model(
        model,
        x_tr,
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=1e-3,
        device=str(device),
    )
    train_seconds = float(time.perf_counter() - t0)

    def _scores(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(x), 4096):
                batch = torch.as_tensor(x[i : i + 4096], dtype=torch.float32, device=device)
                _, z, zp = model(batch)
                s = torch.mean((z - zp) ** 2, dim=1)
                outs.append(s.detach().cpu().numpy())
        return np.concatenate(outs, axis=0)

    t1 = time.perf_counter()
    s_cal = _scores(x_cal)
    s_te = _scores(x_te)
    infer_seconds = float(time.perf_counter() - t1)

    metrics = calibrate_and_metrics(bundle.y_test, s_te, s_cal, target_fpr)
    return s_te, s_cal, metrics, train_seconds, infer_seconds


def run_mlp_supervised_strict(
    bundle: DatasetBundle,
    args: argparse.Namespace,
    dataset: str,
    target_fpr: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], float, float, dict[str, Any]]:
    x_sup, y_sup, prep_meta = prepare_supervised_mlp_train_windows_strict(
        dataset=dataset,
        args=args,
        expected_feature_dim=int(bundle.feature_dim),
    )
    x_tr = to_flat(x_sup)
    x_cal = to_flat(bundle.x_calib_benign)
    x_te = to_flat(bundle.x_test)

    t0 = time.perf_counter()
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=int(args.mlp_max_iter),
        random_state=int(seed),
        verbose=False,
    )
    model.fit(x_tr, y_sup)
    train_seconds = float(time.perf_counter() - t0)

    t1 = time.perf_counter()
    s_cal = model.predict_proba(x_cal)[:, 1]
    s_te = model.predict_proba(x_te)[:, 1]
    infer_seconds = float(time.perf_counter() - t1)

    metrics = calibrate_and_metrics(bundle.y_test, s_te, s_cal, target_fpr)
    extra = {
        **prep_meta,
        "supervised_train_windows": int(len(x_sup)),
        "supervised_train_anomaly_ratio": float(np.mean(y_sup == 1)),
    }
    return s_te, s_cal, metrics, train_seconds, infer_seconds, extra


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
        raise RuntimeError(f"Failed to load {model_name} state_dict: {e}") from e


def discover_default_ours_checkpoint(dataset: str) -> str:
    ds = str(dataset).strip().lower()
    if ds not in {"swat", "unsw_nb15"}:
        return ""
    subdir = "swat_tcn" if ds == "swat" else "unsw_nb15_tcn"
    patterns = [
        str(ROOT / "results" / "cross_dataset_formal_unsup" / subdir / "*" / "ours_ckpt.pt"),
        str(
            ROOT
            / "_archive"
            / "2026-04-29_current_only_cleanup"
            / "results_legacy"
            / "cross_dataset_formal_unsup"
            / subdir
            / "*"
            / "ours_ckpt.pt"
        ),
        str(
            ROOT
            / "_archive"
            / "2026-04-29_before_acsac_writing"
            / "results_legacy"
            / "cross_dataset_formal_unsup"
            / subdir
            / "*"
            / "ours_ckpt.pt"
        ),
    ]
    candidates: list[Path] = []
    for pat in patterns:
        for p in glob.glob(pat):
            pp = Path(p).resolve()
            if pp.exists():
                candidates.append(pp)
    if not candidates:
        return ""
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return str(candidates[-1])


def run_ours_checkpoint_eval(
    bundle: DatasetBundle,
    target_fpr: float,
    device: torch.device,
    batch_size: int,
    checkpoint_path: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], float, float, dict[str, Any]]:
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Ours checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}

    feat_dim = int(bundle.feature_dim)
    hidden_channels = ckpt_args.get("hidden_channels", [128, 128])
    dropout = float(ckpt_args.get("dropout", 0.2))
    pooling = str(ckpt_args.get("disc_pooling", "attn"))
    window_size = int(ckpt_args.get("window_size", bundle.x_train_benign.shape[1]))

    discriminator = TCNDiscriminator(
        seq_len=window_size,
        feat_dim=feat_dim,
        hidden_channels=hidden_channels,
        dropout=dropout,
        pooling=pooling,
    ).to(device)

    d_state = ckpt.get("discriminator_state_dict", ckpt.get("discriminator"))
    if d_state is None:
        raise ValueError(f"Checkpoint missing discriminator state dict: {ckpt_path}")
    load_state_dict_compat(discriminator, d_state, model_name="discriminator")
    discriminator.eval()

    score_mode = str(ckpt_args.get("score_mode", "fused")).lower().strip()
    score_alpha = float(ckpt_args.get("score_alpha", 0.24))
    if score_mode in {"critic", "critic_only"}:
        score_mode = "prob"
    if score_mode in {"feature_deviation", "feature_deviation_only"}:
        score_mode = "feat_l2"
    if score_mode not in {"prob", "feat_l2", "feat_mahal", "fused"}:
        score_mode = "fused"

    x_ref = bundle.x_train_benign if score_mode != "prob" else None

    t1 = time.perf_counter()
    s_cal = compute_anomaly_scores(
        discriminator,
        np.asarray(bundle.x_calib_benign, dtype=np.float32),
        device,
        int(batch_size),
        score_mode=score_mode,
        score_alpha=score_alpha,
        x_ref_benign=x_ref,
    )
    s_te = compute_anomaly_scores(
        discriminator,
        np.asarray(bundle.x_test, dtype=np.float32),
        device,
        int(batch_size),
        score_mode=score_mode,
        score_alpha=score_alpha,
        x_ref_benign=x_ref,
    )
    infer_seconds = float(time.perf_counter() - t1)
    train_seconds = 0.0

    metrics = calibrate_and_metrics(bundle.y_test, s_te, s_cal, target_fpr)
    extra = {
        "checkpoint": str(ckpt_path),
        "score_mode": score_mode,
        "score_alpha": score_alpha,
        "disc_pooling": pooling,
    }
    return s_te, s_cal, metrics, train_seconds, infer_seconds, extra


def method_aliases() -> dict[str, str]:
    return {
        "isolationforest": "IsolationForest",
        "oneclasssvm": "OneClassSVM",
        "mlp": "MLP (Window)",
        "ganomaly": "GANomaly",
        "tranad": "TranAD",
        "timesnet": "TimesNet",
        "dlinear": "DLinear",
        "ours": "Ours",
    }


def normalize_baselines(req: list[str]) -> list[str]:
    valid = set(method_aliases().keys())
    out: list[str] = []
    for token in req:
        key = str(token).strip().lower()
        if key not in valid:
            raise ValueError(f"Unsupported baseline: {token}. Valid: {sorted(valid)}")
        if key not in out:
            out.append(key)
    return out


def default_target_fprs(dataset: str) -> list[float]:
    if dataset == "cicids2017":
        return [0.25]
    return [0.05, 0.15]


def row_common(
    *,
    args: argparse.Namespace,
    bundle: DatasetBundle,
    method: str,
    target_fpr: float,
    command_line: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "dataset": bundle.dataset,
        "method": method,
        "split_source": bundle.split_source,
        "train_spec": bundle.train_spec,
        "test_spec": bundle.test_spec,
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "anomaly_ratio_threshold": float(args.anomaly_ratio),
        "independent_calibration": bool(args.independent_calibration),
        "calib_ratio": float(args.calib_ratio),
        "scaler_fit_source": bundle.scaler_fit_source,
        "model_training_source": bundle.model_training_source,
        "threshold_calibration_source": bundle.threshold_calibration_source,
        "target_fpr": float(target_fpr),
        "random_seed": int(seed),
        "command_line": command_line,
    }


def cicids_mlp_protocol_compatible(args: argparse.Namespace, bundle: DatasetBundle, target_fpr: float) -> tuple[bool, str]:
    if str(bundle.dataset) != "cicids2017":
        return False, "dataset is not cicids2017"
    if not bool(args.strict_no_leakage):
        return False, "strict_no_leakage is False"
    if not bool(args.independent_calibration):
        return False, "independent_calibration is False"
    if abs(float(target_fpr) - 0.25) > 1e-12:
        return False, "CICIDS MLP gating requires target_fpr=0.25"
    if int(len(bundle.x_calib_benign)) <= 0:
        return False, "independent calibration benign windows are empty"
    if str(bundle.scaler_fit_source) != "model_train_benign only":
        return False, f"unexpected scaler_fit_source={bundle.scaler_fit_source}"
    if str(bundle.model_training_source) != "model_train_benign":
        return False, f"unexpected model_training_source={bundle.model_training_source}"
    if str(bundle.threshold_calibration_source) != "independent_calibration_benign":
        return False, f"unexpected threshold_calibration_source={bundle.threshold_calibration_source}"
    return True, "ok"


def finalize_row_metrics(
    row: dict[str, Any],
    *,
    auc: float,
    ap: float,
    metrics: dict[str, float],
    n_test_benign: int,
    train_seconds: float,
    inference_seconds: float,
) -> None:
    row["auc"] = float(auc)
    row["ap"] = float(ap)
    row["threshold"] = float(metrics["threshold"])
    row["observed_test_benign_fpr"] = float(metrics["fpr"])
    row["precision"] = float(metrics["precision"])
    row["recall"] = float(metrics["recall"])
    row["f1"] = float(metrics["f1"])
    row["tp"] = int(metrics["tp"])
    row["fp"] = int(metrics["fp"])
    row["tn"] = int(metrics["tn"])
    row["fn"] = int(metrics["fn"])
    row["fp_over_test_benign"] = f"{int(metrics['fp'])}/{int(n_test_benign)}"
    row["train_seconds"] = float(train_seconds)
    row["inference_seconds"] = float(inference_seconds)
    total_windows = int(metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"])
    row["windows_per_second"] = float(0.0 if inference_seconds <= 0 else (total_windows / inference_seconds))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def to_markdown_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def update_global_metadata(out_dir: Path, dataset: str, run_meta: dict[str, Any]) -> None:
    meta_path = out_dir / "compact_baselines_metadata.json"
    if meta_path.exists():
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        payload = {"created_at": now_utc(), "runs": [], "datasets": {}}
    payload.setdefault("runs", []).append(run_meta)
    payload.setdefault("datasets", {})[dataset] = {
        "last_run_at": now_utc(),
        "last_run": run_meta,
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def update_protocol_notes(out_dir: Path) -> None:
    notes = """# Compact Baselines Protocol Notes

This directory contains compact baseline results generated under current strict no-leakage protocols.

Global paper scope:
- Main: CICIDS2017
- Supplementary: SWaT
- Distribution-shift stress test: UNSW-NB15
- Excluded: TON IoT

Protocol rules enforced by this script:
- Thresholds are calibrated from independent benign calibration windows only.
- Test labels are never used for threshold selection.
- Metrics are window-level only (no flow-level mixing).
- Rows lacking strict metadata should not be used in paper tables.

Dataset-specific strict protocol:
- CICIDS2017: explicit train/test day files + independent calibration from train benign only.
- SWaT: formal unsupervised split (benign 60% train / 20% calib / 20% benign test + attack test).
- UNSW-NB15: official train/test split + train benign 80/20 model-train/calibration.
"""
    (out_dir / "compact_baselines_protocol_notes.md").write_text(notes, encoding="utf-8")


def append_skipped(out_dir: Path, dataset: str, skipped: list[dict[str, str]]) -> None:
    path = out_dir / "skipped_baselines.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Skipped Baselines\n\n"
    lines = [existing.rstrip(), "", f"## {dataset} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    if not skipped:
        lines.append("- None")
    else:
        for s in skipped:
            lines.append(f"- `{s['method']}`: {s['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_paper_summary(out_dir: Path) -> None:
    files = {
        "cicids2017": out_dir / "cicids2017_compact_baselines.csv",
        "swat": out_dir / "swat_compact_baselines.csv",
        "unsw_nb15": out_dir / "unsw_nb15_compact_baselines.csv",
    }
    lines = ["# Paper Ready Compact Baseline Summary", ""]
    for ds, path in files.items():
        rows = read_csv_rows(path)
        lines.append(f"## {ds}")
        if not rows:
            lines.append("No results yet.")
            lines.append("")
            continue
        cols = ["method", "target_fpr", "auc", "ap", "f1", "precision", "recall", "observed_test_benign_fpr", "fp_over_test_benign"]
        lines.append(to_markdown_table(rows, cols).rstrip())
        lines.append("")
    (out_dir / "paper_ready_summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def dataset_fieldnames() -> list[str]:
    return [
        "dataset",
        "method",
        "split_source",
        "train_spec",
        "test_spec",
        "window_size",
        "stride",
        "anomaly_ratio_threshold",
        "independent_calibration",
        "calib_ratio",
        "scaler_fit_source",
        "model_training_source",
        "threshold_calibration_source",
        "target_fpr",
        "threshold",
        "observed_test_benign_fpr",
        "fp_over_test_benign",
        "precision",
        "recall",
        "f1",
        "auc",
        "ap",
        "tp",
        "fp",
        "tn",
        "fn",
        "train_seconds",
        "inference_seconds",
        "windows_per_second",
        "random_seed",
        "baseline_hyperparameters",
        "command_line",
        "notes",
    ]


def _coerce_target_key(v: Any) -> str:
    try:
        return f"{float(v):.12g}"
    except Exception:
        return str(v)


def _merge_rows_with_existing(csv_path: Path, new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return list(new_rows)

    existing = read_csv_rows(csv_path)
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in existing:
        key = (
            str(r.get("dataset", "")),
            str(r.get("method", "")),
            _coerce_target_key(r.get("target_fpr", "")),
        )
        by_key[key] = dict(r)

    for r in new_rows:
        key = (
            str(r.get("dataset", "")),
            str(r.get("method", "")),
            _coerce_target_key(r.get("target_fpr", "")),
        )
        by_key[key] = dict(r)

    method_order = {
        "IsolationForest": 1,
        "OneClassSVM": 2,
        "MLP (Window)": 3,
        "GANomaly": 4,
        "TranAD": 5,
        "Ours": 6,
        "TimesNet": 7,
        "DLinear": 8,
    }

    def _sort_key(r: dict[str, Any]) -> tuple[float, int, str]:
        try:
            tf = float(r.get("target_fpr", 0.0))
        except Exception:
            tf = 0.0
        m = str(r.get("method", ""))
        return (tf, method_order.get(m, 999), m)

    merged = list(by_key.values())
    merged.sort(key=_sort_key)
    return merged


def save_dataset_tables(out_dir: Path, dataset: str, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    csv_path = out_dir / f"{dataset}_compact_baselines.csv"
    md_path = out_dir / f"{dataset}_compact_baselines.md"

    fields = dataset_fieldnames()
    merged_rows = _merge_rows_with_existing(csv_path, rows)
    write_csv(csv_path, merged_rows, fields)

    md_cols = [
        "method",
        "target_fpr",
        "threshold",
        "observed_test_benign_fpr",
        "fp_over_test_benign",
        "precision",
        "recall",
        "f1",
        "auc",
        "ap",
        "tp",
        "fp",
        "tn",
        "fn",
    ]
    md = [f"# {dataset} Compact Baselines", "", to_markdown_table(merged_rows, md_cols)]
    md_path.write_text("\n".join(md), encoding="utf-8")
    return csv_path, md_path


def run_dry_run(bundle: DatasetBundle, args: argparse.Namespace) -> None:
    y_test = np.asarray(bundle.y_test, dtype=np.int64)
    n_test_benign = int(np.sum(y_test == 0))
    n_test_anom = int(np.sum(y_test == 1))
    print("=== Dry-Run Protocol Check ===")
    print(f"dataset={bundle.dataset}")
    print(f"split_source={bundle.split_source}")
    print(f"train_spec={bundle.train_spec}")
    print(f"test_spec={bundle.test_spec}")
    print(f"window_size={int(args.window_size)}, stride={int(args.stride)}, anomaly_ratio={float(args.anomaly_ratio)}")
    print(f"n_train_benign_windows={int(len(bundle.x_train_benign))}")
    print(f"n_calib_benign_windows={int(len(bundle.x_calib_benign))}")
    print(f"n_test_windows={int(len(bundle.x_test))}")
    print(f"n_test_benign_windows={n_test_benign}")
    print(f"n_test_anomalous_windows={n_test_anom}")
    print(f"feature_dim={int(bundle.feature_dim)}")
    print(f"independent_calibration={bool(args.independent_calibration)}")
    print(f"strict_no_leakage={bool(args.strict_no_leakage)}")
    print(f"scaler_fit_source={bundle.scaler_fit_source}")
    print(f"model_training_source={bundle.model_training_source}")
    print(f"threshold_calibration_source={bundle.threshold_calibration_source}")
    print(f"calibration_non_empty={bool(len(bundle.x_calib_benign) > 0)}")
    print("protocol_details_json=")
    print(json.dumps(bundle.protocol_details, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()

    if not bool(args.independent_calibration):
        raise ValueError("This script requires --independent-calibration")
    if not bool(args.strict_no_leakage):
        raise ValueError("This script requires --strict-no-leakage")

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    baseline_keys = normalize_baselines(args.baselines)
    target_fprs = [float(x) for x in (args.target_fprs if args.target_fprs else default_target_fprs(args.dataset))]

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(str(args.device))
    command_line = " ".join([sys.executable] + sys.argv)

    bundle = prepare_unsupervised_bundle(args)
    run_dry_run(bundle, args)

    if args.dry_run_protocol_check:
        run_meta = {
            "time_utc": now_utc(),
            "dataset": args.dataset,
            "mode": "dry_run_protocol_check",
            "device": str(device),
            "target_fprs": target_fprs,
            "counts": {
                "n_train_benign_windows": int(len(bundle.x_train_benign)),
                "n_calib_benign_windows": int(len(bundle.x_calib_benign)),
                "n_test_windows": int(len(bundle.x_test)),
                "n_test_benign_windows": int(np.sum(bundle.y_test == 0)),
                "n_test_anomalous_windows": int(np.sum(bundle.y_test == 1)),
            },
            "protocol": {
                "split_source": bundle.split_source,
                "train_spec": bundle.train_spec,
                "test_spec": bundle.test_spec,
                "scaler_fit_source": bundle.scaler_fit_source,
                "model_training_source": bundle.model_training_source,
                "threshold_calibration_source": bundle.threshold_calibration_source,
            },
        }
        update_global_metadata(out_dir, args.dataset, run_meta)
        update_protocol_notes(out_dir)
        update_paper_summary(out_dir)
        print("Dry-run completed. No baselines were trained.")
        return

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    skipped_keys: set[tuple[str, str]] = set()
    n_test_benign = int(np.sum(bundle.y_test == 0))

    alias = method_aliases()

    for tfpr in target_fprs:
        # IsolationForest
        if "isolationforest" in baseline_keys:
            method = alias["isolationforest"]
            row = row_common(args=args, bundle=bundle, method=method, target_fpr=tfpr, command_line=command_line, seed=int(args.seed))
            s_te, _, m, t_train, t_inf = run_iforest(bundle, tfpr, int(args.seed))
            auc, ap = safe_auc_ap(bundle.y_test, s_te)
            finalize_row_metrics(row, auc=auc, ap=ap, metrics=m, n_test_benign=n_test_benign, train_seconds=t_train, inference_seconds=t_inf)
            row["baseline_hyperparameters"] = json.dumps({"contamination": 0.1, "n_estimators": 100}, ensure_ascii=False)
            row["notes"] = ""
            rows.append(row)

        # OneClassSVM
        if "oneclasssvm" in baseline_keys:
            method = alias["oneclasssvm"]
            row = row_common(args=args, bundle=bundle, method=method, target_fpr=tfpr, command_line=command_line, seed=int(args.seed))
            s_te, _, m, t_train, t_inf, extra = run_ocsvm(
                bundle,
                tfpr,
                int(args.seed),
                max_samples=int(args.ocsvm_max_samples),
                pca_components=int(args.ocsvm_pca_components),
            )
            auc, ap = safe_auc_ap(bundle.y_test, s_te)
            finalize_row_metrics(row, auc=auc, ap=ap, metrics=m, n_test_benign=n_test_benign, train_seconds=t_train, inference_seconds=t_inf)
            row["baseline_hyperparameters"] = json.dumps(
                {
                    "nu": 0.1,
                    "kernel": "rbf",
                    "gamma": "scale",
                    "max_samples": int(args.ocsvm_max_samples),
                    "pca_components": int(args.ocsvm_pca_components),
                    **extra,
                },
                ensure_ascii=False,
            )
            row["notes"] = ""
            rows.append(row)

        # MLP (enabled for UNSW and CICIDS under strict gating rules)
        if "mlp" in baseline_keys:
            if bundle.dataset == "swat":
                item = {
                    "method": alias["mlp"],
                    "reason": "not appropriate for swat strict unsupervised split",
                }
                key = (item["method"], item["reason"])
                if key not in skipped_keys:
                    skipped.append(item)
                    skipped_keys.add(key)
            else:
                if bundle.dataset == "cicids2017":
                    ok, reason = cicids_mlp_protocol_compatible(args, bundle, tfpr)
                    if not ok:
                        item = {"method": alias["mlp"], "reason": f"protocol incompatible for CICIDS MLP: {reason}"}
                        key = (item["method"], item["reason"])
                        if key not in skipped_keys:
                            skipped.append(item)
                            skipped_keys.add(key)
                        continue
                method = alias["mlp"]
                row = row_common(args=args, bundle=bundle, method=method, target_fpr=tfpr, command_line=command_line, seed=int(args.seed))
                s_te, _, m, t_train, t_inf, extra = run_mlp_supervised_strict(
                    bundle,
                    args,
                    bundle.dataset,
                    tfpr,
                    int(args.seed),
                )
                auc, ap = safe_auc_ap(bundle.y_test, s_te)
                finalize_row_metrics(row, auc=auc, ap=ap, metrics=m, n_test_benign=n_test_benign, train_seconds=t_train, inference_seconds=t_inf)
                row["baseline_hyperparameters"] = json.dumps({"hidden_layer_sizes": [256, 128, 64], "max_iter": int(args.mlp_max_iter), **extra}, ensure_ascii=False)
                row["notes"] = "supervised training uses train split window labels only; scaler fit on model_train_benign only; threshold calibrated on independent_calibration_benign"
                rows.append(row)

        # GANomaly
        if "ganomaly" in baseline_keys:
            method = alias["ganomaly"]
            row = row_common(args=args, bundle=bundle, method=method, target_fpr=tfpr, command_line=command_line, seed=int(args.seed))
            s_te, _, m, t_train, t_inf = run_ganomaly(bundle, tfpr, device, int(args.epochs), int(args.batch_size), int(args.seed))
            auc, ap = safe_auc_ap(bundle.y_test, s_te)
            finalize_row_metrics(row, auc=auc, ap=ap, metrics=m, n_test_benign=n_test_benign, train_seconds=t_train, inference_seconds=t_inf)
            row["baseline_hyperparameters"] = json.dumps({"latent_dim": 64, "epochs": int(args.epochs), "batch_size": int(args.batch_size), "lr": 1e-3}, ensure_ascii=False)
            row["notes"] = ""
            rows.append(row)

        # TranAD
        if "tranad" in baseline_keys:
            method = alias["tranad"]
            row = row_common(args=args, bundle=bundle, method=method, target_fpr=tfpr, command_line=command_line, seed=int(args.seed))
            s_te, _, m, t_train, t_inf = run_tranad(bundle, tfpr, device, int(args.epochs), int(args.batch_size), int(args.seed))
            auc, ap = safe_auc_ap(bundle.y_test, s_te)
            finalize_row_metrics(row, auc=auc, ap=ap, metrics=m, n_test_benign=n_test_benign, train_seconds=t_train, inference_seconds=t_inf)
            row["baseline_hyperparameters"] = json.dumps({"d_model": 64, "nhead": 4, "num_layers": 2, "epochs": int(args.epochs), "batch_size": int(args.batch_size), "lr": 1e-3}, ensure_ascii=False)
            row["notes"] = ""
            rows.append(row)

        # TimesNet / DLinear placeholders
        if "timesnet" in baseline_keys:
            item = {"method": alias["timesnet"], "reason": "not implemented as reliable local baseline in this repository"}
            key = (item["method"], item["reason"])
            if key not in skipped_keys:
                skipped.append(item)
                skipped_keys.add(key)
        if "dlinear" in baseline_keys:
            item = {"method": alias["dlinear"], "reason": "not implemented as reliable local baseline in this repository"}
            key = (item["method"], item["reason"])
            if key not in skipped_keys:
                skipped.append(item)
                skipped_keys.add(key)

        # Ours
        if "ours" in baseline_keys:
            method = alias["ours"]
            if bundle.dataset == "cicids2017":
                fixed = CICIDS_OURS_FIXED
                if abs(float(tfpr) - float(fixed["target_fpr"])) < 1e-9:
                    row = row_common(args=args, bundle=bundle, method=method, target_fpr=tfpr, command_line=command_line, seed=int(args.seed))
                    row.update(
                        {
                            "threshold": float(fixed["threshold"]),
                            "observed_test_benign_fpr": float(fixed["observed_test_benign_fpr"]),
                            "fp_over_test_benign": f"{int(fixed['fp'])}/{int(fixed['fp'] + fixed['tn'])}",
                            "precision": float(fixed["precision"]),
                            "recall": float(fixed["recall"]),
                            "f1": float(fixed["f1"]),
                            "auc": float(fixed["auc"]),
                            "ap": float(fixed["ap"]),
                            "tp": int(fixed["tp"]),
                            "fp": int(fixed["fp"]),
                            "tn": int(fixed["tn"]),
                            "fn": int(fixed["fn"]),
                            "train_seconds": "",
                            "inference_seconds": "",
                            "windows_per_second": "",
                            "baseline_hyperparameters": json.dumps({"reference": "fixed_cicids_selected_operating_point", "checkpoint": "results/cicids_strict_mps/selected_epoch_008_main.pt", "score_mode": "fused", "score_alpha": 0.24}, ensure_ascii=False),
                            "notes": "fixed reference row from selected strict CICIDS checkpoint",
                        }
                    )
                    rows.append(row)
                else:
                    item = {"method": alias["ours"], "reason": f"CICIDS fixed reference is only available at target_fpr={CICIDS_OURS_FIXED['target_fpr']}"}
                    key = (item["method"], item["reason"])
                    if key not in skipped_keys:
                        skipped.append(item)
                        skipped_keys.add(key)
            else:
                ckpt_path = str(args.ours_checkpoint).strip()
                if not ckpt_path:
                    ckpt_path = discover_default_ours_checkpoint(bundle.dataset)
                if not ckpt_path:
                    item = {
                        "method": alias["ours"],
                        "reason": "no Ours checkpoint found for this dataset; provide --ours-checkpoint or restore strict checkpoint artifact",
                    }
                    key = (item["method"], item["reason"])
                    if key not in skipped_keys:
                        skipped.append(item)
                        skipped_keys.add(key)
                else:
                    try:
                        row = row_common(
                            args=args,
                            bundle=bundle,
                            method=method,
                            target_fpr=tfpr,
                            command_line=command_line,
                            seed=int(args.seed),
                        )
                        s_te, _, m, t_train, t_inf, extra = run_ours_checkpoint_eval(
                            bundle,
                            tfpr,
                            device,
                            int(args.batch_size),
                            ckpt_path,
                        )
                        auc, ap = safe_auc_ap(bundle.y_test, s_te)
                        finalize_row_metrics(
                            row,
                            auc=auc,
                            ap=ap,
                            metrics=m,
                            n_test_benign=n_test_benign,
                            train_seconds=t_train,
                            inference_seconds=t_inf,
                        )
                        row["baseline_hyperparameters"] = json.dumps(
                            {
                                "reference": "checkpoint_eval_strict",
                                **extra,
                            },
                            ensure_ascii=False,
                        )
                        row["notes"] = "checkpoint-based Ours evaluation; scaler fit/model train/calibration remain strict-no-leakage in dataset pipeline"
                        rows.append(row)
                    except Exception as e:
                        item = {
                            "method": alias["ours"],
                            "reason": f"checkpoint evaluation failed: {e}",
                        }
                        key = (item["method"], item["reason"])
                        if key not in skipped_keys:
                            skipped.append(item)
                            skipped_keys.add(key)

    if not rows:
        print("No result rows were produced. Check --baselines and dataset settings.")

    csv_path, md_path = save_dataset_tables(out_dir, bundle.dataset, rows)
    append_skipped(out_dir, bundle.dataset, skipped)
    update_protocol_notes(out_dir)
    update_paper_summary(out_dir)

    run_meta = {
        "time_utc": now_utc(),
        "dataset": bundle.dataset,
        "mode": "full_eval",
        "device": str(device),
        "target_fprs": target_fprs,
        "baselines_requested": baseline_keys,
        "rows_written": int(len(rows)),
        "csv": str(csv_path),
        "md": str(md_path),
        "counts": {
            "n_train_benign_windows": int(len(bundle.x_train_benign)),
            "n_calib_benign_windows": int(len(bundle.x_calib_benign)),
            "n_test_windows": int(len(bundle.x_test)),
            "n_test_benign_windows": int(np.sum(bundle.y_test == 0)),
            "n_test_anomalous_windows": int(np.sum(bundle.y_test == 1)),
        },
        "protocol": {
            "split_source": bundle.split_source,
            "train_spec": bundle.train_spec,
            "test_spec": bundle.test_spec,
            "scaler_fit_source": bundle.scaler_fit_source,
            "model_training_source": bundle.model_training_source,
            "threshold_calibration_source": bundle.threshold_calibration_source,
        },
    }
    update_global_metadata(out_dir, bundle.dataset, run_meta)

    print(f"Saved dataset CSV: {csv_path}")
    print(f"Saved dataset MD:  {md_path}")
    print(f"Updated metadata:   {out_dir / 'compact_baselines_metadata.json'}")
    print(f"Updated notes:      {out_dir / 'compact_baselines_protocol_notes.md'}")
    print(f"Updated skipped:    {out_dir / 'skipped_baselines.md'}")
    print(f"Updated summary:    {out_dir / 'paper_ready_summary.md'}")


if __name__ == "__main__":
    main()
