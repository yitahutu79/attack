#!/usr/bin/env python3
"""Shared dataset loaders for window-level anomaly detection experiments."""

from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


BENIGN_TOKENS = {
    "0",
    "benign",
    "normal",
    "no",
    "false",
}

DEFAULT_TRAIN_TEST: dict[str, tuple[list[str], list[str]]] = {
    "swat": (
        ["benign_data/benign_samples_5sec.csv"],
        ["attack_data/attack_samples_5sec.csv"],
    ),
}

DEFAULT_MIXED_FILES: dict[str, list[str]] = {
    "swat": [
        "benign_data/benign_samples_5sec.csv",
        "attack_data/attack_samples_5sec.csv",
    ],
}

DEFAULT_UNSUPERVISED_SPLIT: dict[str, tuple[str, str]] = {
    "swat": (
        "benign_data/benign_samples_5sec.csv",
        "attack_data/attack_samples_5sec.csv",
    ),
}

DEFAULT_CHRONO_UNSUP_FILE: dict[str, str] = {
    "ton_iot": "Processed_datasets/Processed_Linux_dataset/linux_memory1.csv",
}

NON_FEATURE_COLS = {
    "label",
    "label_full",
    "label1",
    "label2",
    "label3",
    "label4",
    "id",
    "type",
    "class",
    "attack",
    "attack_cat",
    "flow id",
    "source ip",
    "destination ip",
    "src ip",
    "dst ip",
    "timestamp",
    "timestamp_start",
    "timestamp_end",
    "date",
    "time",
    "device_name",
    "device_mac",
}


@dataclass
class WindowedDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    train_files: list[str]
    test_files: list[str]
    dataset: str
    scaler: str
    x_calib: np.ndarray | None = None
    y_calib: np.ndarray | None = None
    calib_files: list[str] = field(default_factory=list)


def normalize_dataset_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    aliases = {
        "cicids": "cicids2017",
        "cicids_2017": "cicids2017",
        "cic_ids_2017": "cicids2017",
        "unsw": "unsw_nb15",
        "unsw15": "unsw_nb15",
        "unsw_nb15": "unsw_nb15",
        "ton": "ton_iot",
        "toniot": "ton_iot",
        "ton_lot": "ton_iot",
        "ton_iot": "ton_iot",
        "swat": "swat",
        "generic": "generic",
    }
    return aliases.get(key, key)


def list_dataset_files(data_dir: str, dataset: str = "generic") -> list[str]:
    """Return CSV files relative to data_dir."""
    dataset = normalize_dataset_name(dataset)
    root = Path(data_dir)
    if dataset == "swat":
        patterns = ["benign_data/*.csv", "attack_data/*.csv"]
    elif dataset == "ton_iot":
        patterns = [
            "Processed_datasets/Processed_IoT_dataset/*.csv",
            "Processed_datasets/Processed_Linux_dataset/*.csv",
            "__Train_Test_datasets/**/*.csv",
        ]
    elif dataset == "unsw_nb15":
        patterns = ["Training and Testing Sets/*.csv", "*.csv"]
    else:
        patterns = ["*.csv", "**/*.csv"]

    out: list[str] = []
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                out.append(str(path.relative_to(root)))
    return sorted(dict.fromkeys(out))


def _resolve_files(data_dir: str, files: Iterable[str]) -> list[Path]:
    root = Path(data_dir)
    out: list[Path] = []
    for item in files:
        token = str(item).strip()
        if not token:
            continue
        candidates = [Path(p) for p in glob.glob(token)]
        if not candidates:
            candidates = [Path(p) for p in glob.glob(str(root / token))]
        if not candidates:
            p = Path(token)
            candidates = [p if p.is_absolute() else root / p]
        for path in candidates:
            if not path.exists():
                raise FileNotFoundError(f"CSV 文件不存在：{path}")
            out.append(path)
    return out


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True, low_memory=False)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df.replace([np.inf, -np.inf], np.nan).dropna(axis=0).copy()


def _candidate_label_cols(df: pd.DataFrame, dataset: str) -> list[str]:
    cols = list(df.columns)
    by_lower = {str(c).strip().lower(): c for c in cols}
    preferred: list[str] = []
    if dataset == "swat":
        preferred = ["label1", "label", "type"]
    elif dataset == "ton_iot":
        preferred = ["label", "type"]
    elif dataset == "cicids2017":
        preferred = ["label"]
    else:
        preferred = ["label", "label1", "type", "class", "attack"]
    return [by_lower[p] for p in preferred if p in by_lower]


def _labels_from_frame(df: pd.DataFrame, dataset: str) -> np.ndarray:
    candidates = _candidate_label_cols(df, dataset)
    if not candidates:
        raise ValueError(f"找不到标签列，可用列包括：{list(df.columns)[:20]}")
    col = candidates[0]
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return (pd.to_numeric(s, errors="coerce").fillna(0).to_numpy() != 0).astype(np.uint8)

    values = s.astype(str).str.strip().str.lower()
    return (~values.isin(BENIGN_TOKENS)).astype(np.uint8).to_numpy()


def _numeric_features(df: pd.DataFrame, labels: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    drop_cols = [c for c in df.columns if str(c).strip().lower() in NON_FEATURE_COLS]
    numeric = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError("未找到数值型特征列，无法窗口化")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    if len(numeric) != len(labels):
        # Re-align after dropna by using the surviving index.
        labels = labels[numeric.index.to_numpy()]
        numeric = numeric.reset_index(drop=True)
    return numeric, labels


def _frame_to_numeric_and_labels(df: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, np.ndarray]:
    labels = _labels_from_frame(df, dataset)
    numeric, labels = _numeric_features(df, labels)
    if len(numeric) != len(labels):
        raise ValueError("特征和标签长度不一致")
    return numeric, labels


def build_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int,
    anomaly_ratio: float,
    *,
    progress: bool = False,
    desc: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    total = len(features) - int(window_size) + 1
    if total <= 0:
        raise ValueError(f"window_size={window_size} 大于可用样本数 {len(features)}")

    starts: Iterable[int] = range(0, total, int(stride))
    if progress and tqdm is not None:
        starts = tqdm(
            starts,
            total=int((total + int(stride) - 1) // int(stride)),
            desc=desc or "build windows",
            leave=False,
            file=sys.stderr,
        )

    xs: list[np.ndarray] = []
    ys: list[int] = []
    for start in starts:
        end = start + int(window_size)
        xs.append(features[start:end])
        ys.append(int(labels[start:end].mean() >= float(anomaly_ratio)))
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.uint8)


def _fit_scaler(
    frames: list[pd.DataFrame],
    labels: list[np.ndarray],
    *,
    benign_only: bool,
    scaler: str,
) -> MinMaxScaler | StandardScaler:
    parts: list[pd.DataFrame] = []
    for frame, y in zip(frames, labels, strict=True):
        if benign_only and np.any(y == 0):
            parts.append(frame.loc[y == 0])
        else:
            parts.append(frame)
    if not parts:
        raise ValueError("没有可用于拟合 scaler 的训练样本")
    if str(scaler).lower().strip() == "standard":
        obj: MinMaxScaler | StandardScaler = StandardScaler()
    else:
        obj = MinMaxScaler()
    obj.fit(pd.concat(parts, ignore_index=True).astype(np.float32))
    return obj


def load_windowed_dataset(
    *,
    dataset: str,
    data_dir: str,
    train_files: Iterable[str] | None,
    test_files: Iterable[str] | None,
    window_size: int,
    stride: int,
    anomaly_ratio: float,
    train_benign_only: bool = True,
    scaler: str = "minmax",
    clip_minmax: bool = True,
    progress: bool = False,
) -> WindowedDataset:
    """Load train/test CSV files and return same-granularity windows.

    Window construction is per file, so windows never cross file/day/device boundaries.
    """
    dataset = normalize_dataset_name(dataset)
    train_list = list(train_files or [])
    test_list = list(test_files or [])
    if not train_list and not test_list and dataset in DEFAULT_TRAIN_TEST:
        train_list, test_list = DEFAULT_TRAIN_TEST[dataset]
    if not train_list or not test_list:
        raise ValueError(
            "请提供 --train-files 和 --test-files；SWaT 可省略，会默认使用 5sec benign/attack CSV"
        )

    train_paths = _resolve_files(data_dir, train_list)
    test_paths = _resolve_files(data_dir, test_list)

    train_frames: list[pd.DataFrame] = []
    train_labels: list[np.ndarray] = []
    test_frames: list[pd.DataFrame] = []
    test_labels: list[np.ndarray] = []

    for path in train_paths:
        numeric, y = _frame_to_numeric_and_labels(_read_csv(path), dataset)
        train_frames.append(numeric)
        train_labels.append(y)
    for path in test_paths:
        numeric, y = _frame_to_numeric_and_labels(_read_csv(path), dataset)
        test_frames.append(numeric)
        test_labels.append(y)

    common_cols = set(train_frames[0].columns)
    for frame in train_frames[1:] + test_frames:
        common_cols &= set(frame.columns)
    feature_names = [c for c in train_frames[0].columns if c in common_cols]
    if not feature_names:
        raise ValueError("训练/测试文件之间没有共同数值特征列")

    train_frames = [f.loc[:, feature_names] for f in train_frames]
    test_frames = [f.loc[:, feature_names] for f in test_frames]
    scaler_obj = _fit_scaler(train_frames, train_labels, benign_only=True, scaler=scaler)

    def transform(frame: pd.DataFrame) -> np.ndarray:
        x = scaler_obj.transform(frame.astype(np.float32)).astype(np.float32)
        if clip_minmax and str(scaler).lower().strip() == "minmax":
            x = np.clip(x, 0.0, 1.0)
        return x

    train_xs: list[np.ndarray] = []
    train_ys: list[np.ndarray] = []
    for idx, (frame, y) in enumerate(zip(train_frames, train_labels, strict=True), start=1):
        xw, yw = build_sequences(
            transform(frame),
            y,
            window_size,
            stride,
            anomaly_ratio,
            progress=progress,
            desc=f"train windows {idx}/{len(train_frames)}",
        )
        if train_benign_only:
            keep = yw == 0
            xw = xw[keep]
            yw = yw[keep]
        if len(xw):
            train_xs.append(xw)
            train_ys.append(yw)
    if not train_xs:
        raise ValueError("训练窗口为空，请检查标签、窗口大小或 train_benign_only 设置")

    test_xs: list[np.ndarray] = []
    test_ys: list[np.ndarray] = []
    for idx, (frame, y) in enumerate(zip(test_frames, test_labels, strict=True), start=1):
        xw, yw = build_sequences(
            transform(frame),
            y,
            window_size,
            stride,
            anomaly_ratio,
            progress=progress,
            desc=f"test windows {idx}/{len(test_frames)}",
        )
        test_xs.append(xw)
        test_ys.append(yw)

    return WindowedDataset(
        x_train=np.concatenate(train_xs, axis=0),
        y_train=np.concatenate(train_ys, axis=0),
        x_test=np.concatenate(test_xs, axis=0),
        y_test=np.concatenate(test_ys, axis=0),
        x_calib=None,
        y_calib=None,
        feature_names=[str(c) for c in feature_names],
        train_files=[str(p.relative_to(Path(data_dir))) if Path(data_dir) in p.parents else str(p) for p in train_paths],
        test_files=[str(p.relative_to(Path(data_dir))) if Path(data_dir) in p.parents else str(p) for p in test_paths],
        calib_files=[],
        dataset=dataset,
        scaler=str(scaler),
    )


def load_windowed_mixed_split(
    *,
    dataset: str,
    data_dir: str,
    files: Iterable[str] | None,
    window_size: int,
    stride: int,
    anomaly_ratio: float,
    train_fraction: float = 0.5,
    scaler: str = "standard",
    clip_minmax: bool = False,
    progress: bool = False,
) -> WindowedDataset:
    """Build a supervised train/test split by splitting each file chronologically.

    This is useful for datasets such as SWaT where the natural unsupervised setup is
    "benign file for train, attack file for test"; supervised baselines need both
    classes in the training split to be meaningful.
    """
    dataset = normalize_dataset_name(dataset)
    file_list = list(files or [])
    if not file_list and dataset in DEFAULT_MIXED_FILES:
        file_list = DEFAULT_MIXED_FILES[dataset]
    if not file_list:
        raise ValueError("mixed split 需要提供 --mixed-files，或使用带默认 mixed files 的数据集")
    frac = float(train_fraction)
    if not (0.0 < frac < 1.0):
        raise ValueError("train_fraction 必须在 (0,1) 之间")

    paths = _resolve_files(data_dir, file_list)
    frames: list[pd.DataFrame] = []
    labels_list: list[np.ndarray] = []
    for path in paths:
        numeric, y = _frame_to_numeric_and_labels(_read_csv(path), dataset)
        frames.append(numeric)
        labels_list.append(y)

    common_cols = set(frames[0].columns)
    for frame in frames[1:]:
        common_cols &= set(frame.columns)
    feature_names = [c for c in frames[0].columns if c in common_cols]
    if not feature_names:
        raise ValueError("mixed split 文件之间没有共同数值特征列")
    frames = [f.loc[:, feature_names].reset_index(drop=True) for f in frames]

    train_frames: list[pd.DataFrame] = []
    train_labels: list[np.ndarray] = []
    test_frames: list[pd.DataFrame] = []
    test_labels: list[np.ndarray] = []
    for frame, y in zip(frames, labels_list, strict=True):
        n = len(frame)
        cut = int(n * frac)
        cut = max(cut, int(window_size))
        cut = min(cut, n - int(window_size))
        if cut <= 0 or cut >= n:
            raise ValueError(f"文件样本数 {n} 无法按 train_fraction={frac} 切出可滑窗的 train/test")
        train_frames.append(frame.iloc[:cut].reset_index(drop=True))
        train_labels.append(y[:cut])
        test_frames.append(frame.iloc[cut:].reset_index(drop=True))
        test_labels.append(y[cut:])

    scaler_obj = _fit_scaler(train_frames, train_labels, benign_only=False, scaler=scaler)

    def transform(frame: pd.DataFrame) -> np.ndarray:
        x = scaler_obj.transform(frame.astype(np.float32)).astype(np.float32)
        if clip_minmax and str(scaler).lower().strip() == "minmax":
            x = np.clip(x, 0.0, 1.0)
        return x

    train_xs: list[np.ndarray] = []
    train_ys: list[np.ndarray] = []
    test_xs: list[np.ndarray] = []
    test_ys: list[np.ndarray] = []
    for idx, (frame, y) in enumerate(zip(train_frames, train_labels, strict=True), start=1):
        xw, yw = build_sequences(
            transform(frame),
            y,
            window_size,
            stride,
            anomaly_ratio,
            progress=progress,
            desc=f"mixed train windows {idx}/{len(train_frames)}",
        )
        train_xs.append(xw)
        train_ys.append(yw)
    for idx, (frame, y) in enumerate(zip(test_frames, test_labels, strict=True), start=1):
        xw, yw = build_sequences(
            transform(frame),
            y,
            window_size,
            stride,
            anomaly_ratio,
            progress=progress,
            desc=f"mixed test windows {idx}/{len(test_frames)}",
        )
        test_xs.append(xw)
        test_ys.append(yw)

    return WindowedDataset(
        x_train=np.concatenate(train_xs, axis=0),
        y_train=np.concatenate(train_ys, axis=0),
        x_test=np.concatenate(test_xs, axis=0),
        y_test=np.concatenate(test_ys, axis=0),
        x_calib=None,
        y_calib=None,
        feature_names=[str(c) for c in feature_names],
        train_files=[f"{str(p.relative_to(Path(data_dir))) if Path(data_dir) in p.parents else str(p)}[:{frac:.2f}]" for p in paths],
        test_files=[f"{str(p.relative_to(Path(data_dir))) if Path(data_dir) in p.parents else str(p)}[{frac:.2f}:]" for p in paths],
        calib_files=[],
        dataset=dataset,
        scaler=str(scaler),
    )


def load_windowed_unsupervised_split(
    *,
    dataset: str,
    data_dir: str,
    benign_file: str | None,
    attack_file: str | None,
    window_size: int,
    stride: int,
    anomaly_ratio: float,
    benign_train_fraction: float = 0.6,
    benign_calib_fraction: float = 0.2,
    scaler: str = "minmax",
    clip_minmax: bool = True,
    progress: bool = False,
) -> WindowedDataset:
    """Build an anomaly-detection split with separate benign train/calibration/test windows.

    Train: benign-only windows from the first benign segment.
    Calibration: benign-only windows from the middle benign segment.
    Test: benign windows from the final benign segment + all attack windows.
    """
    dataset = normalize_dataset_name(dataset)
    if dataset in DEFAULT_UNSUPERVISED_SPLIT:
        default_benign, default_attack = DEFAULT_UNSUPERVISED_SPLIT[dataset]
        benign_file = benign_file or default_benign
        attack_file = attack_file or default_attack
    if not benign_file or not attack_file:
        raise ValueError("unsupervised split 需要 benign_file 和 attack_file，或使用带默认设置的数据集")

    train_frac = float(benign_train_fraction)
    calib_frac = float(benign_calib_fraction)
    if not (0.0 < train_frac < 1.0) or not (0.0 < calib_frac < 1.0):
        raise ValueError("benign_train_fraction 和 benign_calib_fraction 必须在 (0,1) 内")
    if train_frac + calib_frac >= 1.0:
        raise ValueError("benign_train_fraction + benign_calib_fraction 必须小于 1")

    benign_path = _resolve_files(data_dir, [benign_file])[0]
    attack_path = _resolve_files(data_dir, [attack_file])[0]

    benign_numeric, benign_y = _frame_to_numeric_and_labels(_read_csv(benign_path), dataset)
    attack_numeric, attack_y = _frame_to_numeric_and_labels(_read_csv(attack_path), dataset)

    common_cols = [c for c in benign_numeric.columns if c in set(attack_numeric.columns)]
    if not common_cols:
        raise ValueError("benign/attack 文件之间没有共同数值特征列")
    benign_numeric = benign_numeric.loc[:, common_cols].reset_index(drop=True)
    attack_numeric = attack_numeric.loc[:, common_cols].reset_index(drop=True)

    n_benign = len(benign_numeric)
    train_end = int(n_benign * train_frac)
    calib_end = int(n_benign * (train_frac + calib_frac))
    min_rows = int(window_size)
    train_end = max(train_end, min_rows)
    calib_end = max(calib_end, train_end + min_rows)
    calib_end = min(calib_end, n_benign - min_rows)
    if train_end <= 0 or calib_end <= train_end or calib_end >= n_benign:
        raise ValueError("benign 文件长度不足，无法切出 train/calib/test 三段可滑窗区间")

    benign_train_frame = benign_numeric.iloc[:train_end].reset_index(drop=True)
    benign_train_y = benign_y[:train_end]
    benign_calib_frame = benign_numeric.iloc[train_end:calib_end].reset_index(drop=True)
    benign_calib_y = benign_y[train_end:calib_end]
    benign_test_frame = benign_numeric.iloc[calib_end:].reset_index(drop=True)
    benign_test_y = benign_y[calib_end:]

    scaler_obj = _fit_scaler([benign_train_frame], [benign_train_y], benign_only=True, scaler=scaler)

    def transform(frame: pd.DataFrame) -> np.ndarray:
        x = scaler_obj.transform(frame.astype(np.float32)).astype(np.float32)
        if clip_minmax and str(scaler).lower().strip() == "minmax":
            x = np.clip(x, 0.0, 1.0)
        return x

    x_train, y_train = build_sequences(
        transform(benign_train_frame),
        benign_train_y,
        window_size,
        stride,
        anomaly_ratio,
        progress=progress,
        desc="unsup train benign windows",
    )
    keep_train = y_train == 0
    x_train = x_train[keep_train]
    y_train = y_train[keep_train]

    x_calib, y_calib = build_sequences(
        transform(benign_calib_frame),
        benign_calib_y,
        window_size,
        stride,
        anomaly_ratio,
        progress=progress,
        desc="unsup calib benign windows",
    )
    keep_calib = y_calib == 0
    x_calib = x_calib[keep_calib]
    y_calib = y_calib[keep_calib]

    x_test_benign, y_test_benign = build_sequences(
        transform(benign_test_frame),
        benign_test_y,
        window_size,
        stride,
        anomaly_ratio,
        progress=progress,
        desc="unsup test benign windows",
    )
    x_test_attack, y_test_attack = build_sequences(
        transform(attack_numeric),
        attack_y,
        window_size,
        stride,
        anomaly_ratio,
        progress=progress,
        desc="unsup test attack windows",
    )

    x_test = np.concatenate([x_test_benign, x_test_attack], axis=0)
    y_test = np.concatenate([y_test_benign, y_test_attack], axis=0)

    return WindowedDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_calib=x_calib,
        y_calib=y_calib,
        feature_names=[str(c) for c in common_cols],
        train_files=[f"{str(benign_path.relative_to(Path(data_dir))) if Path(data_dir) in benign_path.parents else str(benign_path)}[:{train_frac:.2f}]"],
        test_files=[
            f"{str(benign_path.relative_to(Path(data_dir))) if Path(data_dir) in benign_path.parents else str(benign_path)}[{train_frac + calib_frac:.2f}:]",
            str(attack_path.relative_to(Path(data_dir))) if Path(data_dir) in attack_path.parents else str(attack_path),
        ],
        calib_files=[f"{str(benign_path.relative_to(Path(data_dir))) if Path(data_dir) in benign_path.parents else str(benign_path)}[{train_frac:.2f}:{train_frac + calib_frac:.2f}]"],
        dataset=dataset,
        scaler=str(scaler),
    )


def load_windowed_chrono_unsupervised_split(
    *,
    dataset: str,
    data_dir: str,
    mixed_file: str | None,
    window_size: int,
    stride: int,
    anomaly_ratio: float,
    train_fraction: float = 0.6,
    calib_fraction: float = 0.1,
    scaler: str = "minmax",
    clip_minmax: bool = True,
    progress: bool = False,
) -> WindowedDataset:
    """Single-file chronological split for anomaly detection.

    Useful for datasets like TON_IoT where one device file contains a benign prefix
    followed by mixed benign/attack traffic later in time.
    """
    dataset = normalize_dataset_name(dataset)
    if dataset in DEFAULT_CHRONO_UNSUP_FILE:
        mixed_file = mixed_file or DEFAULT_CHRONO_UNSUP_FILE[dataset]
    if not mixed_file:
        raise ValueError("chrono unsupervised split 需要 mixed_file，或使用带默认文件的数据集")

    train_frac = float(train_fraction)
    calib_frac = float(calib_fraction)
    if not (0.0 < train_frac < 1.0) or not (0.0 < calib_frac < 1.0):
        raise ValueError("train_fraction 和 calib_fraction 必须在 (0,1) 内")
    if train_frac + calib_frac >= 1.0:
        raise ValueError("train_fraction + calib_fraction 必须小于 1")

    mixed_path = _resolve_files(data_dir, [mixed_file])[0]
    numeric, y = _frame_to_numeric_and_labels(_read_csv(mixed_path), dataset)
    numeric = numeric.reset_index(drop=True)
    n = len(numeric)
    train_end = int(n * train_frac)
    calib_end = int(n * (train_frac + calib_frac))
    min_rows = int(window_size)
    train_end = max(train_end, min_rows)
    calib_end = max(calib_end, train_end + min_rows)
    calib_end = min(calib_end, n - min_rows)
    if train_end <= 0 or calib_end <= train_end or calib_end >= n:
        raise ValueError("mixed 文件长度不足，无法切出 train/calib/test 三段可滑窗区间")

    train_frame = numeric.iloc[:train_end].reset_index(drop=True)
    train_y = y[:train_end]
    calib_frame = numeric.iloc[train_end:calib_end].reset_index(drop=True)
    calib_y = y[train_end:calib_end]
    test_frame = numeric.iloc[calib_end:].reset_index(drop=True)
    test_y = y[calib_end:]

    scaler_obj = _fit_scaler([train_frame], [train_y], benign_only=True, scaler=scaler)

    def transform(frame: pd.DataFrame) -> np.ndarray:
        x = scaler_obj.transform(frame.astype(np.float32)).astype(np.float32)
        if clip_minmax and str(scaler).lower().strip() == "minmax":
            x = np.clip(x, 0.0, 1.0)
        return x

    x_train, y_train = build_sequences(
        transform(train_frame),
        train_y,
        window_size,
        stride,
        anomaly_ratio,
        progress=progress,
        desc="chrono train windows",
    )
    keep_train = y_train == 0
    x_train = x_train[keep_train]
    y_train = y_train[keep_train]

    x_calib, y_calib = build_sequences(
        transform(calib_frame),
        calib_y,
        window_size,
        stride,
        anomaly_ratio,
        progress=progress,
        desc="chrono calib windows",
    )
    keep_calib = y_calib == 0
    x_calib = x_calib[keep_calib]
    y_calib = y_calib[keep_calib]

    x_test, y_test = build_sequences(
        transform(test_frame),
        test_y,
        window_size,
        stride,
        anomaly_ratio,
        progress=progress,
        desc="chrono test windows",
    )

    return WindowedDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_calib=x_calib,
        y_calib=y_calib,
        feature_names=[str(c) for c in numeric.columns],
        train_files=[f"{str(mixed_path.relative_to(Path(data_dir))) if Path(data_dir) in mixed_path.parents else str(mixed_path)}[:{train_frac:.2f}]"],
        test_files=[f"{str(mixed_path.relative_to(Path(data_dir))) if Path(data_dir) in mixed_path.parents else str(mixed_path)}[{train_frac + calib_frac:.2f}:]"],
        calib_files=[f"{str(mixed_path.relative_to(Path(data_dir))) if Path(data_dir) in mixed_path.parents else str(mixed_path)}[{train_frac:.2f}:{train_frac + calib_frac:.2f}]"],
        dataset=dataset,
        scaler=str(scaler),
    )
