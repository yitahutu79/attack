from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from journal_rebuild.src.utils.hashing import sha256_file, sha256_json, sha256_pickle


BENIGN_TOKENS = {"benign", "normal", "0", "false", "no"}
NON_FEATURE_COLS = {
    "label",
    "timestamp",
    "time",
    "date",
    "source ip",
    "destination ip",
    "src ip",
    "dst ip",
    "flow id",
}


@dataclass
class PreparedFrame:
    rel_path: str
    source_day: str
    numeric: pd.DataFrame
    labels: np.ndarray
    attack_types: np.ndarray
    row_indices: np.ndarray
    timestamp_values: np.ndarray | None


@dataclass
class SplitSegment:
    split_role: str
    source_file: str
    source_day: str
    numeric: pd.DataFrame
    labels: np.ndarray
    attack_types: np.ndarray
    row_indices: np.ndarray


@dataclass
class SplitArrays:
    features: np.ndarray
    labels: np.ndarray
    attack_code: np.ndarray
    source_file_code: np.ndarray
    row_index: np.ndarray


def source_day_from_rel_path(rel_path: str) -> str:
    token = Path(rel_path).name.lower()
    for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"):
        if day in token:
            return day.capitalize()
    return Path(rel_path).stem


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True, low_memory=False)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df.replace([np.inf, -np.inf], np.nan).dropna(axis=0).copy()


def _label_column(df: pd.DataFrame) -> str:
    for column in df.columns:
        if str(column).strip().lower() == "label":
            return str(column)
    raise ValueError("CICIDS2017 CSV does not contain a Label column")


def _timestamp_column(df: pd.DataFrame) -> str | None:
    by_lower = {str(c).strip().lower(): c for c in df.columns}
    for key in ("timestamp", "time", "date", "flow start time", "start time"):
        if key in by_lower:
            return str(by_lower[key])
    return None


def _labels_from_series(series: pd.Series) -> np.ndarray:
    values = series.astype(str).str.strip()
    return (~values.str.lower().isin(BENIGN_TOKENS)).astype(np.uint8).to_numpy()


def _prepare_frame(data_root: Path, rel_path: str) -> PreparedFrame:
    path = data_root / rel_path
    df = _read_csv(path)
    label_col = _label_column(df)
    ts_col = _timestamp_column(df)
    labels = pd.Series(_labels_from_series(df[label_col]), index=df.index)
    attack_types = df[label_col].astype(str).str.strip()
    timestamp_values = None if ts_col is None else df[ts_col].astype(str)

    drop_cols = [c for c in df.columns if str(c).strip().lower() in NON_FEATURE_COLS]
    numeric = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError(f"No numeric features available in {path}")
    keep_idx = numeric.index.to_numpy(dtype=np.int64)
    numeric = numeric.reset_index(drop=True)
    return PreparedFrame(
        rel_path=rel_path,
        source_day=source_day_from_rel_path(rel_path),
        numeric=numeric,
        labels=np.asarray(labels.loc[keep_idx].to_numpy(), dtype=np.uint8),
        attack_types=np.asarray(attack_types.loc[keep_idx].to_numpy(), dtype=object),
        row_indices=keep_idx,
        timestamp_values=None if timestamp_values is None else np.asarray(timestamp_values.loc[keep_idx].to_numpy(), dtype=object),
    )


def _align_common_features(frames: list[PreparedFrame]) -> list[str]:
    common = set(frames[0].numeric.columns)
    for frame in frames[1:]:
        common &= set(frame.numeric.columns)
    feature_names = [str(c) for c in frames[0].numeric.columns if c in common]
    if not feature_names:
        raise ValueError("No common numeric feature columns across selected CICIDS2017 files")
    for frame in frames:
        frame.numeric = frame.numeric.loc[:, feature_names].reset_index(drop=True)
    return feature_names


def _split_train_calibration_segments(frames: list[PreparedFrame], calibration_ratio: float) -> tuple[list[SplitSegment], list[SplitSegment]]:
    benign_frames: list[SplitSegment] = []
    total_benign = 0
    for frame in frames:
        mask = frame.labels == 0
        if not np.any(mask):
            continue
        seg = SplitSegment(
            split_role="pending",
            source_file=frame.rel_path,
            source_day=frame.source_day,
            numeric=frame.numeric.loc[mask].reset_index(drop=True),
            labels=np.zeros(int(np.sum(mask)), dtype=np.uint8),
            attack_types=np.asarray(["BENIGN"] * int(np.sum(mask)), dtype=object),
            row_indices=np.asarray(frame.row_indices[mask], dtype=np.int64),
        )
        benign_frames.append(seg)
        total_benign += len(seg.numeric)
    if total_benign < 2:
        raise ValueError("Not enough benign training-day records to build train/calibration split")

    calibration_n = max(1, min(total_benign - 1, int(total_benign * float(calibration_ratio))))
    train_n = total_benign - calibration_n
    train_segments: list[SplitSegment] = []
    calib_segments: list[SplitSegment] = []
    remain_train = train_n
    for seg in benign_frames:
        seg_len = len(seg.numeric)
        if remain_train <= 0:
            calib_segments.append(
                SplitSegment(
                    split_role="independent_calibration_benign",
                    source_file=seg.source_file,
                    source_day=seg.source_day,
                    numeric=seg.numeric.reset_index(drop=True),
                    labels=seg.labels.copy(),
                    attack_types=seg.attack_types.copy(),
                    row_indices=seg.row_indices.copy(),
                )
            )
            continue
        if remain_train >= seg_len:
            train_segments.append(
                SplitSegment(
                    split_role="model_train_benign",
                    source_file=seg.source_file,
                    source_day=seg.source_day,
                    numeric=seg.numeric.reset_index(drop=True),
                    labels=seg.labels.copy(),
                    attack_types=seg.attack_types.copy(),
                    row_indices=seg.row_indices.copy(),
                )
            )
            remain_train -= seg_len
            continue
        train_segments.append(
            SplitSegment(
                split_role="model_train_benign",
                source_file=seg.source_file,
                source_day=seg.source_day,
                numeric=seg.numeric.iloc[:remain_train].reset_index(drop=True),
                labels=seg.labels[:remain_train].copy(),
                attack_types=seg.attack_types[:remain_train].copy(),
                row_indices=seg.row_indices[:remain_train].copy(),
            )
        )
        calib_segments.append(
            SplitSegment(
                split_role="independent_calibration_benign",
                source_file=seg.source_file,
                source_day=seg.source_day,
                numeric=seg.numeric.iloc[remain_train:].reset_index(drop=True),
                labels=seg.labels[remain_train:].copy(),
                attack_types=seg.attack_types[remain_train:].copy(),
                row_indices=seg.row_indices[remain_train:].copy(),
            )
        )
        remain_train = 0
    return train_segments, calib_segments


def _build_test_segments(frames: list[PreparedFrame]) -> list[SplitSegment]:
    out: list[SplitSegment] = []
    for frame in frames:
        out.append(
            SplitSegment(
                split_role="test",
                source_file=frame.rel_path,
                source_day=frame.source_day,
                numeric=frame.numeric.reset_index(drop=True),
                labels=frame.labels.copy(),
                attack_types=frame.attack_types.copy(),
                row_indices=frame.row_indices.copy(),
            )
        )
    return out


def _fit_scaler(train_segments: list[SplitSegment], scaler_name: str) -> MinMaxScaler:
    if str(scaler_name).lower().strip() != "minmax":
        raise ValueError("Only minmax scaler is implemented in journal_rebuild pilot")
    scaler = MinMaxScaler()
    scaler.fit(pd.concat([seg.numeric for seg in train_segments], ignore_index=True).astype(np.float32))
    return scaler


def _transform_segments(train_segments: list[SplitSegment], scaler: MinMaxScaler, clip_minmax: bool) -> list[SplitSegment]:
    out: list[SplitSegment] = []
    for seg in train_segments:
        arr = scaler.transform(seg.numeric.astype(np.float32)).astype(np.float32)
        if clip_minmax:
            arr = np.clip(arr, 0.0, 1.0)
        out.append(
            SplitSegment(
                split_role=seg.split_role,
                source_file=seg.source_file,
                source_day=seg.source_day,
                numeric=pd.DataFrame(arr, columns=seg.numeric.columns),
                labels=seg.labels.copy(),
                attack_types=seg.attack_types.copy(),
                row_indices=seg.row_indices.copy(),
            )
        )
    return out


def _encode_attack_families(segments_by_role: dict[str, list[SplitSegment]]) -> dict[str, int]:
    names = {"BENIGN"}
    for segments in segments_by_role.values():
        for seg in segments:
            names.update(str(x).strip() for x in seg.attack_types.tolist())
    ordered = sorted(names)
    return {name: idx for idx, name in enumerate(ordered)}


def _build_split_arrays(
    segments: list[SplitSegment],
    attack_family_to_code: dict[str, int],
    source_file_to_code: dict[str, int],
) -> SplitArrays:
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    attack_codes: list[np.ndarray] = []
    file_codes: list[np.ndarray] = []
    row_indices: list[np.ndarray] = []
    for seg in segments:
        feats.append(seg.numeric.to_numpy(dtype=np.float32, copy=True))
        labels.append(np.asarray(seg.labels, dtype=np.uint8))
        attack_codes.append(np.asarray([attack_family_to_code[str(v).strip()] for v in seg.attack_types], dtype=np.int16))
        file_codes.append(np.full(len(seg.numeric), source_file_to_code[seg.source_file], dtype=np.int16))
        row_indices.append(np.asarray(seg.row_indices, dtype=np.int64))
    return SplitArrays(
        features=np.concatenate(feats, axis=0) if feats else np.zeros((0, 0), dtype=np.float32),
        labels=np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=np.uint8),
        attack_code=np.concatenate(attack_codes, axis=0) if attack_codes else np.zeros((0,), dtype=np.int16),
        source_file_code=np.concatenate(file_codes, axis=0) if file_codes else np.zeros((0,), dtype=np.int16),
        row_index=np.concatenate(row_indices, axis=0) if row_indices else np.zeros((0,), dtype=np.int64),
    )


def _dominant_attack_family(attack_names: list[str]) -> str:
    filtered = [name for name in attack_names if name.strip().lower() not in BENIGN_TOKENS]
    if not filtered:
        return "BENIGN"
    uniq, counts = np.unique(np.asarray(filtered, dtype=object), return_counts=True)
    return str(uniq[int(np.argmax(counts))])


def _build_window_manifest(
    segments_by_role: dict[str, list[SplitSegment]],
    *,
    window_size: int,
    stride: int,
    anomaly_ratio: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    role_order = ["model_train_benign", "independent_calibration_benign", "test"]
    role_offsets = {role: 0 for role in role_order}
    role_window_counter = {role: 0 for role in role_order}
    global_window_index = 0
    for role in role_order:
        for seg in segments_by_role.get(role, []):
            seg_len = len(seg.numeric)
            if seg_len < int(window_size):
                role_offsets[role] += seg_len
                continue
            local_labels = np.asarray(seg.labels, dtype=np.uint8)
            local_attacks = np.asarray(seg.attack_types, dtype=object)
            for local_start in range(0, seg_len - int(window_size) + 1, int(stride)):
                local_end = local_start + int(window_size)
                window_labels = local_labels[local_start:local_end]
                window_attack_names = local_attacks[local_start:local_end].tolist()
                if role == "test":
                    attack_ratio = float(window_labels.mean())
                    label = int(attack_ratio >= float(anomaly_ratio))
                    attack_family = _dominant_attack_family(window_attack_names) if label == 1 else "BENIGN"
                else:
                    attack_ratio = 0.0
                    label = 0
                    attack_family = "BENIGN"
                start_row = int(seg.row_indices[local_start])
                end_row = int(seg.row_indices[local_end - 1])
                split_start_offset = role_offsets[role] + local_start
                split_end_offset = role_offsets[role] + local_end - 1
                rows.append(
                    {
                        "window_id": f"{role}:{global_window_index:06d}",
                        "split_role": role,
                        "source_file": seg.source_file,
                        "source_day": seg.source_day,
                        "window_index_in_split": role_window_counter[role],
                        "start_row": start_row,
                        "end_row": end_row,
                        "start_line": start_row + 2,
                        "end_line": end_row + 2,
                        "split_start_offset": split_start_offset,
                        "split_end_offset": split_end_offset,
                        "label": label,
                        "attack_family": attack_family,
                        "benign_flag": bool(label == 0),
                        "attack_ratio": attack_ratio,
                        "timestamp_start": "",
                        "timestamp_end": "",
                    }
                )
                role_window_counter[role] += 1
                global_window_index += 1
            role_offsets[role] += seg_len
    return pd.DataFrame(rows)


def build_canonical_artifacts(root_dir: Path, data_config: dict[str, Any]) -> dict[str, Any]:
    data_root = root_dir / str(data_config["data_dir"])
    processed_dir = root_dir / str(data_config["processed_dir"])
    manifests_dir = (root_dir / "journal_rebuild" / "data" / "manifests").resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    train_files = [str(x) for x in data_config["train_files"]]
    test_files = [str(x) for x in data_config["test_files"]]
    all_files = [str(x) for x in data_config["include_files"]]

    source_file_hashes = {name: sha256_file(data_root / name) for name in all_files}
    train_frames = [_prepare_frame(data_root, rel_path) for rel_path in train_files]
    test_frames = [_prepare_frame(data_root, rel_path) for rel_path in test_files]
    feature_names = _align_common_features(train_frames + test_frames)

    train_segments_raw, calib_segments_raw = _split_train_calibration_segments(
        train_frames,
        float(data_config["calibration_ratio"]),
    )
    test_segments_raw = _build_test_segments(test_frames)

    scaler = _fit_scaler(train_segments_raw, str(data_config["scaler"]))
    scaler_hash, scaler_bytes = sha256_pickle(scaler)
    scaler_path = root_dir / str(data_config["scaler_path"])
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.write_bytes(scaler_bytes)

    clip_minmax = bool(data_config.get("clip_minmax", True))
    train_segments = _transform_segments(train_segments_raw, scaler, clip_minmax)
    calib_segments = _transform_segments(calib_segments_raw, scaler, clip_minmax)
    test_segments = _transform_segments(test_segments_raw, scaler, clip_minmax)

    segments_by_role = {
        "model_train_benign": train_segments,
        "independent_calibration_benign": calib_segments,
        "test": test_segments,
    }
    source_file_to_code = {name: idx for idx, name in enumerate(all_files)}
    attack_family_to_code = _encode_attack_families(segments_by_role)

    split_arrays = {
        role: _build_split_arrays(segments, attack_family_to_code, source_file_to_code)
        for role, segments in segments_by_role.items()
    }
    manifest_df = _build_window_manifest(
        segments_by_role,
        window_size=int(data_config["window_size"]),
        stride=int(data_config["stride"]),
        anomaly_ratio=float(data_config["anomaly_ratio"]),
    )
    manifest_path = root_dir / str(data_config["manifest_path"])
    manifest_df.to_csv(manifest_path, index=False)
    manifest_hash = sha256_file(manifest_path)

    split_npz_paths: dict[str, str] = {}
    for role, arrays in split_arrays.items():
        out_path = processed_dir / f"{role}_records.npz"
        np.savez_compressed(
            out_path,
            features=arrays.features,
            labels=arrays.labels,
            attack_code=arrays.attack_code,
            source_file_code=arrays.source_file_code,
            row_index=arrays.row_index,
        )
        split_npz_paths[role] = str(out_path.relative_to(root_dir))

    feature_path = processed_dir / "feature_names.json"
    feature_path.write_text(json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8")
    mapping_path = processed_dir / "mappings.json"
    mapping_path.write_text(
        json.dumps(
            {
                "source_file_to_code": source_file_to_code,
                "attack_family_to_code": attack_family_to_code,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    split_record_counts = {role: int(len(arrays.features)) for role, arrays in split_arrays.items()}
    split_window_counts = {
        role: int((manifest_df["split_role"] == role).sum())
        for role in ("model_train_benign", "independent_calibration_benign", "test")
    }
    raw_row_counts = {frame.rel_path: int(len(frame.numeric)) for frame in train_frames + test_frames}
    timestamp_present = any(frame.timestamp_values is not None for frame in train_frames + test_frames)

    data_manifest = {
        "dataset_name": str(data_config["dataset_name"]),
        "window_size": int(data_config["window_size"]),
        "stride": int(data_config["stride"]),
        "anomaly_ratio": float(data_config["anomaly_ratio"]),
        "calibration_ratio": float(data_config["calibration_ratio"]),
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "raw_row_counts": raw_row_counts,
        "split_record_counts": split_record_counts,
        "split_window_counts": split_window_counts,
        "source_file_hashes": source_file_hashes,
        "manifest_path": str(manifest_path.relative_to(root_dir)),
        "manifest_hash": manifest_hash,
        "scaler_path": str(scaler_path.relative_to(root_dir)),
        "scaler_hash": scaler_hash,
        "split_npz_paths": split_npz_paths,
        "mapping_path": str(mapping_path.relative_to(root_dir)),
        "feature_path": str(feature_path.relative_to(root_dir)),
        "scaler_fit_source": "model_train_benign",
        "excluded_files": [str(x) for x in data_config.get("excluded_files", [])],
        "timestamp_present": bool(timestamp_present),
        "split_definition": {
            "model_train_benign": "first chronological benign-only training-day records after cleaning",
            "independent_calibration_benign": "remaining chronological benign-only training-day records after cleaning",
            "test": "all chronological Friday records after cleaning",
        },
    }
    data_manifest["data_manifest_hash"] = sha256_json(data_manifest)
    data_manifest_path = root_dir / str(data_config["data_manifest_path"])
    data_manifest_path.write_text(json.dumps(data_manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return data_manifest


def load_scaler(path: str | Path) -> MinMaxScaler:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def load_data_manifest(root_dir: Path, path: str | Path) -> dict[str, Any]:
    return json.loads((root_dir / Path(path)).read_text(encoding="utf-8"))


def load_split_arrays(root_dir: Path, data_manifest: dict[str, Any], role: str) -> SplitArrays:
    npz = np.load(root_dir / str(data_manifest["split_npz_paths"][role]))
    return SplitArrays(
        features=np.asarray(npz["features"], dtype=np.float32),
        labels=np.asarray(npz["labels"], dtype=np.uint8),
        attack_code=np.asarray(npz["attack_code"], dtype=np.int16),
        source_file_code=np.asarray(npz["source_file_code"], dtype=np.int16),
        row_index=np.asarray(npz["row_index"], dtype=np.int64),
    )


def load_window_manifest(root_dir: Path, data_manifest: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(root_dir / str(data_manifest["manifest_path"]))
