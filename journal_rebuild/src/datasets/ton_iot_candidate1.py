from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from journal_rebuild.src.utils.hashing import sha256_file, sha256_json, sha256_pickle


MISSING_TOKENS = {"", "-", "nan", "none", "null", "nat"}
BENIGN_TOKENS = {"normal", "benign", "0", "false", "no"}


@dataclass
class SplitSegment:
    split_role: str
    source_file: str
    df: pd.DataFrame


@dataclass
class SplitArrays:
    features: np.ndarray
    labels: np.ndarray
    attack_code: np.ndarray
    source_file_code: np.ndarray
    row_index: np.ndarray


def _normalize_text(value: Any) -> str:
    text = str(value).strip()
    return "" if text.lower() in MISSING_TOKENS else text


def _normalize_category(value: Any) -> str:
    text = _normalize_text(value)
    if text == "":
        return "MISSING"
    return text.lower()


def _sanitize_token(token: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(token)).strip("_") or "token"


def _dominant_attack_type(values: list[str]) -> str:
    filtered = [str(v).strip() for v in values if str(v).strip().lower() not in BENIGN_TOKENS]
    if not filtered:
        return "BENIGN"
    uniq, counts = np.unique(np.asarray(filtered, dtype=object), return_counts=True)
    return str(uniq[int(np.argmax(counts))])


def _entity_summary(values: pd.Series, max_items: int = 3) -> str:
    counts = values.astype(str).value_counts()
    chunks = [f"{idx}({int(cnt)})" for idx, cnt in counts.head(max_items).items()]
    extra = int(max(len(counts) - max_items, 0))
    return ",".join(chunks) + (f",+{extra}more" if extra > 0 else "")


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def _scan_file_ranges(data_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(data_root.glob("Network_dataset_*.csv")):
        ts = pd.read_csv(path, usecols=["ts"], low_memory=False)["ts"]
        rows.append(
            {
                "file": path.name,
                "path": path,
                "min_ts": int(ts.min()),
                "max_ts": int(ts.max()),
            }
        )
    rows.sort(key=lambda x: (x["min_ts"], x["file"]))
    for idx, row in enumerate(rows):
        row["file_order"] = idx
    return rows


def _interval_overlaps(interval: tuple[int, int], lo: int, hi: int) -> bool:
    return hi >= interval[0] and lo <= interval[1]


def _select_rows_for_ranges(
    *,
    file_meta: dict[str, Any],
    split_ranges: dict[str, tuple[int, int]],
    required_columns: list[str],
) -> list[pd.DataFrame]:
    path = Path(file_meta["path"])
    df = _read_csv(path)
    df["_original_row"] = np.arange(len(df), dtype=np.int64)
    df["_file_order"] = int(file_meta["file_order"])
    selected: list[pd.DataFrame] = []
    for split_role, (lo, hi) in split_ranges.items():
        mask = (df["ts"] >= int(lo)) & (df["ts"] <= int(hi))
        if not mask.any():
            continue
        sub = df.loc[mask, required_columns + ["_original_row", "_file_order"]].copy()
        sub["split_role"] = split_role
        sub["source_file"] = path.name
        selected.append(sub)
    return selected


def _stable_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["ts", "_file_order", "_original_row"], kind="mergesort").reset_index(drop=True)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if series.dtype.kind in {"i", "u", "f"}:
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).map(_normalize_text)
    cleaned = cleaned.replace("", np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def _record_manifest_frame(segments_by_role: dict[str, list[SplitSegment]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    global_idx = 0
    for role in ["model_train_benign", "reference_benign", "independent_calibration_benign", "test"]:
        for seg in segments_by_role.get(role, []):
            sub = seg.df.copy()
            sub["record_id"] = [f"{role}:record:{global_idx + i:07d}" for i in range(len(sub))]
            global_idx += len(sub)
            sub["attack_type"] = sub["type"].astype(str)
            sub["benign_flag"] = sub["label"].astype(int) == 0
            rows.append(
                sub[
                    [
                        "record_id",
                        "split_role",
                        "source_file",
                        "_original_row",
                        "ts",
                        "src_ip",
                        "src_port",
                        "dst_ip",
                        "dst_port",
                        "proto",
                        "service",
                        "conn_state",
                        "label",
                        "attack_type",
                        "benign_flag",
                    ]
                ].rename(columns={"_original_row": "source_row"})
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _fit_category_vocab(train_df: pd.DataFrame, categorical_cols: list[str]) -> dict[str, list[str]]:
    vocab: dict[str, list[str]] = {}
    for col in categorical_cols:
        tokens = sorted({_normalize_category(v) for v in train_df[col].tolist()})
        if "UNK" not in tokens:
            tokens.append("UNK")
        vocab[col] = tokens
    return vocab


def _encode_categories(df: pd.DataFrame, vocab: dict[str, list[str]]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for col, tokens in vocab.items():
        normalized = df[col].map(_normalize_category)
        normalized = normalized.where(normalized.isin(tokens), "UNK")
        for token in tokens:
            feature_name = f"{col}__{_sanitize_token(token)}"
            parts.append(pd.DataFrame({feature_name: (normalized == token).astype(np.float32)}))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)


def _transform_numeric(
    split_frames: dict[str, pd.DataFrame],
    numeric_cols: list[str],
    *,
    drop_missing_ratio_threshold: float,
    clip_minmax: bool,
    scaler_name: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    train_numeric = pd.DataFrame({col: _coerce_numeric(split_frames["model_train_benign"][col]) for col in numeric_cols})
    missing_ratio = train_numeric.isna().mean()
    nunique = train_numeric.nunique(dropna=True)
    kept_numeric = [
        col
        for col in numeric_cols
        if float(missing_ratio[col]) <= float(drop_missing_ratio_threshold) and int(nunique[col]) > 1
    ]
    drop_reasons = {
        col: (
            "near_all_missing"
            if float(missing_ratio[col]) > float(drop_missing_ratio_threshold)
            else "constant_or_single_value"
        )
        for col in numeric_cols
        if col not in kept_numeric
    }
    train_numeric = train_numeric.loc[:, kept_numeric].copy()
    imputer_values = {col: float(train_numeric[col].median()) for col in kept_numeric}
    train_imputed = train_numeric.fillna(imputer_values).astype(np.float32)
    if str(scaler_name).lower().strip() != "minmax":
        raise ValueError("TON_IoT candidate1 currently implements only MinMaxScaler")
    scaler = MinMaxScaler()
    scaler.fit(train_imputed)

    transformed: dict[str, pd.DataFrame] = {}
    for role, frame in split_frames.items():
        numeric = pd.DataFrame({col: _coerce_numeric(frame[col]) for col in kept_numeric})
        numeric = numeric.fillna(imputer_values).astype(np.float32)
        arr = scaler.transform(numeric).astype(np.float32)
        if clip_minmax:
            arr = np.clip(arr, 0.0, 1.0)
        transformed[role] = pd.DataFrame(arr, columns=kept_numeric, index=frame.index)

    diagnostics = {
        "kept_numeric_cols": kept_numeric,
        "drop_reasons": drop_reasons,
        "missing_ratio_train": {col: float(missing_ratio[col]) for col in numeric_cols},
        "nunique_train": {col: int(nunique[col]) for col in numeric_cols},
        "imputer_values": imputer_values,
        "scaler": scaler,
    }
    return transformed, diagnostics


def _build_split_arrays(
    segments: list[SplitSegment],
    *,
    feature_names: list[str],
    attack_family_to_code: dict[str, int],
    source_file_to_code: dict[str, int],
) -> SplitArrays:
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    attack_codes: list[np.ndarray] = []
    file_codes: list[np.ndarray] = []
    row_indices: list[np.ndarray] = []
    for seg in segments:
        feats.append(seg.df.loc[:, feature_names].to_numpy(dtype=np.float32, copy=True))
        labels.append(seg.df["label"].to_numpy(dtype=np.uint8))
        attack_codes.append(
            np.asarray([attack_family_to_code[str(v)] for v in seg.df["type"].astype(str).tolist()], dtype=np.int16)
        )
        file_codes.append(np.full(len(seg.df), int(source_file_to_code[seg.source_file]), dtype=np.int16))
        row_indices.append(seg.df["_original_row"].to_numpy(dtype=np.int64))
    feature_dim = len(feature_names)
    return SplitArrays(
        features=np.concatenate(feats, axis=0) if feats else np.zeros((0, feature_dim), dtype=np.float32),
        labels=np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=np.uint8),
        attack_code=np.concatenate(attack_codes, axis=0) if attack_codes else np.zeros((0,), dtype=np.int16),
        source_file_code=np.concatenate(file_codes, axis=0) if file_codes else np.zeros((0,), dtype=np.int16),
        row_index=np.concatenate(row_indices, axis=0) if row_indices else np.zeros((0,), dtype=np.int64),
    )


def _build_window_manifest(
    segments_by_role: dict[str, list[SplitSegment]],
    *,
    window_size: int,
    stride: int,
    anomaly_ratio: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    dropped_short_records = 0
    role_offsets = {role: 0 for role in ["model_train_benign", "reference_benign", "independent_calibration_benign", "test"]}
    role_window_counter = {role: 0 for role in role_offsets}
    global_window_index = 0
    for role in ["model_train_benign", "reference_benign", "independent_calibration_benign", "test"]:
        for seg in segments_by_role.get(role, []):
            seg_len = len(seg.df)
            if seg_len < int(window_size):
                dropped_short_records += seg_len
                role_offsets[role] += seg_len
                continue
            labels = seg.df["label"].to_numpy(dtype=np.uint8)
            attack_types = seg.df["type"].astype(str).to_numpy(dtype=object)
            for local_start in range(0, seg_len - int(window_size) + 1, int(stride)):
                local_end = local_start + int(window_size)
                window = seg.df.iloc[local_start:local_end]
                window_labels = labels[local_start:local_end]
                window_attack_types = attack_types[local_start:local_end].tolist()
                if role == "test":
                    attack_ratio = float(window_labels.mean())
                    label = int(attack_ratio >= float(anomaly_ratio))
                    attack_type = _dominant_attack_type(window_attack_types) if label == 1 else "BENIGN"
                else:
                    attack_ratio = float(window_labels.mean())
                    label = 0
                    attack_type = "BENIGN"
                start_row = int(window["_original_row"].iloc[0])
                end_row = int(window["_original_row"].iloc[-1])
                start_ts = int(window["ts"].iloc[0])
                end_ts = int(window["ts"].iloc[-1])
                rows.append(
                    {
                        "window_id": f"{role}:{global_window_index:06d}",
                        "split_role": role,
                        "source_file": seg.source_file,
                        "window_index_in_split": role_window_counter[role],
                        "start_row": start_row,
                        "end_row": end_row,
                        "start_line": start_row + 2,
                        "end_line": end_row + 2,
                        "split_start_offset": role_offsets[role] + local_start,
                        "split_end_offset": role_offsets[role] + local_end - 1,
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "source_rows": f"{seg.source_file}:{start_row}-{end_row}",
                        "label": int(label),
                        "attack_type": attack_type,
                        "attack_family": attack_type,
                        "benign_flag": bool(label == 0),
                        "attack_ratio": float(attack_ratio),
                        "benign_ratio": float(1.0 - attack_ratio),
                        "mixed_label_window": bool(np.any(window_labels == 1) and np.any(window_labels == 0)),
                        "src_entity_summary": _entity_summary(window["src_ip"]),
                        "dst_entity_summary": _entity_summary(window["dst_ip"]),
                    }
                )
                role_window_counter[role] += 1
                global_window_index += 1
            role_offsets[role] += seg_len
    return pd.DataFrame(rows), {"dropped_short_records": int(dropped_short_records)}


def build_candidate1_artifacts(root_dir: Path, data_config: dict[str, Any]) -> dict[str, Any]:
    data_root = root_dir / str(data_config["data_dir"])
    manifest_path = root_dir / str(data_config["manifest_path"])
    window_manifest_path = root_dir / str(data_config["window_manifest_path"])
    data_manifest_path = root_dir / str(data_config["data_manifest_path"])
    processed_dir = root_dir / str(data_config["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    split_ranges = {role: tuple(map(int, bounds)) for role, bounds in data_config["split_ranges"].items()}
    file_ranges = _scan_file_ranges(data_root)
    relevant = [
        row
        for row in file_ranges
        if any(_interval_overlaps(split_ranges[role], int(row["min_ts"]), int(row["max_ts"])) for role in split_ranges)
    ]
    relevant.sort(key=lambda x: int(x["file_order"]))

    required_columns = sorted(
        set(
            [
                "ts",
                "src_ip",
                "src_port",
                "dst_ip",
                "dst_port",
                "label",
                "type",
                "proto",
                "service",
                "conn_state",
            ]
            + list(data_config["numeric_candidate_cols"])
            + list(data_config.get("excluded_text_cols", []))
        )
    )

    selected_rows: list[pd.DataFrame] = []
    for meta in relevant:
        selected_rows.extend(
            _select_rows_for_ranges(
                file_meta=meta,
                split_ranges=split_ranges,
                required_columns=required_columns,
            )
        )
    if not selected_rows:
        raise ValueError("TON_IoT candidate1 selected no rows")

    split_frames: dict[str, pd.DataFrame] = {}
    contamination: dict[str, dict[str, int]] = {}
    for role in ["model_train_benign", "reference_benign", "independent_calibration_benign", "test"]:
        sub_parts = [frame for frame in selected_rows if frame["split_role"].iloc[0] == role]
        if not sub_parts:
            split_frames[role] = pd.DataFrame(columns=required_columns + ["_original_row", "_file_order", "split_role", "source_file"])
            contamination[role] = {"rows_before_drop": 0, "non_benign_rows_dropped": 0}
            continue
        role_df = _stable_sort(pd.concat(sub_parts, ignore_index=True))
        rows_before = len(role_df)
        dropped = 0
        if role in {"model_train_benign", "reference_benign", "independent_calibration_benign"}:
            bad_mask = role_df["label"].astype(int) != 0
            dropped = int(bad_mask.sum())
            if dropped and bool(data_config.get("drop_contaminated_benign_rows", True)):
                role_df = role_df.loc[~bad_mask].reset_index(drop=True)
            elif dropped:
                raise ValueError(f"{role} contains non-benign rows and drop_contaminated_benign_rows=false")
        split_frames[role] = role_df.reset_index(drop=True)
        contamination[role] = {"rows_before_drop": int(rows_before), "non_benign_rows_dropped": int(dropped)}

    numeric_transformed, numeric_diag = _transform_numeric(
        split_frames,
        list(data_config["numeric_candidate_cols"]),
        drop_missing_ratio_threshold=float(data_config["drop_missing_ratio_threshold"]),
        clip_minmax=bool(data_config.get("clip_minmax", True)),
        scaler_name=str(data_config["scaler"]),
    )
    categorical_vocab = _fit_category_vocab(split_frames["model_train_benign"], list(data_config["categorical_cols"]))
    categorical_frames = {role: _encode_categories(frame, categorical_vocab) for role, frame in split_frames.items()}

    feature_names = list(numeric_diag["kept_numeric_cols"]) + list(categorical_frames["model_train_benign"].columns)
    for role, frame in split_frames.items():
        enriched = frame.copy()
        for col in numeric_diag["kept_numeric_cols"]:
            enriched[col] = numeric_transformed[role][col].to_numpy(dtype=np.float32)
        for col in categorical_frames[role].columns:
            enriched[col] = categorical_frames[role][col].to_numpy(dtype=np.float32)
        split_frames[role] = enriched

    segments_by_role: dict[str, list[SplitSegment]] = {}
    for role, frame in split_frames.items():
        segments: list[SplitSegment] = []
        if len(frame) > 0:
            for source_file, sub in frame.groupby("source_file", sort=False):
                segments.append(SplitSegment(split_role=role, source_file=str(source_file), df=sub.reset_index(drop=True)))
        segments_by_role[role] = segments

    source_file_to_code = {meta["file"]: idx for idx, meta in enumerate(relevant)}
    attack_names = {"BENIGN"}
    for frame in split_frames.values():
        attack_names.update(str(x) for x in frame["type"].astype(str).unique().tolist())
    attack_family_to_code = {name: idx for idx, name in enumerate(sorted(attack_names))}

    split_arrays = {
        role: _build_split_arrays(
            segments,
            feature_names=feature_names,
            attack_family_to_code=attack_family_to_code,
            source_file_to_code=source_file_to_code,
        )
        for role, segments in segments_by_role.items()
    }

    record_manifest = _record_manifest_frame(segments_by_role)
    record_manifest.to_csv(manifest_path, index=False)
    manifest_hash = sha256_file(manifest_path)

    window_manifest, window_diag = _build_window_manifest(
        segments_by_role,
        window_size=int(data_config["window_size"]),
        stride=int(data_config["stride"]),
        anomaly_ratio=float(data_config["anomaly_ratio"]),
    )
    window_manifest.to_csv(window_manifest_path, index=False)
    window_manifest_hash = sha256_file(window_manifest_path)

    scaler = numeric_diag["scaler"]
    scaler_hash, scaler_bytes = sha256_pickle(scaler)
    scaler_path = root_dir / str(data_config["scaler_path"])
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.write_bytes(scaler_bytes)

    feature_path = processed_dir / "feature_names.json"
    feature_path.write_text(json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8")
    mapping_path = processed_dir / "mappings.json"
    mapping_path.write_text(
        json.dumps(
            {
                "source_file_to_code": source_file_to_code,
                "attack_family_to_code": attack_family_to_code,
                "categorical_vocab": categorical_vocab,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    imputer_path = processed_dir / "imputer.json"
    imputer_path.write_text(
        json.dumps(
            {
                "imputer_values": numeric_diag["imputer_values"],
                "kept_numeric_cols": numeric_diag["kept_numeric_cols"],
                "drop_reasons": numeric_diag["drop_reasons"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

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

    split_record_counts = {role: int(len(frame)) for role, frame in split_frames.items()}
    split_window_counts = {role: int((window_manifest["split_role"] == role).sum()) for role in split_frames}
    raw_overlap_summary = {
        meta["file"]: {
            "min_ts": int(meta["min_ts"]),
            "max_ts": int(meta["max_ts"]),
            "sha256": sha256_file(meta["path"]),
        }
        for meta in relevant
    }

    data_manifest = {
        "dataset_name": str(data_config["dataset_name"]),
        "window_size": int(data_config["window_size"]),
        "stride": int(data_config["stride"]),
        "anomaly_ratio": float(data_config["anomaly_ratio"]),
        "feature_names": feature_names,
        "feature_count": int(len(feature_names)),
        "numeric_feature_count": int(len(numeric_diag["kept_numeric_cols"])),
        "categorical_feature_count": int(len(feature_names) - len(numeric_diag["kept_numeric_cols"])),
        "split_record_counts": split_record_counts,
        "split_window_counts": split_window_counts,
        "source_file_hashes": {name: meta["sha256"] for name, meta in raw_overlap_summary.items()},
        "relevant_files": [meta["file"] for meta in relevant],
        "manifest_path": str(manifest_path.relative_to(root_dir)),
        "manifest_hash": manifest_hash,
        "window_manifest_path": str(window_manifest_path.relative_to(root_dir)),
        "window_manifest_hash": window_manifest_hash,
        "scaler_path": str(scaler_path.relative_to(root_dir)),
        "scaler_hash": scaler_hash,
        "split_npz_paths": split_npz_paths,
        "mapping_path": str(mapping_path.relative_to(root_dir)),
        "feature_path": str(feature_path.relative_to(root_dir)),
        "imputer_path": str(imputer_path.relative_to(root_dir)),
        "scaler_fit_source": "model_train_benign",
        "categorical_vocab_fit_source": "model_train_benign",
        "split_ranges": {role: [int(v[0]), int(v[1])] for role, v in split_ranges.items()},
        "contamination": contamination,
        "dropped_short_records": int(window_diag["dropped_short_records"]),
        "data_quality": {
            "drop_missing_ratio_threshold": float(data_config["drop_missing_ratio_threshold"]),
            "numeric_drop_reasons": numeric_diag["drop_reasons"],
            "categorical_vocab": categorical_vocab,
        },
        "split_definition": {
            "model_train_benign": "fixed candidate1 benign interval after stable ts sort; non-benign contaminants dropped if present",
            "reference_benign": "fixed candidate1 benign reference interval after stable ts sort; non-benign contaminants dropped if present",
            "independent_calibration_benign": "fixed candidate1 benign calibration interval after stable ts sort; non-benign contaminants dropped if present",
            "test": "fixed candidate1 mixed interval after stable ts sort, preserving benign and attack rows",
        },
    }
    data_manifest["data_manifest_hash"] = sha256_json(data_manifest)
    data_manifest_path.write_text(json.dumps(data_manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return data_manifest


def load_scaler(path: str | Path) -> MinMaxScaler:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def load_data_manifest(root_dir: Path, path: str | Path) -> dict[str, Any]:
    return json.loads((root_dir / Path(path)).read_text(encoding="utf-8"))


def load_split_arrays(root_dir: Path, data_manifest: dict[str, Any], role: str) -> SplitArrays:
    npz = np.load(root_dir / str(data_manifest["split_npz_paths"][role]))
    feature_dim = int(data_manifest["feature_count"])
    return SplitArrays(
        features=np.asarray(npz["features"], dtype=np.float32).reshape(-1, feature_dim),
        labels=np.asarray(npz["labels"], dtype=np.uint8),
        attack_code=np.asarray(npz["attack_code"], dtype=np.int16),
        source_file_code=np.asarray(npz["source_file_code"], dtype=np.int16),
        row_index=np.asarray(npz["row_index"], dtype=np.int64),
    )


def load_record_manifest(root_dir: Path, data_manifest: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(root_dir / str(data_manifest["manifest_path"]))


def load_window_manifest(root_dir: Path, data_manifest: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(root_dir / str(data_manifest["window_manifest_path"]))
