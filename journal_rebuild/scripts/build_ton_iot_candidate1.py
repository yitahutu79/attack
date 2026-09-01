#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from journal_rebuild.src.datasets.ton_iot_candidate1 import (  # noqa: E402
    _normalize_category,
    _normalize_text,
    _scan_file_ranges,
    _select_rows_for_ranges,
    _stable_sort,
    build_candidate1_artifacts,
    load_data_manifest,
    load_record_manifest,
    load_window_manifest,
)
from journal_rebuild.src.utils.config import load_yaml_like  # noqa: E402


REPORT_ROOT = ROOT / "journal_rebuild" / "reports" / "ton_iot_candidate1"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns.tolist()]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = [str(row[col]) for col in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _load_selected_raw_frames(data_config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    data_root = ROOT / str(data_config["data_dir"])
    split_ranges = {role: tuple(map(int, bounds)) for role, bounds in data_config["split_ranges"].items()}
    file_ranges = _scan_file_ranges(data_root)
    relevant = [
        row
        for row in file_ranges
        if any(row["max_ts"] >= bounds[0] and row["min_ts"] <= bounds[1] for bounds in split_ranges.values())
    ]
    required_columns = sorted(
        set(
            ["ts", "src_ip", "src_port", "dst_ip", "dst_port", "label", "type", "proto", "service", "conn_state"]
            + list(data_config["numeric_candidate_cols"])
            + list(data_config.get("excluded_text_cols", []))
        )
    )
    rows: list[pd.DataFrame] = []
    for meta in relevant:
        rows.extend(
            _select_rows_for_ranges(
                file_meta=meta,
                split_ranges=split_ranges,
                required_columns=required_columns,
            )
        )
    split_frames: dict[str, pd.DataFrame] = {}
    for role in split_ranges:
        parts = [frame for frame in rows if frame["split_role"].iloc[0] == role]
        if not parts:
            split_frames[role] = pd.DataFrame(columns=required_columns + ["_original_row", "_file_order", "split_role", "source_file"])
            continue
        role_df = _stable_sort(pd.concat(parts, ignore_index=True))
        if role in {"model_train_benign", "reference_benign", "independent_calibration_benign"}:
            role_df = role_df.loc[role_df["label"].astype(int) == 0].reset_index(drop=True)
        split_frames[role] = role_df
    return split_frames


def _write_split_validation(data_manifest: dict[str, Any], record_manifest: pd.DataFrame, window_manifest: pd.DataFrame) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    split_counts = record_manifest.groupby("split_role").size().to_dict()
    test_sub = record_manifest[record_manifest["split_role"] == "test"]
    overlap_rows = []
    entity_sets = {}
    for role in ["model_train_benign", "reference_benign", "independent_calibration_benign", "test"]:
        sub = record_manifest[record_manifest["split_role"] == role]
        entity_sets[role] = set(sub["src_ip"].astype(str).tolist()) | set(sub["dst_ip"].astype(str).tolist())
    roles = list(entity_sets)
    for left in roles:
        for right in roles:
            if left >= right:
                continue
            overlap_rows.append(
                {
                    "left_role": left,
                    "right_role": right,
                    "shared_entities": len(entity_sets[left] & entity_sets[right]),
                }
            )
    overlap_df = pd.DataFrame(overlap_rows)
    attack_types = sorted(x for x in test_sub["attack_type"].astype(str).unique().tolist() if x != "normal")

    summary_rows = []
    for role in ["model_train_benign", "reference_benign", "independent_calibration_benign", "test"]:
        sub = record_manifest[record_manifest["split_role"] == role]
        summary_rows.append(
            {
                "split_role": role,
                "records": int(len(sub)),
                "windows": int((window_manifest["split_role"] == role).sum()),
                "source_files": ",".join(sorted(sub["source_file"].astype(str).unique().tolist())),
                "min_ts": int(sub["ts"].min()) if len(sub) else "",
                "max_ts": int(sub["ts"].max()) if len(sub) else "",
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    contamination = data_manifest["contamination"]
    text = "\n".join(
        [
            "# TON_IoT Candidate 1 Split Validation",
            "",
            "## Split counts",
            "",
            dataframe_to_markdown(summary_df),
            "",
            "## Test composition",
            "",
            f"- Test benign records: `{int((test_sub['label'] == 0).sum())}`",
            f"- Test attack records: `{int((test_sub['label'] == 1).sum())}`",
            f"- Test attack categories: `{', '.join(attack_types)}`",
            "",
            "## Benign-split contamination audit",
            "",
            f"- `model_train_benign` dropped non-benign rows: `{contamination['model_train_benign']['non_benign_rows_dropped']}`",
            f"- `reference_benign` dropped non-benign rows: `{contamination['reference_benign']['non_benign_rows_dropped']}`",
            f"- `independent_calibration_benign` dropped non-benign rows: `{contamination['independent_calibration_benign']['non_benign_rows_dropped']}`",
            "",
            "## Cross-split entity overlap",
            "",
            dataframe_to_markdown(overlap_df),
            "",
            "## Verdict",
            "",
            f"- Candidate 1 fixed ranges are buildable: `True`.",
            f"- Pure-benign validation required dropping `{contamination['reference_benign']['non_benign_rows_dropped'] + contamination['independent_calibration_benign']['non_benign_rows_dropped']}` contaminant attack rows from file `Network_dataset_22.csv`.",
            f"- Cross-split entity overlap exists: `{bool((overlap_df['shared_entities'] > 0).any())}`.",
            f"- Test includes only `backdoor` and `mitm` attacks after the fixed Candidate 1 time boundary.",
        ]
    )
    (REPORT_ROOT / "split_validation.md").write_text(text, encoding="utf-8")


def _build_feature_inventory(data_config: dict[str, Any], data_manifest: dict[str, Any], split_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    train_df = split_frames["model_train_benign"]
    mapping = json.loads((ROOT / str(data_manifest["mapping_path"])).read_text(encoding="utf-8"))
    imputer = json.loads((ROOT / str(data_manifest["imputer_path"])).read_text(encoding="utf-8"))
    kept_numeric = set(imputer["kept_numeric_cols"])
    categorical_cols = set(data_config["categorical_cols"])
    forbidden = set(data_config["forbidden_model_cols"])
    excluded_text = set(data_config.get("excluded_text_cols", []))
    rows = []
    for col in [c for c in train_df.columns if not c.startswith("_") and c not in {"split_role", "source_file"}]:
        series = train_df[col]
        if col in kept_numeric:
            role_group = "A_numeric_input"
            note = "scaled from train-only median-imputed numeric feature"
        elif col in categorical_cols:
            role_group = "B_categorical_input"
            note = "train-fit vocabulary; one-hot with UNK for unseen categories"
        elif col in {"ts", "src_ip", "src_port", "dst_ip", "dst_port"}:
            role_group = "C_metadata_only"
            note = "retained only for ordering/traceability; forbidden as model input"
        elif col in {"label", "type"} or col in forbidden or col in excluded_text:
            role_group = "D_forbidden_or_excluded"
            note = "excluded from model features"
        else:
            role_group = "D_forbidden_or_excluded"
            note = "not selected by candidate1 feature plan"
        if series.dtype == object:
            missing_like_ratio = float(series.astype(str).map(_normalize_text).eq("").mean())
        else:
            missing_like_ratio = float(series.isna().mean())
        rows.append(
            {
                "column": col,
                "train_dtype": str(series.dtype),
                "role_group": role_group,
                "train_nunique": int(series.nunique(dropna=False)),
                "train_missing_like_ratio": missing_like_ratio,
                "note": note,
            }
        )
    return pd.DataFrame(rows).sort_values(["role_group", "column"]).reset_index(drop=True)


def _write_feature_plan(
    data_config: dict[str, Any],
    data_manifest: dict[str, Any],
    split_frames: dict[str, pd.DataFrame],
    feature_inventory: pd.DataFrame,
) -> None:
    mapping = json.loads((ROOT / str(data_manifest["mapping_path"])).read_text(encoding="utf-8"))
    imputer = json.loads((ROOT / str(data_manifest["imputer_path"])).read_text(encoding="utf-8"))
    train_raw = split_frames["model_train_benign"]
    test_raw = split_frames["test"]
    train_src_bytes_non_numeric = int(pd.to_numeric(train_raw["src_bytes"].map(_normalize_text).replace("", np.nan), errors="coerce").isna().sum())
    test_src_bytes_non_numeric = int(pd.to_numeric(test_raw["src_bytes"].map(_normalize_text).replace("", np.nan), errors="coerce").isna().sum())
    lines = [
        "# TON_IoT Candidate 1 Feature Processing Plan",
        "",
        "## Final model inputs",
        "",
        f"- Numeric inputs: `{', '.join(imputer['kept_numeric_cols'])}`",
        f"- Categorical inputs: `{', '.join(data_config['categorical_cols'])}`",
        f"- One-hot feature count: `{data_manifest['categorical_feature_count']}`",
        "",
        "## Metadata only",
        "",
        "- `ts`, `src_ip`, `dst_ip`, `src_port`, `dst_port`, `source_file`, `source_row`",
        "",
        "## Forbidden / excluded",
        "",
        f"- Explicitly forbidden: `{', '.join(data_config['forbidden_model_cols'])}`",
        f"- Excluded sparse text fields: `{', '.join(data_config['excluded_text_cols'])}`",
        f"- Train-only dropped numeric columns: `{', '.join(sorted(imputer['drop_reasons'].keys())) if imputer['drop_reasons'] else 'none'}`",
        "",
        "## Train-only fitting rules",
        "",
        "- Numeric missing values are estimated and filled using train-only medians.",
        "- Category vocabularies are fit only on `model_train_benign`; calibration/test unseen tokens map to `UNK`.",
        "- `MinMaxScaler` is fit only on `model_train_benign` after numeric imputation.",
        "- Constant / near-all-missing numeric columns are removed using train-only criteria.",
        "",
        "## `src_bytes` mixed-type audit",
        "",
        f"- Train rows with non-numeric `src_bytes` after normalization: `{train_src_bytes_non_numeric}`",
        f"- Test rows with non-numeric `src_bytes` after normalization: `{test_src_bytes_non_numeric}`",
        "",
        "## Category vocabulary sizes",
        "",
        *[f"- `{col}`: `{len(vocab)}` tokens (including `UNK`)" for col, vocab in mapping["categorical_vocab"].items()],
    ]
    (REPORT_ROOT / "feature_processing_plan.md").write_text("\n".join(lines), encoding="utf-8")
    feature_inventory.to_csv(REPORT_ROOT / "feature_inventory.csv", index=False)


def _compute_leakage_statistics(split_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    union = pd.concat(
        [split_frames["model_train_benign"], split_frames["reference_benign"], split_frames["independent_calibration_benign"], split_frames["test"]],
        ignore_index=True,
    )
    union["split_group"] = np.where(union["split_role"] == "test", "test", "benign_splits")
    rows = []
    for field in ["src_ip", "dst_ip", "proto", "service", "conn_state"]:
        grouped = union.groupby(field, dropna=False)
        for value, sub in grouped:
            if len(sub) < 50:
                continue
            attack_rate = float(sub["label"].mean())
            dominant_type = str(sub["type"].astype(str).value_counts().idxmax())
            purity = float(sub["type"].astype(str).value_counts(normalize=True).iloc[0])
            rows.append(
                {
                    "field": field,
                    "value": str(value),
                    "count": int(len(sub)),
                    "attack_rate": attack_rate,
                    "dominant_type": dominant_type,
                    "dominant_type_purity": purity,
                    "train_like_count": int((sub["split_group"] == "benign_splits").sum()),
                    "test_count": int((sub["split_group"] == "test").sum()),
                }
            )
    bucket = ((union["ts"].astype(np.int64) - int(union["ts"].min())) // 3600).astype(int)
    union["ts_hour_bucket"] = bucket
    for value, sub in union.groupby("ts_hour_bucket"):
        attack_rate = float(sub["label"].mean())
        dominant_type = str(sub["type"].astype(str).value_counts().idxmax())
        purity = float(sub["type"].astype(str).value_counts(normalize=True).iloc[0])
        rows.append(
            {
                "field": "ts_hour_bucket",
                "value": str(value),
                "count": int(len(sub)),
                "attack_rate": attack_rate,
                "dominant_type": dominant_type,
                "dominant_type_purity": purity,
                "train_like_count": int((sub["split_group"] == "benign_splits").sum()),
                "test_count": int((sub["split_group"] == "test").sum()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["field", "count"], ascending=[True, False]).reset_index(drop=True)
    out.to_csv(REPORT_ROOT / "leakage_statistics.csv", index=False)
    return out


def _write_leakage_audit(split_frames: dict[str, pd.DataFrame], leakage_stats: pd.DataFrame) -> None:
    train_df = split_frames["model_train_benign"]
    test_df = split_frames["test"]
    train_entities = set(train_df["src_ip"].astype(str)) | set(train_df["dst_ip"].astype(str))
    test_entities = set(test_df["src_ip"].astype(str)) | set(test_df["dst_ip"].astype(str))
    src_overlap = len(set(train_df["src_ip"].astype(str)) & set(test_df["src_ip"].astype(str)))
    dst_overlap = len(set(train_df["dst_ip"].astype(str)) & set(test_df["dst_ip"].astype(str)))
    entity_overlap = len(train_entities & test_entities)
    top_ip = leakage_stats[leakage_stats["field"].isin(["src_ip", "dst_ip"])].head(10).copy()
    top_cat = leakage_stats[leakage_stats["field"].isin(["proto", "service", "conn_state"])].head(12).copy()
    top_ip["attack_rate"] = top_ip["attack_rate"].map(lambda v: f"{v:.4f}")
    top_ip["dominant_type_purity"] = top_ip["dominant_type_purity"].map(lambda v: f"{v:.4f}")
    top_cat["attack_rate"] = top_cat["attack_rate"].map(lambda v: f"{v:.4f}")
    top_cat["dominant_type_purity"] = top_cat["dominant_type_purity"].map(lambda v: f"{v:.4f}")
    lines = [
        "# TON_IoT Candidate 1 Leakage Audit",
        "",
        "## Entity overlap",
        "",
        f"- Shared `src_ip` values between train and test: `{src_overlap}`",
        f"- Shared `dst_ip` values between train and test: `{dst_overlap}`",
        f"- Shared union entities between train and test: `{entity_overlap}`",
        "",
        "## Strongest IP-linked associations",
        "",
        dataframe_to_markdown(top_ip),
        "",
        "## Strongest protocol/service/state associations",
        "",
        dataframe_to_markdown(top_cat),
        "",
        "## Readout",
        "",
        "- `ts` is highly scenario-linked because the fixed Candidate 1 protocol uses disjoint time blocks; it must remain metadata only.",
        "- `src_ip` / `dst_ip` show non-trivial train/test overlap and strong attack-type purity in test-heavy values, so they remain forbidden as model inputs.",
        "- `proto`, `service`, and `conn_state` are allowed but still need limitation text because some values are highly attack-pure in Candidate 1.",
    ]
    (REPORT_ROOT / "leakage_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    data_config = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "data" / "ton_iot_candidate1.yaml")
    data_manifest = build_candidate1_artifacts(ROOT, data_config)
    data_manifest = load_data_manifest(ROOT, data_config["data_manifest_path"])
    record_manifest = load_record_manifest(ROOT, data_manifest)
    window_manifest = load_window_manifest(ROOT, data_manifest)
    split_frames = _load_selected_raw_frames(data_config)

    _write_split_validation(data_manifest, record_manifest, window_manifest)
    feature_inventory = _build_feature_inventory(data_config, data_manifest, split_frames)
    _write_feature_plan(data_config, data_manifest, split_frames, feature_inventory)
    leakage_stats = _compute_leakage_statistics(split_frames)
    _write_leakage_audit(split_frames, leakage_stats)

    print(json.dumps({"manifest_hash": data_manifest["manifest_hash"], "scaler_hash": data_manifest["scaler_hash"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
