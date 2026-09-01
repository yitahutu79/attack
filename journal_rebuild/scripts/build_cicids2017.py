#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from journal_rebuild.src.datasets.cicids2017 import build_canonical_artifacts  # noqa: E402
from journal_rebuild.src.utils.config import load_yaml_like  # noqa: E402


def write_data_audit(root: Path, data_config: dict, data_manifest: dict) -> None:
    out_path = root / "journal_rebuild" / "reports" / "data_audit" / "data_audit.md"
    source_hash_lines = [
        f"- `{name}`: `{digest}`"
        for name, digest in sorted(data_manifest["source_file_hashes"].items())
    ]
    split_record_lines = [
        f"- `{role}` records: `{count}`"
        for role, count in data_manifest["split_record_counts"].items()
    ]
    split_window_lines = [
        f"- `{role}` windows: `{count}`"
        for role, count in data_manifest["split_window_counts"].items()
    ]
    feature_preview = ", ".join(data_manifest["feature_names"][:10])
    text = "\n".join(
        [
            "# CICIDS2017 Data Audit",
            "",
            "## Protocol",
            "",
            "- Raw CSV source rebuilt from `dataset/CICIDS2017`.",
            f"- Window size: `{data_manifest['window_size']}`",
            f"- Stride: `{data_manifest['stride']}`",
            f"- Window anomaly ratio: `{data_manifest['anomaly_ratio']}`",
            f"- Calibration ratio: `{data_manifest['calibration_ratio']}`",
            "- Split logic: record-level chronological split first, then per-segment windowization without crossing split boundaries.",
            "- `Monday-WorkingHours.pcap_ISCX.csv` is excluded from the canonical rebuild and recorded only as excluded metadata.",
            "",
            "## Column / provenance notes",
            "",
            f"- Common numeric feature count: `{data_manifest['feature_count']}`",
            f"- Feature preview: `{feature_preview}`",
            f"- Timestamp column present in used raw files: `{data_manifest['timestamp_present']}`",
            "- Since CICIDS2017 used here has no timestamp field in the selected CSVs, each window is traced by source file and original row interval.",
            "",
            "## Split counts",
            "",
            *split_record_lines,
            *split_window_lines,
            "",
            "## Artifact hashes",
            "",
            f"- Manifest hash: `{data_manifest['manifest_hash']}`",
            f"- Scaler hash: `{data_manifest['scaler_hash']}`",
            f"- Data manifest hash: `{data_manifest['data_manifest_hash']}`",
            "",
            "## Raw file hashes",
            "",
            *source_hash_lines,
            "",
            "## Paths",
            "",
            f"- Manifest: `{data_manifest['manifest_path']}`",
            f"- Data manifest: `{data_config['data_manifest_path']}`",
            f"- Scaler: `{data_manifest['scaler_path']}`",
        ]
    )
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    data_config = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "data" / "cicids2017.yaml")
    data_manifest = build_canonical_artifacts(ROOT, data_config)
    write_data_audit(ROOT, data_config, data_manifest)
    print(json.dumps({"manifest_hash": data_manifest["manifest_hash"], "scaler_hash": data_manifest["scaler_hash"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
