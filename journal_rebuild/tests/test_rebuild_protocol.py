from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]

from journal_rebuild.src.calibration.thresholding import threshold_from_benign_fpr  # noqa: E402
from journal_rebuild.src.evaluation.metrics import metrics_at_threshold  # noqa: E402
from journal_rebuild.src.models.tcn_adversarial import compare_model_configs_except_loss, load_checkpoint_validated  # noqa: E402
from journal_rebuild.src.utils.config import load_yaml_like  # noqa: E402


class RebuildProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_manifest_path = ROOT / "journal_rebuild" / "data" / "manifests" / "data_manifest.json"
        cls.manifest_path = ROOT / "journal_rebuild" / "data" / "manifests" / "canonical_split_manifest.csv"
        cls.summary_path = ROOT / "journal_rebuild" / "runs" / "metrics" / "pilot_seed0_summary.csv"
        if not cls.data_manifest_path.exists() or not cls.manifest_path.exists():
            raise unittest.SkipTest("journal_rebuild data artifacts not built yet")
        cls.data_manifest = json.loads(cls.data_manifest_path.read_text(encoding="utf-8"))
        cls.manifest = pd.read_csv(cls.manifest_path)
        cls.summary = pd.read_csv(cls.summary_path) if cls.summary_path.exists() else pd.DataFrame()

    def test_train_calibration_test_windows_do_not_cross_roles(self) -> None:
        roles = ["model_train_benign", "independent_calibration_benign", "test"]
        for source_file in self.manifest["source_file"].unique().tolist():
            role_ranges: dict[str, tuple[int, int]] = {}
            for role in roles:
                sub = self.manifest[(self.manifest["source_file"] == source_file) & (self.manifest["split_role"] == role)]
                if sub.empty:
                    continue
                role_ranges[role] = (int(sub["start_row"].min()), int(sub["end_row"].max()))
            if "model_train_benign" in role_ranges and "independent_calibration_benign" in role_ranges:
                self.assertLess(role_ranges["model_train_benign"][1], role_ranges["independent_calibration_benign"][0])
            if "test" in role_ranges and "model_train_benign" in role_ranges:
                self.assertNotEqual(role_ranges["test"], role_ranges["model_train_benign"])

    def test_scaler_metadata_records_train_only_fit_source(self) -> None:
        self.assertEqual(self.data_manifest["scaler_fit_source"], "model_train_benign")

    def test_threshold_uses_calibration_scores_only(self) -> None:
        if self.summary.empty:
            raise unittest.SkipTest("pilot summary not available yet")
        for _, row in self.summary.iterrows():
            calib = pd.read_csv(ROOT / row["scores_calibration_csv"])
            threshold = threshold_from_benign_fpr(calib["fused_score"].to_numpy(dtype=np.float32), 0.25)
            self.assertAlmostEqual(float(threshold), float(row["threshold"]), places=6)

    def test_gan_and_wgan_configs_match_except_loss(self) -> None:
        gan_cfg = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "models" / "tcn_gan.yaml")
        wgan_cfg = load_yaml_like(ROOT / "journal_rebuild" / "configs" / "models" / "tcn_wgan_gp.yaml")
        same, diffs = compare_model_configs_except_loss(gan_cfg, wgan_cfg)
        self.assertTrue(same, f"unexpected config diffs: {diffs}")

    def test_checkpoint_metadata_complete(self) -> None:
        if self.summary.empty:
            raise unittest.SkipTest("pilot summary not available yet")
        required_keys = {
            "model_name",
            "loss_type",
            "seed",
            "epoch",
            "model_config",
            "data_config",
            "feature_names",
            "scaler_hash",
            "manifest_hash",
            "source_file_hashes",
            "git_commit",
            "timestamp",
            "optimizer_state",
            "model_state",
        }
        for _, row in self.summary.iterrows():
            payload = torch.load(ROOT / row["checkpoint"], map_location="cpu")
            self.assertTrue(required_keys.issubset(payload.keys()))

    def test_checkpoint_manifest_mismatch_is_rejected(self) -> None:
        if self.summary.empty:
            raise unittest.SkipTest("pilot summary not available yet")
        row = self.summary.iloc[0]
        src = ROOT / row["checkpoint"]
        payload = torch.load(src, map_location="cpu")
        payload["manifest_hash"] = "bad-hash"
        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            torch.save(payload, handle.name)
            with self.assertRaises(ValueError):
                load_checkpoint_validated(
                    Path(handle.name),
                    expected_model_name=str(payload["model_name"]),
                    expected_loss_type=str(payload["loss_type"]),
                    expected_manifest_hash=str(self.data_manifest["manifest_hash"]),
                    expected_scaler_hash=str(self.data_manifest["scaler_hash"]),
                )

    def test_score_direction_formula_is_consistent(self) -> None:
        if self.summary.empty:
            raise unittest.SkipTest("pilot summary not available yet")
        alpha = 0.24
        row = self.summary.iloc[0]
        scores = pd.read_csv(ROOT / row["scores_test_csv"])
        fused = alpha * scores["SD_normalized"].to_numpy(dtype=np.float32) + (1.0 - alpha) * scores["SF_normalized"].to_numpy(dtype=np.float32)
        self.assertTrue(np.allclose(fused, scores["fused_score"].to_numpy(dtype=np.float32), atol=1e-6))

    def test_repeat_metric_computation_is_deterministic(self) -> None:
        if self.summary.empty:
            raise unittest.SkipTest("pilot summary not available yet")
        row = self.summary.iloc[0]
        calib = pd.read_csv(ROOT / row["scores_calibration_csv"])
        test = pd.read_csv(ROOT / row["scores_test_csv"])
        th1 = threshold_from_benign_fpr(calib["fused_score"].to_numpy(dtype=np.float32), 0.25)
        th2 = threshold_from_benign_fpr(calib["fused_score"].to_numpy(dtype=np.float32), 0.25)
        self.assertAlmostEqual(th1, th2, places=8)
        m1 = metrics_at_threshold(test["label"].to_numpy(dtype=np.uint8), test["fused_score"].to_numpy(dtype=np.float32), th1)
        m2 = metrics_at_threshold(test["label"].to_numpy(dtype=np.uint8), test["fused_score"].to_numpy(dtype=np.float32), th2)
        self.assertEqual(m1, m2)


if __name__ == "__main__":
    unittest.main()
