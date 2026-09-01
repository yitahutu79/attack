# Manuscript-to-artifact mapping

The table numbers below refer to the CAPAD journal manuscript, not the older
ACSAC paper visible in previous commits. Stored scores use beta=0.25 and the
default fusion weight alpha=0.24.

| Manuscript item | Saved source / verification route |
| --- | --- |
| Table 2 | `journal_rebuild/runs/scores/formal_tcn_wgan_gp_seed1/scores_test.csv.gz`; first TN `test:082957` and first TP `test:096061`; see `verified_results/table2_records.csv` |
| Table 3 | `journal_rebuild/data/manifests/data_manifest.json` and `canonical_split_manifest.csv.gz`; split logic in `journal_rebuild/src/datasets/cicids2017.py` |
| Table 4 | Attack-family and window-label metadata in the main manifest / test score exports |
| Table 5, TCN rows | `journal_rebuild/runs/metrics/*/metrics.json`; ten `pilot`/`formal` score pairs; `verified_results/main_summary.csv` |
| Table 5, IF/AE/GANomaly | `results/current_paper_compact_baselines/cicids2017_compact_baselines.csv`; saved reference runs, not independently rescored by the verifier |
| Table 6 | `verified_results/table6.csv`; per-run calibration and labeled-test F1 Oracle, then five-run means |
| Table 7 | `verified_results/table7.csv`; SD-only, SF-only and alpha=0.24, recalibrated per run; original offline float64 comparison convention |
| Table 8 | `journal_rebuild/reports/model_selection/table_efficiency.csv` and the ten main metric JSON files; seconds, mean/sample SD |
| Tables 9–10 / Figure 5 data | `journal_rebuild/runs/metrics/window_sensitivity_w64_tcn_wgan_gp_seed1/metrics.json`, W=256 equivalent, and `formal_tcn_wgan_gp_seed1`; single seed, no retraining during release preparation |
| Table 11 | `reports/external_protocol_check/table_external_protocol_check.csv`, four `runs/external_protocol_check/TON_IoT/*/metrics.json` files and compressed window scores; `verified_results/ton_iot.csv` |

Figure artwork and manuscripts are intentionally not part of this public
repository revision. Numerical sources are retained. The two Table 2 records
are selected by file order, not a representative sample of overall performance.
