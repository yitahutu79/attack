# CAPAD: reproducibility artifacts

Code, configurations and saved experimental results for:

**A Detector-Agnostic, Deployment-Consistent Evaluation and Audit Framework for Window-Level Network Intrusion Detection**

CAPAD means **Calibrated and Auditable Protocol for Anomaly Detection**.
This repository supports the CICIDS2017 evaluation and the bounded TON_IoT
external protocol check reported in the manuscript. It is not a new raw
network-traffic dataset and does not claim that the manuscript has been accepted.

## Start here: reproduce metrics without training

Use Python 3.10 or newer in a dedicated environment:

```bash
git clone https://github.com/yitahutu79/attack.git
cd attack
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-audit.txt
python scripts/verify_artifact_integrity.py
python scripts/verify_saved_results.py
```

On Windows activate the environment with `.venv\Scripts\activate` instead.
The score verification needs only NumPy and pandas: no original dataset,
checkpoint download, GPU or model training is required. It reads the included
`.csv.gz` files directly and writes to `verification/recomputed/`.

It independently recomputes the thresholds, confusion counts, AUC, AP,
precision, recall, F1 and FPR for 10 main runs and four external detector
outputs, and reconstructs Tables 2, 6 and 7. It exits with an error if the
main or external metric discrepancy exceeds 1e-6. This verifies saved scores,
not the entire process from raw traffic to trained models.

## Protocol and limitations

- Main TCN runs: seeds 0–4 for TCN-GAN and TCN-WGAN-GP, W=128, stride=16,
  eight epochs. The `pilot_*_seed0` directories contain the full-size seed-0
  runs used with `formal_*_seed1` through `formal_*_seed4`.
- The actual target benign calibration FPR is **beta=0.25**, not 0.05.
  The threshold is the linearly interpolated 0.75-quantile after float32
  conversion; alarms use **score >= threshold**. This is not a guarantee of
  a 25% future-test FPR.
- The default fusion weight is **alpha=0.24**. We report the saved setting
  without claiming optimality or independent evidence of its original
  selection. The weight sweep uses test labels and is post-hoc sensitivity
  analysis. Oracle thresholds also use test labels and are diagnostic only.
- CICIDS2017 reference baselines use seed 42: LSTM-AE trains for eight epochs,
  GANomaly for 30. Saved configuration/result records are included; their
  model outputs are not independently reconstructed by the quick verifier.
- TON_IoT is a bounded chronological stress check, not a universal
  cross-dataset generalization claim. The reused TCN case uses an additional
  `reference_benign` segment for score references. The three other detectors
  use model-training benign data. High external benign FPR is retained.
- Table 7 follows the original offline float64 comparison path; the main
  evaluation uses float32 comparisons. Ties can affect counts.
- Means and sample standard deviations use five TCN runs; the window-size
  sensitivity comparison uses seed 1. Hardware-dependent timing is recorded
  evidence, not a runtime guarantee for another machine.

## Repository contents

| Path | Purpose |
| --- | --- |
| `journal_rebuild/src/` | Data, model, score, threshold and metric implementation |
| `journal_rebuild/configs/` | Dataset, experiment and detector configurations |
| `journal_rebuild/scripts/` | Data builds, training and numerical analyses |
| `journal_rebuild/runs/scores/` | Compressed calibration/test scores for main and sensitivity runs |
| `journal_rebuild/runs/metrics/` | Saved numerical summaries and per-run metrics |
| `journal_rebuild/runs/checkpoints/*/resolved_config.yaml` | Configurations only; no serialized model binaries |
| `scripts/run_external_protocol_check.py` | External protocol implementation |
| `runs/external_protocol_check/TON_IoT/` | Compressed external scores and saved metrics |
| `reports/external_protocol_check/` | Split descriptions and result tables |
| `results/current_paper_compact_baselines/` | CICIDS2017 saved baseline configuration/results |
| `artifacts/` | Source hashes, numerical table mapping and verified summaries |

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for commands and scope and
[DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for original dataset sources.
The `.csv.gz` artifacts are gzip-compressed CSV, not Git LFS pointers.

## Publication boundary

Raw datasets, processed feature matrices, checkpoints, manuscripts, Word
documents, intermediate paper drafts, downloaded literature, presentations,
and unrelated follow-up projects are intentionally omitted from this revision.
TON_IoT exported scores omit the source/destination entity-summary columns;
the labels, scores, thresholds and file/row provenance needed for evaluation
are retained. These exports are not the full analyst interface.

Earlier commits concern an ACSAC-oriented experiment generation. They must
not be mixed with this manuscript's beta=0.25 results. Removing paper files
from the latest tree does not erase old Git history.

This publication preparation did not retrain models or modify the original
experimental records. Only portable import paths and release/verification
utilities were changed. See `artifacts/source_manifest.json` for source and
published-file hashes and recorded transformations.

No software license has yet been designated by the authors. Public visibility
is not an unrestricted reuse license; third-party data and dependencies retain
their own terms. For citation use [CITATION.cff](CITATION.cff), identifying the
commit used. Manuscript acceptance, a DOI and an archival release are not implied.
