# attack (ACSAC artifact workspace)

This repository is the cleaned workspace for the paper:

**Calibrated and Auditable Window-Level Network Intrusion Detection with TCN-WGAN-GP**

The current repository scope is artifact-oriented:
- reproducible experiment scripts,
- paper source (`paper_acsac/`),
- compact result artifacts used by the manuscript.

## Current structure

- `paper_acsac/`: ACSAC paper source (tex/figures/tables).
- `experiments/`: script entry points for compact baselines, threshold sweep, XAI trace, and sensitivity plots.
- `models/`, `baselines/`, `dataset_loaders.py`: model/baseline/dataset utilities.
- `results/cicids_strict_mps/`: main CICIDS2017 strict-protocol outputs used by paper tables/figures.
- `results/current_paper_compact_baselines/`: protocol-compatible baseline summaries.
- `results/protocol_checks/`: strict split/protocol consistency checks.

## Environment

Use one of the following:

1. Conda (recommended)
```bash
conda env create -f environment.lock.yml
conda activate attack
```

2. Minimal conda spec
```bash
conda env create -f environment.yml
conda activate attack
```

3. Pip lock (if needed in your own env)
```bash
pip install -r requirements-lock.txt
```

## Notes

- Large raw datasets and archived intermediate materials are intentionally excluded from Git tracking.
- The paper’s primary evidence chain is CICIDS2017 under strict no-leakage roles; SWaT/UNSW-NB15 are supplementary operating-boundary checks.
