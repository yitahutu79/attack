# Reproduction levels

## 1. Saved-score verification (tested for this release)

```bash
python -m pip install -r requirements-audit.txt
python scripts/verify_artifact_integrity.py
python scripts/verify_saved_results.py
```

Expected numerical discrepancies relative to saved metrics are about 1.5e-8
for the ten main TCN runs and 2.3e-16 for the four TON_IoT outputs. The final
checks allow an absolute tolerance of 1e-6. The verifier uses independent NumPy
logic, not the original model-evaluation functions. Output is written under
`verification/recomputed/`; archived verification summaries are under
`artifacts/verified_results/`.

The verifier covers threshold computation, fused score consistency,
confusion counts, AUC/AP and classification metrics. It also reconstructs the
Table 6 calibration/Oracle comparison and Table 7 endpoint fusion ablation.
It does not independently reproduce the three main baseline models' scores
or retrain the window-length experiments.

To use the original offline analysis script, first expand CSVs:

```bash
python scripts/unpack_saved_artifacts.py
python -m pip install matplotlib
python journal_rebuild/scripts/run_offline_ablation_analysis.py
```

This analysis writes regenerated outputs under `journal_rebuild/runs/metrics/`
and `journal_rebuild/reports/ablation/`. Run it in a separate working copy if you
need to preserve the distributed result files byte for byte. It uses test
labels for Oracle and alpha sensitivity; do not call it independent validation
for hyperparameter selection.

## 2. Raw-data reconstruction and training (not rerun during release preparation)

Use a **separate clone** and Python 3.10. Training writes model files and result
records into this project's run directories. Do not rerun it on top of the
only copy of the saved evidence.

```bash
python -m pip install -r requirements-training.txt
python journal_rebuild/scripts/build_cicids2017.py
python journal_rebuild/scripts/run_multiseed_model_selection.py --device cpu
```

Obtain the raw CSVs first, as described in DATA_AVAILABILITY.md. The data-build
step is required: metadata distributed in the repository is not a replacement
for processed feature arrays. GPU users may choose `--device cuda` or
`--device mps` when supported. Published timing measurements used MPS.
Hardware, library and serialization differences can prevent bitwise agreement.
The formal runner checks fixed manifest/scaler hashes; a mismatch must be
investigated rather than silently changing the expected hash.

The code also contains a smoke-plus-pilot entry point:

```bash
python journal_rebuild/scripts/run_pilot_seed0.py
```

This runs training and may overwrite the pilot output files; it is not the
lightweight score verifier. The model configs are JSON-compatible YAML files.

## 3. Main baselines and sensitivity

The baseline entry point is
`experiments/run_current_paper_compact_baselines.py`. Run it with `--help` to
inspect options. The exact saved per-row configuration and historical command
line are in `results/current_paper_compact_baselines/cicids2017_compact_baselines.csv`.
Adapt machine-specific executable paths before rerunning. Do not use a single
default command to claim all three reference runs have been reproduced:
LSTM-AE used eight epochs and GANomaly used 30, with seed 42 and beta=0.25.

The window-sensitivity entry point is:

```bash
python journal_rebuild/scripts/run_window_sensitivity_delay_bot.py --device cpu
```

It performs training and analysis for the seed-1 sensitivity experiment; it
was not rerun during publication preparation. The saved W=64/W=256 scores
and metrics are included along with the main W=128 seed-1 result.

## 4. TON_IoT external protocol

After obtaining the processed network CSV variant:

```bash
python journal_rebuild/scripts/build_ton_iot_candidate1.py
python scripts/run_external_protocol_check.py --device cpu --skip-existing-tcn
```

This trains/evaluates the three standalone baseline families. The optional
TCN output is a reuse of the saved pilot chain, not a newly fitted fourth
baseline with identical reference normalization. To reconstruct that chain,
inspect and run `journal_rebuild/scripts/run_ton_iot_candidate1_seed0.py` in a
separate working copy. Without `--skip-existing-tcn`, the external wrapper can
reuse previously saved pilot score CSVs, expanded from the provided gzip files.

Main TON_IoT split counts are 12,549 training benign, 2,607 calibration benign
and 21,181 test windows. The optional TCN has a separate 357-window benign
reference segment. The test interval contains 1,152 benign and 20,029 attack
windows. This stress check exposes high external benign false-positive rates;
those results are not removed or optimized away.

## 5. Scope of validation

The release preparation ran score verification and artifact hashes. It did not
retrain the models, validate every original processing step or prove that the
original alpha choice was independent of test observations. Protocol tests in
`journal_rebuild/tests/` require locally rebuilt manifests and checkpoints;
some skip or fail if those training artifacts have not been produced.

Machine-local absolute paths in historical command/configuration records are
retained as evidence, not as setup instructions. Publication code changes are
limited to folder-independent imports and release verification utilities.

The training requirements are transcribed from the historical repository lock,
not independently established as the exact environment of every saved run.
They were not freshly installed during release preparation. CLI import checks
used an existing Python 3.10.20 environment with NumPy 1.26.4, pandas 2.3.3,
scikit-learn 1.7.2 and PyTorch 2.11.0; this differs from the historical PyTorch
2.2.2 pin. The score verifier was also run independently under Python 3.12
with NumPy 2.3.5 and pandas 2.2.3. Successful import/score checks are not a
full training-environment validation.
