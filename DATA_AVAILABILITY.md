# Data availability and provenance

This repository shares code, configuration, derived window scores, thresholds,
result summaries and selected provenance records produced by the CAPAD study.
It does **not** claim original collection or ownership of CICIDS2017 or TON_IoT.

## Original benchmark sources

- CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
  Use the labeled flow CSV files named in
  `journal_rebuild/configs/data/cicids2017.yaml`, placed in `dataset/CICIDS2017/`.
  Cite I. Sharafaldin, A. H. Lashkari and A. A. Ghorbani, *Toward Generating a New
  Intrusion Detection Dataset and Intrusion Traffic Characterization*, ICISSP 2018.
- TON_IoT: https://research.unsw.edu.au/projects/toniot-datasets
  Use the processed network CSV variant specified in
  `journal_rebuild/configs/data/ton_iot_candidate1.yaml`, under
  `dataset/TON_IoT/Processed_datasets/Processed_Network_dataset/`.
  Cite the dataset authors and the network-dataset paper referenced by the
  official site; do not substitute telemetry-only data for network records.

Obtain each dataset from its provider and follow its terms and attribution
requirements. Original captures, full flow tables and processed feature arrays
are not redistributed here. File hashes, feature names and split roles in the
saved manifests identify the benchmark variants actually used.

## Included derived evidence

- Main TCN calibration/test score CSVs, losslessly gzip-compressed.
- Main-run model configuration JSON/YAML and metrics.
- Two additional seed-1 window-size experiments (64 and 256).
- Four TON_IoT detector outputs, excluding `src_entity_summary` and
  `dst_entity_summary` from the public score exports. All other CSV columns
  and numeric strings are preserved.
- CICIDS2017 baseline result/configuration records and the chronological window
  manifest. Model binaries, full raw records and private source paths are not
  needed by the score-only verifier.

`artifacts/source_manifest.json` records original-file SHA-256, published-file
SHA-256, compressed/uncompressed representations and any transformations.
Some unchanged historical records contain the author's original machine paths
or command lines; those strings are provenance, not portable execution commands.
Use the commands in REPRODUCIBILITY.md instead.

## Availability limits

The quick verifier does not retrain models, regenerate raw windows, establish
the historical independence of alpha selection, or certify absence of all forms
of data leakage. Checkpoints and raw datasets are not included in this release.
The baseline summary is not a substitute for independently regenerated baseline
predictions. The optional TON_IoT TCN case uses a separate benign reference pool.

These are study-generated reproducibility artifacts based on third-party
benchmark data, not a newly collected traffic dataset. No DOI or permanent
external data archive has been created by this publication step.
