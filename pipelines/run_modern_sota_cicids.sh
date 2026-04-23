#!/bin/bash
# Run modern SOTA baselines (DeepSVDD + TranAD) on CICIDS2017 at target FPR 0.05 and 0.15.

set -euo pipefail

PY=${PY:-/Users/lijie/miniforge3/envs/attack/bin/python}
DATA=${DATA:-attack/dataset/CICIDS2017}
OUT_DIR=${OUT_DIR:-attack/results/sota_modern_cicids}
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-128}

TRAIN_FILES=(
  Tuesday-WorkingHours.pcap_ISCX.csv
  Wednesday-workingHours.pcap_ISCX.csv
  Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
  Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
)

TEST_FILES=(
  Friday-WorkingHours-Morning.pcap_ISCX.csv
  Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
  Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
)

mkdir -p "$OUT_DIR"

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
"$PY" attack/baselines/window_baselines.py \
  --dataset cicids2017 \
  --data-dir "$DATA" \
  --train-files "${TRAIN_FILES[@]}" \
  --test-files "${TEST_FILES[@]}" \
  --window-size 128 \
  --stride 16 \
  --anomaly-ratio 0.15 \
  --target-fpr 0.05 \
  --target-fpr-grid 0.15 \
  --methods deepsvdd tranad \
  --device cpu \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --output-dir "$OUT_DIR"
