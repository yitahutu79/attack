#!/bin/bash
# UNSW-NB15 official train/test split for window-level baselines and SOTA models.

set -euo pipefail

PY=${PY:-/Users/lijie/miniforge3/envs/attack/bin/python}
DATA=${DATA:-attack/dataset/UNSW-NB15}
OUT_DIR=${OUT_DIR:-attack/results/sota_unsw_nb15}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-128}
METHODS=${METHODS:-iforest ocsvm deepsvdd tranad ganomaly lstm_ae lstm_ad}

mkdir -p "$OUT_DIR" attack/results/logs

echo "UNSW-NB15 baseline evaluation"
echo "data:    $DATA"
echo "out:     $OUT_DIR"
echo "epochs:  $EPOCHS"
echo "methods: $METHODS"

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
"$PY" attack/baselines/window_baselines.py \
  --dataset unsw_nb15 \
  --data-dir "$DATA" \
  --train-files "Training and Testing Sets/UNSW_NB15_training-set.csv" \
  --test-files "Training and Testing Sets/UNSW_NB15_testing-set.csv" \
  --window-size 128 \
  --stride 16 \
  --anomaly-ratio 0.15 \
  --target-fpr 0.05 \
  --target-fpr-grid 0.15 \
  --methods $METHODS \
  --device cpu \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --scaler standard \
  --output-dir "$OUT_DIR" \
  --out-log attack/results/logs/unsw_baselines.log
