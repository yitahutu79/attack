#!/bin/bash
# UNSW-NB15 official train/test split for the proposed TCN-GAN detector.

set -euo pipefail

PY=${PY:-/Users/lijie/miniforge3/envs/attack/bin/python}
DATA=${DATA:-attack/dataset/UNSW-NB15}
OUT_ROOT=${OUT_ROOT:-attack/results/cross_dataset_formal_unsup/unsw_nb15_tcn}
EPOCHS=${EPOCHS:-8}
BATCH_SIZE=${BATCH_SIZE:-256}
TARGET_FPR=${TARGET_FPR:-0.05}

RUN_ID=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$OUT_ROOT/$RUN_ID"
mkdir -p "$OUT_DIR" attack/results/logs

echo "UNSW-NB15 TCN-GAN evaluation"
echo "data:       $DATA"
echo "out:        $OUT_DIR"
echo "epochs:     $EPOCHS"
echo "target_fpr: $TARGET_FPR"

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
"$PY" attack/pipelines/run_tcn_cross_dataset_minimal.py \
  --dataset unsw_nb15 \
  --data-dir "$DATA" \
  --train-files "Training and Testing Sets/UNSW_NB15_training-set.csv" \
  --test-files "Training and Testing Sets/UNSW_NB15_testing-set.csv" \
  --window-size 128 \
  --stride 16 \
  --anomaly-ratio 0.15 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --target-fpr "$TARGET_FPR" \
  --disc-pooling attn \
  --score-mode fused \
  --score-alpha 0.24 \
  --out-json "$OUT_DIR/ours.json" \
  --save-best "$OUT_DIR/ours_ckpt.pt" \
  2>&1 | tee "$OUT_DIR/ours.log"
