#!/bin/bash
# Formal SWaT TCN evaluation with the same mixed split used by the baseline table.

set -eo pipefail

PY=${PY:-/Users/lijie/miniforge3/envs/attack/bin/python}
DATA=${DATA:-attack/dataset/SWaT}
OUT_ROOT=${OUT_ROOT:-attack/results/cross_dataset_formal/swat}
RUN_ID=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$OUT_ROOT/$RUN_ID"

WINDOW_SIZE=${WINDOW_SIZE:-128}
STRIDE=${STRIDE:-16}
ANOMALY_RATIO=${ANOMALY_RATIO:-0.15}
TARGET_FPR=${TARGET_FPR:-0.05}
EPOCHS=${EPOCHS:-8}
BATCH_SIZE=${BATCH_SIZE:-256}
MIXED_TRAIN_FRACTION=${MIXED_TRAIN_FRACTION:-0.5}
TCN_GAN_DISABLE_WEIGHT_NORM=${TCN_GAN_DISABLE_WEIGHT_NORM:-1}

mkdir -p "$OUT_DIR"

echo "=========================================="
echo "Formal SWaT TCN evaluation"
echo "=========================================="
echo "data:               $DATA"
echo "out:                $OUT_DIR"
echo "window/stride:      $WINDOW_SIZE / $STRIDE"
echo "target_fpr:         $TARGET_FPR"
echo "epochs:             $EPOCHS"
echo "mixed_train_frac:   $MIXED_TRAIN_FRACTION"
echo ""

TCN_GAN_DISABLE_WEIGHT_NORM="$TCN_GAN_DISABLE_WEIGHT_NORM" "$PY" attack/pipelines/run_tcn_cross_dataset_minimal.py \
  --dataset swat \
  --data-dir "$DATA" \
  --supervised-mixed-split \
  --mixed-train-fraction "$MIXED_TRAIN_FRACTION" \
  --window-size "$WINDOW_SIZE" \
  --stride "$STRIDE" \
  --anomaly-ratio "$ANOMALY_RATIO" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --test-batch-size "$BATCH_SIZE" \
  --target-fpr "$TARGET_FPR" \
  --save-best "$OUT_DIR/ours_ckpt.pt" \
  --out-json "$OUT_DIR/ours.json" \
  2>&1 | tee "$OUT_DIR/ours.log"

echo ""
echo "Done. Key outputs:"
echo "  - $OUT_DIR/ours.json"
echo "  - $OUT_DIR/ours.log"
echo "  - $OUT_DIR/ours_ckpt.pt"
