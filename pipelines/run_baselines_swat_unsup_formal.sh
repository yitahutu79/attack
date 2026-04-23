#!/bin/bash
# SWaT formal anomaly-detection split for unsupervised/deep baselines.

set -eo pipefail

PY=${PY:-/Users/lijie/miniforge3/envs/attack/bin/python}
DATA=${DATA:-attack/dataset/SWaT}
OUT_ROOT=${OUT_ROOT:-attack/results/cross_dataset_formal_unsup/swat_baselines}
RUN_ID=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$OUT_ROOT/$RUN_ID"

WINDOW_SIZE=${WINDOW_SIZE:-128}
STRIDE=${STRIDE:-16}
ANOMALY_RATIO=${ANOMALY_RATIO:-0.15}
TARGET_FPR=${TARGET_FPR:-0.05}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-256}
UNSUP_TRAIN_FRACTION=${UNSUP_TRAIN_FRACTION:-0.6}
UNSUP_CALIB_FRACTION=${UNSUP_CALIB_FRACTION:-0.2}

mkdir -p "$OUT_DIR"

echo "=========================================="
echo "SWaT formal unsupervised baselines"
echo "=========================================="
echo "data:               $DATA"
echo "out:                $OUT_DIR"
echo "window/stride:      $WINDOW_SIZE / $STRIDE"
echo "target_fpr:         $TARGET_FPR"
echo "epochs:             $EPOCHS"
echo "train/calib frac:   $UNSUP_TRAIN_FRACTION / $UNSUP_CALIB_FRACTION"
echo ""

"$PY" attack/baselines/window_baselines.py \
  --dataset swat \
  --data-dir "$DATA" \
  --unsupervised-formal-split \
  --unsup-train-fraction "$UNSUP_TRAIN_FRACTION" \
  --unsup-calib-fraction "$UNSUP_CALIB_FRACTION" \
  --window-size "$WINDOW_SIZE" \
  --stride "$STRIDE" \
  --anomaly-ratio "$ANOMALY_RATIO" \
  --target-fpr "$TARGET_FPR" \
  --methods iforest ocsvm lstm_ae lstm_ad \
  --device cpu \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --output-dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/baselines.log"

echo ""
echo "Done. Key outputs:"
echo "  - $OUT_DIR/baseline_results.csv"
echo "  - $OUT_DIR/baseline_results.json"
echo "  - $OUT_DIR/baselines.log"
