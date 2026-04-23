#!/bin/bash
# TON_IoT formal chronological anomaly-detection split for TCN.

set -eo pipefail

PY=${PY:-/Users/lijie/miniforge3/envs/attack/bin/python}
DATA=${DATA:-attack/dataset/TON_loT}
OUT_ROOT=${OUT_ROOT:-attack/results/cross_dataset_formal_unsup/ton_iot_tcn}
RUN_ID=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$OUT_ROOT/$RUN_ID"

CHRONO_FILE=${CHRONO_FILE:-Processed_datasets/Processed_Linux_dataset/linux_memory1.csv}
WINDOW_SIZE=${WINDOW_SIZE:-128}
STRIDE=${STRIDE:-16}
ANOMALY_RATIO=${ANOMALY_RATIO:-0.15}
TARGET_FPR=${TARGET_FPR:-0.05}
EPOCHS=${EPOCHS:-8}
BATCH_SIZE=${BATCH_SIZE:-256}
CHRONO_TRAIN_FRACTION=${CHRONO_TRAIN_FRACTION:-0.6}
CHRONO_CALIB_FRACTION=${CHRONO_CALIB_FRACTION:-0.1}
TCN_GAN_DISABLE_WEIGHT_NORM=${TCN_GAN_DISABLE_WEIGHT_NORM:-1}

mkdir -p "$OUT_DIR"

echo "=========================================="
echo "TON_IoT formal chronological TCN evaluation"
echo "=========================================="
echo "data:               $DATA"
echo "file:               $CHRONO_FILE"
echo "out:                $OUT_DIR"
echo "window/stride:      $WINDOW_SIZE / $STRIDE"
echo "target_fpr:         $TARGET_FPR"
echo "epochs:             $EPOCHS"
echo "train/calib frac:   $CHRONO_TRAIN_FRACTION / $CHRONO_CALIB_FRACTION"
echo ""

TCN_GAN_DISABLE_WEIGHT_NORM="$TCN_GAN_DISABLE_WEIGHT_NORM" "$PY" attack/pipelines/run_tcn_cross_dataset_minimal.py \
  --dataset ton_iot \
  --data-dir "$DATA" \
  --chrono-unsupervised-split \
  --chrono-file "$CHRONO_FILE" \
  --chrono-train-fraction "$CHRONO_TRAIN_FRACTION" \
  --chrono-calib-fraction "$CHRONO_CALIB_FRACTION" \
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
