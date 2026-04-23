#!/bin/bash
# TON_IoT formal chronological anomaly-detection baselines.

set -eo pipefail

PY=${PY:-/Users/lijie/miniforge3/envs/attack/bin/python}
DATA=${DATA:-attack/dataset/TON_loT}
OUT_ROOT=${OUT_ROOT:-attack/results/cross_dataset_formal_unsup/ton_iot_baselines}
RUN_ID=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$OUT_ROOT/$RUN_ID"

CHRONO_FILE=${CHRONO_FILE:-Processed_datasets/Processed_Linux_dataset/linux_memory1.csv}
WINDOW_SIZE=${WINDOW_SIZE:-128}
STRIDE=${STRIDE:-16}
ANOMALY_RATIO=${ANOMALY_RATIO:-0.15}
TARGET_FPR=${TARGET_FPR:-0.05}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-256}
CHRONO_TRAIN_FRACTION=${CHRONO_TRAIN_FRACTION:-0.6}
CHRONO_CALIB_FRACTION=${CHRONO_CALIB_FRACTION:-0.1}

mkdir -p "$OUT_DIR"

echo "=========================================="
echo "TON_IoT formal chronological baselines"
echo "=========================================="
echo "data:               $DATA"
echo "file:               $CHRONO_FILE"
echo "out:                $OUT_DIR"
echo "window/stride:      $WINDOW_SIZE / $STRIDE"
echo "target_fpr:         $TARGET_FPR"
echo "epochs:             $EPOCHS"
echo "train/calib frac:   $CHRONO_TRAIN_FRACTION / $CHRONO_CALIB_FRACTION"
echo ""

"$PY" attack/baselines/window_baselines.py \
  --dataset ton_iot \
  --data-dir "$DATA" \
  --chrono-unsupervised-split \
  --chrono-file "$CHRONO_FILE" \
  --chrono-train-fraction "$CHRONO_TRAIN_FRACTION" \
  --chrono-calib-fraction "$CHRONO_CALIB_FRACTION" \
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
