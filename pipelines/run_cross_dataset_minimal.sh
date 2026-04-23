#!/bin/bash
# Minimal cross-dataset validation for the A-tier revision plan.
#
# Default target: SWaT 5-second aggregate CSVs already present in attack/dataset/SWaT.
# This runs:
#   1) Attentive TCN-WGAN-GP on the second dataset
#   2) two classical baselines with the same window protocol
#
# Usage from /Users/lijie/Desktop/work:
#   bash attack/pipelines/run_cross_dataset_minimal.sh
#
# You can override variables, for example:
#   PY=python DATASET=ton_iot DATA=attack/dataset/TON_loT \
#   TRAIN_FILES="Processed_datasets/Processed_IoT_dataset/IoT_Fridge.csv" \
#   TEST_FILES="Processed_datasets/Processed_IoT_dataset/IoT_Modbus.csv" \
#   bash attack/pipelines/run_cross_dataset_minimal.sh

set -eo pipefail

PY=${PY:-/Users/lijie/miniforge3/envs/attack/bin/python}
DATASET=${DATASET:-swat}
DATA=${DATA:-attack/dataset/SWaT}
OUT_ROOT=${OUT_ROOT:-attack/results/cross_dataset_minimal}
RUN_ID=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$OUT_ROOT/$DATASET/$RUN_ID"

WINDOW_SIZE=${WINDOW_SIZE:-128}
STRIDE=${STRIDE:-16}
ANOMALY_RATIO=${ANOMALY_RATIO:-0.15}
TARGET_FPR=${TARGET_FPR:-0.05}
EPOCHS=${EPOCHS:-8}
BATCH_SIZE=${BATCH_SIZE:-256}
DEVICE=${DEVICE:-mps}
TCN_DEVICE=${TCN_DEVICE:-cpu}
TCN_GAN_DISABLE_WEIGHT_NORM=${TCN_GAN_DISABLE_WEIGHT_NORM:-1}
GAN_LOSS=${GAN_LOSS:-vanilla}
N_CRITIC=${N_CRITIC:-1}
SCORE_MODE=${SCORE_MODE:-prob}
SCORE_ALPHA=${SCORE_ALPHA:-0.24}

# Empty TRAIN_FILES/TEST_FILES means the loader will use dataset defaults when available.
TRAIN_FILES=${TRAIN_FILES:-}
TEST_FILES=${TEST_FILES:-}

TRAIN_ARGS=()
TEST_ARGS=()
if [ -n "$TRAIN_FILES" ]; then
  read -r -a TRAIN_ARR <<< "$TRAIN_FILES"
  TRAIN_ARGS=(--train-files "${TRAIN_ARR[@]}")
fi
if [ -n "$TEST_FILES" ]; then
  read -r -a TEST_ARR <<< "$TEST_FILES"
  TEST_ARGS=(--test-files "${TEST_ARR[@]}")
fi

mkdir -p "$OUT_DIR"

echo "=========================================="
echo "Cross-dataset minimal validation"
echo "=========================================="
echo "dataset:       $DATASET"
echo "data:          $DATA"
echo "out:           $OUT_DIR"
echo "window/stride: $WINDOW_SIZE / $STRIDE"
echo "target_fpr:    $TARGET_FPR"
echo "gan_loss:      $GAN_LOSS"
echo "score_mode:    $SCORE_MODE"
echo ""

echo "[1/3] Listing available files"
TCN_GAN_DISABLE_WEIGHT_NORM="$TCN_GAN_DISABLE_WEIGHT_NORM" "$PY" attack/models/tcn_gan_experiment.py \
  --dataset "$DATASET" \
  --data-dir "$DATA" \
  --list-files \
  > "$OUT_DIR/files.txt"
sed -n '1,80p' "$OUT_DIR/files.txt"

echo ""
echo "[2/3] Running minimal Attentive TCN-GAN"
TCN_GAN_DISABLE_WEIGHT_NORM="$TCN_GAN_DISABLE_WEIGHT_NORM" "$PY" attack/pipelines/run_tcn_cross_dataset_minimal.py \
  --dataset "$DATASET" \
  --data-dir "$DATA" \
  "${TRAIN_ARGS[@]}" \
  "${TEST_ARGS[@]}" \
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
echo "[3/3] Running classical baselines"
"$PY" attack/baselines/window_baselines.py \
  --dataset "$DATASET" \
  --data-dir "$DATA" \
  "${TRAIN_ARGS[@]}" \
  "${TEST_ARGS[@]}" \
  --window-size "$WINDOW_SIZE" \
  --stride "$STRIDE" \
  --anomaly-ratio "$ANOMALY_RATIO" \
  --target-fpr "$TARGET_FPR" \
  --methods iforest ocsvm \
  --device "$DEVICE" \
  --output-dir "$OUT_DIR/baselines" \
  2>&1 | tee "$OUT_DIR/baselines.log"

echo ""
echo "Done. Key outputs:"
echo "  - $OUT_DIR/ours.json"
echo "  - $OUT_DIR/baselines/baseline_results.csv"
echo "  - $OUT_DIR/files.txt"
