#!/bin/bash
# 完整窗口级 baseline 的目标误报率扫描实验
# 用法：bash attack/run_baseline_fpr_sweep.sh
#
# 说明：
#   这个脚本调用正式 baseline 脚本 window_baselines.py。
#   每个模型只训练一次，然后在多个 target_fpr 下重新计算阈值和指标。

set -euo pipefail

PY=${PY:-/Users/lijie/miniforge3/envs/attack/bin/python}
DATA=${DATA:-attack/dataset/CICIDS2017}
OUT_ROOT=${OUT_ROOT:-attack/results/baseline_fpr_sweep}
RUN_ID=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$OUT_ROOT/$RUN_ID"
FPR_GRID=${FPR_GRID:-0.01,0.03,0.05,0.10,0.15,0.20,0.30}

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

echo "=========================================="
echo "完整窗口级 baseline FPR sweep"
echo "=========================================="
echo "输出目录: $OUT_DIR"
echo "FPR_GRID: $FPR_GRID"
echo ""

"$PY" attack/baselines/window_baselines.py \
  --data-dir "$DATA" \
  --train-files "${TRAIN_FILES[@]}" \
  --test-files "${TEST_FILES[@]}" \
  --window-size 128 \
  --stride 16 \
  --anomaly-ratio 0.15 \
  --target-fpr 0.05 \
  --target-fpr-grid "$FPR_GRID" \
  --methods iforest ocsvm rf mlp lstm_ae lstm_ad \
  --device mps \
  --epochs 30 \
  --batch-size 256 \
  --output-dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/baseline_fpr_sweep.log"

echo ""
echo "完成。主要结果："
echo "  - $OUT_DIR/baseline_results.csv"
echo "  - $OUT_DIR/baseline_results.json"
echo "  - $OUT_DIR/baseline_fpr_sweep.log"
