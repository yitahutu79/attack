#!/bin/bash
# Baseline实验重跑脚本
# 用法：bash attack/run_baseline.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Baseline实验重跑脚本"
echo "=========================================="
echo ""

# 设置环境变量
PY=/Users/lijie/miniforge3/envs/attack/bin/python
DATA=attack/dataset/CICIDS2017

# 数据文件
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

# 创建输出目录
OUT_DIR=attack/results/baseline_window_$(date +%Y%m%d_%H%M%S)
mkdir -p $OUT_DIR

echo "输出目录：$OUT_DIR"
echo ""

# 显示参数
echo "实验参数："
echo "  - window_size: 128"
echo "  - stride: 16"
echo "  - anomaly_ratio: 0.15"
echo "  - target_fpr: 0.05"
echo "  - train_benign_only: True"
echo ""

echo "训练文件："
for f in "${TRAIN_FILES[@]}"; do
  echo "  - $f"
done
echo ""

echo "测试文件："
for f in "${TEST_FILES[@]}"; do
  echo "  - $f"
done
echo ""

echo "=========================================="
echo "开始运行baseline实验..."
echo "=========================================="
echo ""

# 运行完整窗口级 baseline 对比
$PY attack/baselines/window_baselines.py \
  --data-dir $DATA \
  --train-files ${TRAIN_FILES[@]} \
  --test-files ${TEST_FILES[@]} \
  --window-size 128 \
  --stride 16 \
  --anomaly-ratio 0.15 \
  --target-fpr 0.05 \
  --methods iforest ocsvm rf mlp lstm_ae lstm_ad \
  --device mps \
  --epochs 30 \
  --batch-size 256 \
  --output-dir $OUT_DIR \
  | tee $OUT_DIR/baseline.log

echo ""
echo "=========================================="
echo "实验完成！"
echo "=========================================="
echo ""
echo "结果文件："
echo "  - $OUT_DIR/baseline_results.csv"
echo "  - $OUT_DIR/baseline_results.json"
echo "  - $OUT_DIR/baseline.log"
echo ""

# 显示结果摘要
if [ -f "$OUT_DIR/baseline_results.csv" ]; then
  echo "结果摘要："
  echo "----------------------------------------"
  cat $OUT_DIR/baseline_results.csv
  echo "----------------------------------------"
  echo ""
fi

echo "完成时间：$(date)"
