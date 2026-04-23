#!/bin/bash
# 最终 TCN-GAN 模型额外 seed 重跑
# 用法：bash attack/run_tcn_seed_runs.sh
#
# 说明：
#   默认补跑 seed=43,44。
#   每个 seed 训练一个 attn+wgan-gp 最终模型，并在 target_fpr=0.05 和 0.15 下评估。

set -euo pipefail

PY=${PY:-/Users/lijie/miniforge3/envs/attack/bin/python}
DATA=${DATA:-attack/dataset/CICIDS2017}
OUT_ROOT=${OUT_ROOT:-attack/results/final_experiments/20260414_132416/seed_runs}
SEEDS=${SEEDS:-43,44}

WINDOW_SIZE=${WINDOW_SIZE:-128}
STRIDE=${STRIDE:-16}
ANOMALY_RATIO=${ANOMALY_RATIO:-0.15}
EPOCHS=${EPOCHS:-12}
SCORE_ALPHA=${SCORE_ALPHA:-0.24}
FPR_GRID=${FPR_GRID:-0.05,0.15}

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

mkdir -p "$OUT_ROOT"

old_ifs=$IFS
IFS=,
for seed in $SEEDS; do
  IFS=$old_ifs
  OUT_DIR="$OUT_ROOT/seed_${seed}"
  mkdir -p "$OUT_DIR"
  CKPT="$OUT_DIR/ckpt_w${WINDOW_SIZE}_s${STRIDE}_seed${seed}.pt"
  TRAIN_LOG="$OUT_DIR/train_w${WINDOW_SIZE}_s${STRIDE}_seed${seed}.log"

  echo "=========================================="
  echo "训练 seed=$seed"
  echo "输出目录: $OUT_DIR"
  echo "=========================================="

  "$PY" attack/models/tcn_gan_experiment.py \
    --data-dir "$DATA" \
    --train-files "${TRAIN_FILES[@]}" \
    --test-files "${TEST_FILES[@]}" \
    --window-size "$WINDOW_SIZE" \
    --stride "$STRIDE" \
    --anomaly-ratio "$ANOMALY_RATIO" \
    --epochs "$EPOCHS" \
    --batch-size 256 \
    --test-batch-size 512 \
    --lr 0.0003 \
    --seed "$seed" \
    --disc-pooling attn \
    --gan-loss wgan-gp \
    --gp-lambda 10 \
    --n-critic 5 \
    --save-best "$CKPT" \
    > "$TRAIN_LOG" 2>&1

  IFS=,
  for fpr in $FPR_GRID; do
    fpr_clean=$(printf "%.3f" "$fpr")
    fpr_tag=${fpr_clean/./p}
    OUT_JSON="$OUT_DIR/eval_w${WINDOW_SIZE}_s${STRIDE}_seed${seed}_fpr${fpr_tag}.json"
    EVAL_LOG="$OUT_DIR/eval_w${WINDOW_SIZE}_s${STRIDE}_seed${seed}_fpr${fpr_tag}.log"

    echo "评估 seed=$seed target_fpr=$fpr_clean"
    "$PY" attack/models/tcn_gan_experiment.py \
      --data-dir "$DATA" \
      --train-files "${TRAIN_FILES[@]}" \
      --test-files "${TEST_FILES[@]}" \
      --window-size "$WINDOW_SIZE" \
      --stride "$STRIDE" \
      --anomaly-ratio "$ANOMALY_RATIO" \
      --load "$CKPT" \
      --eval-only \
      --disc-pooling attn \
      --gan-loss wgan-gp \
      --gp-lambda 10 \
      --n-critic 5 \
      --score-mode fused \
      --score-alpha "$SCORE_ALPHA" \
      --target-fpr "$fpr_clean" \
      --out-json "$OUT_JSON" \
      > "$EVAL_LOG" 2>&1
  done
done
IFS=$old_ifs

"$PY" -c '
import csv
import glob
import json
import os
import re
import sys

root = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(root, "seed_*", "eval_*.json"))):
    d = json.load(open(path, encoding="utf-8"))
    cal = d.get("calibrated", {})
    met = d.get("metrics", {})
    timing = d.get("timing", {})
    m = re.search(r"seed_(\d+)", path)
    rows.append({
        "seed": int(m.group(1)) if m else "",
        "target_fpr": cal.get("target_fpr", ""),
        "auc": met.get("auc", ""),
        "ap": met.get("ap", ""),
        "calib_f1": cal.get("f1", ""),
        "calib_recall": cal.get("recall", ""),
        "calib_precision": cal.get("precision", ""),
        "test_benign_fpr": cal.get("test_benign_fpr", ""),
        "eval_seconds": timing.get("eval_seconds", ""),
        "json": path,
    })
rows.sort(key=lambda r: (r["seed"], float(r["target_fpr"])))
out = os.path.join(root, "seed_summary.csv")
fields = ["seed", "target_fpr", "auc", "ap", "calib_f1", "calib_recall", "calib_precision", "test_benign_fpr", "eval_seconds", "json"]
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"CSV已保存: {out}")
' "$OUT_ROOT"

echo ""
echo "完成。汇总表：$OUT_ROOT/seed_summary.csv"
