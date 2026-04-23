#!/bin/bash
# 最终 TCN-GAN 模型的目标误报率扫描实验
# 用法：bash attack/run_tcn_fpr_sweep.sh
#
# 说明：
#   不重新训练，只加载最终 checkpoint，在多个 target_fpr 下重新标定阈值并评估。

set -euo pipefail

PY=${PY:-/Users/lijie/miniforge3/envs/attack/bin/python}
DATA=${DATA:-attack/dataset/CICIDS2017}
RUN_ROOT=${RUN_ROOT:-attack/results/final_experiments/20260414_132416}
OUT_DIR=${OUT_DIR:-$RUN_ROOT/tcn_fpr_sweep}

# 默认使用最终消融中的 attn+wgan-gp 新模型 checkpoint。
# 如果要换模型，可以运行：
#   CKPT=path/to/ckpt.pt bash attack/run_tcn_fpr_sweep.sh
CKPT=${CKPT:-$RUN_ROOT/ablation_2x2/20260414_204232/20260414_225204/ckpt_w128_s16.pt}

WINDOW_SIZE=${WINDOW_SIZE:-128}
STRIDE=${STRIDE:-16}
ANOMALY_RATIO=${ANOMALY_RATIO:-0.15}
SCORE_ALPHA=${SCORE_ALPHA:-0.24}
SCORE_MODE=${SCORE_MODE:-fused}
DISC_POOLING=${DISC_POOLING:-attn}
GAN_LOSS=${GAN_LOSS:-wgan-gp}
GP_LAMBDA=${GP_LAMBDA:-10}
N_CRITIC=${N_CRITIC:-5}
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

if [ ! -f "$CKPT" ]; then
  echo "找不到 checkpoint: $CKPT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "=========================================="
echo "TCN-GAN FPR sweep"
echo "=========================================="
echo "checkpoint: $CKPT"
echo "输出目录: $OUT_DIR"
echo "FPR_GRID: $FPR_GRID"
echo "window=$WINDOW_SIZE stride=$STRIDE alpha=$SCORE_ALPHA"
echo ""

old_ifs=$IFS
IFS=,
for fpr in $FPR_GRID; do
  fpr_clean=$(printf "%.3f" "$fpr")
  tag=${fpr_clean/./p}
  out_json="$OUT_DIR/eval_w${WINDOW_SIZE}_s${STRIDE}_${SCORE_MODE}_fpr${tag}.json"
  log_path="$OUT_DIR/eval_w${WINDOW_SIZE}_s${STRIDE}_${SCORE_MODE}_fpr${tag}.log"

  echo "评估 target_fpr=$fpr_clean"
  "$PY" attack/models/tcn_gan_experiment.py \
    --data-dir "$DATA" \
    --train-files "${TRAIN_FILES[@]}" \
    --test-files "${TEST_FILES[@]}" \
    --window-size "$WINDOW_SIZE" \
    --stride "$STRIDE" \
    --anomaly-ratio "$ANOMALY_RATIO" \
    --load "$CKPT" \
    --eval-only \
    --disc-pooling "$DISC_POOLING" \
    --gan-loss "$GAN_LOSS" \
    --gp-lambda "$GP_LAMBDA" \
    --n-critic "$N_CRITIC" \
    --score-mode "$SCORE_MODE" \
    --score-alpha "$SCORE_ALPHA" \
    --target-fpr "$fpr_clean" \
    --out-json "$out_json" \
    > "$log_path" 2>&1
done
IFS=$old_ifs

"$PY" -c '
import csv
import glob
import json
import os
import sys

out_dir = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(out_dir, "eval_*.json"))):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    cal = d.get("calibrated", {})
    met = d.get("metrics", {})
    timing = d.get("timing", {})
    rows.append({
        "method": "TCN-GAN (attn+wgan-gp)",
        "window_size": d.get("window_size"),
        "stride": d.get("stride"),
        "score_mode": d.get("score_mode"),
        "score_alpha": d.get("score_alpha"),
        "target_fpr": cal.get("target_fpr"),
        "auc": met.get("auc"),
        "ap": met.get("ap"),
        "calib_f1": cal.get("f1"),
        "calib_recall": cal.get("recall"),
        "calib_precision": cal.get("precision"),
        "calib_threshold": cal.get("threshold"),
        "train_benign_fpr": cal.get("train_benign_fpr"),
        "test_benign_fpr": cal.get("test_benign_fpr"),
        "eval_seconds": timing.get("eval_seconds"),
        "json": path,
    })

rows.sort(key=lambda r: float(r["target_fpr"]))
csv_path = os.path.join(out_dir, "tcn_gan_fpr_sweep.csv")
fields = [
    "method", "window_size", "stride", "score_mode", "score_alpha", "target_fpr",
    "auc", "ap", "calib_f1", "calib_recall", "calib_precision", "calib_threshold",
    "train_benign_fpr", "test_benign_fpr", "eval_seconds", "json",
]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

md_path = os.path.join(out_dir, "tcn_gan_fpr_sweep.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# TCN-GAN FPR Sweep\n\n")
    f.write("| target_fpr | AUC | AP | F1 | Recall | Precision | Test FPR | Threshold |\n")
    f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for r in rows:
        f.write(
            "| {target_fpr:.3f} | {auc:.4f} | {ap:.4f} | {calib_f1:.4f} | {calib_recall:.4f} | {calib_precision:.4f} | {test_benign_fpr:.4f} | {calib_threshold:.6f} |\n".format(
                **{k: float(v) for k, v in r.items() if k not in {"method", "score_mode", "json"} and v is not None}
            )
        )

print(f"CSV已保存: {csv_path}")
print(f"Markdown已保存: {md_path}")
' "$OUT_DIR"

echo ""
echo "完成。主要结果："
echo "  - $OUT_DIR/tcn_gan_fpr_sweep.csv"
echo "  - $OUT_DIR/tcn_gan_fpr_sweep.md"
