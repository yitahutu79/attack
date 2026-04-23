#!/bin/bash
# TCN-GAN 论文主线实验脚本：
#   1) 先固定 window，用 alpha=0.2 细扫 stride
#   2) 再用最佳 stride 对应的 checkpoint 细扫 alpha
#   3) 消融前冻结 window/stride/alpha，最后跑 2x2 消融
#
# 断点/复用说明：
#   - 当前默认复用 final_experiments/20260414_132416/reused_checkpoints 下的 stride=8/16/32
#   - 这些 checkpoint 已从旧 run 复制到最终主线目录，避免再依赖归档历史结果
#   - 只补训 stride=10/12/20/24，然后合并所有 stride 结果再选最佳

set -euo pipefail

PY=${PY:-/Users/lijie/miniforge3/envs/attack/bin/python}
DATA=${DATA:-attack/dataset/CICIDS2017}

WINDOW_SIZE=${WINDOW_SIZE:-128}
ANOMALY_RATIO=${ANOMALY_RATIO:-0.15}
TARGET_FPR=${TARGET_FPR:-0.05}
EPOCHS=${EPOCHS:-12}

# stride 选择阶段使用的锚定 alpha。后续 alpha sweep 会在消融前冻结最终 alpha。
STRIDE_SCORE_ALPHA=${STRIDE_SCORE_ALPHA:-0.2}
ALPHA_STEP=${ALPHA_STEP:-0.02}

# 固定 window=128 后计划覆盖的 stride 范围，用于记录本次主线的完整候选集合。
# 真正要补训哪些点由 TRAIN_STRIDE_GRID 控制，已完成的点由 REUSE_STRIDES 控制。
STRIDE_GRID=${STRIDE_GRID:-8,10,12,16,20,24,32}

# 默认复用最终主线目录内的 checkpoint。旧 autotune 目录已归档，不再作为日常入口。
REUSE_STRIDE_RUNS=${REUSE_STRIDE_RUNS:-attack/results/final_experiments/20260414_132416/reused_checkpoints}
REUSE_STRIDES=${REUSE_STRIDES:-8,16,32}
TRAIN_STRIDE_GRID=${TRAIN_STRIDE_GRID:-10,12,20,24}

# 如果还想补 stride=4/6 等更密的点，可以运行时显式覆盖：
#   TRAIN_STRIDE_GRID="4,6,10,12,20,24" bash attack/run_final_experiments.sh

RUN_TAG=$(date +%Y%m%d_%H%M%S)
RUN_ROOT=${RUN_ROOT:-attack/results/final_experiments/$RUN_TAG}
STRIDE_ROOT=$RUN_ROOT/stride_sweep
ALPHA_OUT=$RUN_ROOT/alpha_sweep
ABLATION_ROOT=$RUN_ROOT/ablation_2x2
CONFIG_JSON=$RUN_ROOT/selected_main_config.json
STRIDE_COMBINED_DIR=$STRIDE_ROOT/combined
REUSED_RECORDS=$STRIDE_COMBINED_DIR/reused_records.csv

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

latest_child_dir() {
  find "$1" -mindepth 1 -maxdepth 1 -type d | sort | tail -1
}

best_stride_field() {
  local csv_path=$1
  local field=$2
  "$PY" -c '
import csv
import math
import sys

csv_path, field = sys.argv[1], sys.argv[2]
with open(csv_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit(f"empty autotune CSV: {csv_path}")

def score(row):
    raw = row.get("calib_f1") or "nan"
    try:
        return float(raw)
    except ValueError:
        return float("nan")

valid = [r for r in rows if not math.isnan(score(r))]
if not valid:
    raise SystemExit(f"no valid calib_f1 rows in: {csv_path}")
best = max(valid, key=score)
print(best[field])
' "$csv_path" "$field"
}

find_reuse_ckpt() {
  local stride=$1
  local old_ifs=$IFS
  local run_dir
  IFS=,
  for run_dir in $REUSE_STRIDE_RUNS; do
    local ckpt="$run_dir/ckpt_w${WINDOW_SIZE}_s${stride}.pt"
    if [ -f "$ckpt" ]; then
      printf "%s\n" "$ckpt"
      IFS=$old_ifs
      return 0
    fi
  done
  IFS=$old_ifs
  return 1
}

alpha_meta_field() {
  local meta_path=$1
  local field=$2
  "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]])' "$meta_path" "$field"
}

mkdir -p "$RUN_ROOT" "$STRIDE_ROOT" "$ALPHA_OUT" "$ABLATION_ROOT" "$STRIDE_COMBINED_DIR"

echo "=========================================="
echo "TCN-GAN 论文主线实验：消融前冻结配置"
echo "开始时间: $(date)"
echo "运行目录: $RUN_ROOT"
echo "=========================================="
echo ""
echo "扫参前固定项:"
echo "  - window_size: $WINDOW_SIZE"
echo "  - anomaly_ratio: $ANOMALY_RATIO"
echo "  - target_fpr: $TARGET_FPR"
echo "  - stride_grid: $STRIDE_GRID"
echo "  - reuse_stride_runs: $REUSE_STRIDE_RUNS"
echo "  - reuse_strides: $REUSE_STRIDES"
echo "  - train_stride_grid: $TRAIN_STRIDE_GRID"
echo "  - stride_score_alpha: $STRIDE_SCORE_ALPHA"
echo ""

# ============================================================================
# 1. 固定 window 后细扫 stride
# ============================================================================
echo "[1/4] 固定 window=${WINDOW_SIZE}，复用/补训 stride（alpha=${STRIDE_SCORE_ALPHA}）"
echo "=========================================="

printf "stride,ckpt_path,out_json,log_train,log_eval\n" > "$REUSED_RECORDS"

old_ifs=$IFS
IFS=,
for s in $REUSE_STRIDES; do
  if ! ckpt=$(find_reuse_ckpt "$s"); then
    echo "警告：没有找到可复用 stride=$s checkpoint，将跳过复用。"
    continue
  fi
  ckpt_dir=$(dirname "$ckpt")
  out_json="$STRIDE_COMBINED_DIR/eval_w${WINDOW_SIZE}_s${s}_fused.json"
  log_eval="$STRIDE_COMBINED_DIR/eval_w${WINDOW_SIZE}_s${s}_fused.log"
  log_train="$ckpt_dir/train_w${WINDOW_SIZE}_s${s}.log"

  echo "复用 stride=$s checkpoint: $ckpt"
  "$PY" attack/models/tcn_gan_experiment.py \
    --data-dir "$DATA" \
    --train-files "${TRAIN_FILES[@]}" \
    --test-files "${TEST_FILES[@]}" \
    --window-size "$WINDOW_SIZE" \
    --stride "$s" \
    --anomaly-ratio "$ANOMALY_RATIO" \
    --load "$ckpt" \
    --eval-only \
    --disc-pooling attn \
    --score-mode fused \
    --score-alpha "$STRIDE_SCORE_ALPHA" \
    --target-fpr "$TARGET_FPR" \
    --out-json "$out_json" \
    > "$log_eval" 2>&1

  printf "%s,%s,%s,%s,%s\n" "$s" "$ckpt" "$out_json" "$log_train" "$log_eval" >> "$REUSED_RECORDS"
done
IFS=$old_ifs

TRAIN_STRIDE_CSV=${TRAIN_STRIDE_CSV:-}
if [ -n "$TRAIN_STRIDE_CSV" ]; then
  echo "复用已完成的补训 CSV: $TRAIN_STRIDE_CSV"
elif [ -n "${TRAIN_STRIDE_GRID//,/}" ]; then
  TRAIN_STRIDE_ROOT=$STRIDE_ROOT/train_missing
  echo "补训 stride: $TRAIN_STRIDE_GRID"
  "$PY" attack/pipelines/run_tcn_gan_autotune.py \
    --python "$PY" \
    --data-dir "$DATA" \
    --train-files "${TRAIN_FILES[@]}" \
    --test-files "${TEST_FILES[@]}" \
    --fixed-window "$WINDOW_SIZE" \
    --stride-grid "$TRAIN_STRIDE_GRID" \
    --anomaly-ratio "$ANOMALY_RATIO" \
    --epochs "$EPOCHS" \
    --disc-pooling attn \
    --gan-loss wgan-gp \
    --gp-lambda 10 \
    --n-critic 5 \
    --score-mode fused \
    --score-alpha "$STRIDE_SCORE_ALPHA" \
    --target-fpr "$TARGET_FPR" \
    --metric-prefer calib_f1 \
    --post-viz \
    --out-root "$TRAIN_STRIDE_ROOT"
  TRAIN_STRIDE_RUN_DIR=$(latest_child_dir "$TRAIN_STRIDE_ROOT")
  TRAIN_STRIDE_CSV=$TRAIN_STRIDE_RUN_DIR/tcn_gan_autotune_results.csv
  echo "补训结果 CSV: $TRAIN_STRIDE_CSV"
else
  echo "TRAIN_STRIDE_GRID 为空，不补训新的 stride。"
fi

STRIDE_RUN_DIR=$STRIDE_COMBINED_DIR
STRIDE_CSV=$STRIDE_RUN_DIR/tcn_gan_autotune_results.csv

"$PY" -c '
import csv
import json
import math
import sys
from pathlib import Path

reused_csv = Path(sys.argv[1])
train_csv = Path(sys.argv[2]) if sys.argv[2] else None
out_csv = Path(sys.argv[3])
out_md = out_csv.with_name("README.md")
score_alpha = float(sys.argv[4])
target_fpr = float(sys.argv[5])

cols = [
    "window",
    "stride",
    "score_mode",
    "score_alpha",
    "target_fpr",
    "auc",
    "ap",
    "best_f1",
    "best_precision",
    "best_recall",
    "best_fpr",
    "calib_threshold",
    "calib_f1",
    "calib_precision",
    "calib_recall",
    "calib_fpr",
    "train_benign_fpr",
    "test_benign_fpr",
    "ckpt_path",
    "out_json",
    "xai_report_json",
    "log_train",
    "log_eval",
]

rows = []

if train_csv and train_csv.exists():
    with train_csv.open(newline="", encoding="utf-8") as f:
        rows.extend(list(csv.DictReader(f)))

if reused_csv.exists():
    with reused_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out_json = Path(r["out_json"])
            with out_json.open(encoding="utf-8") as jf:
                d = json.load(jf)
            m = d.get("metrics", {}) if isinstance(d.get("metrics"), dict) else {}
            c = d.get("calibrated", {}) if isinstance(d.get("calibrated"), dict) else {}
            xai = d.get("xai", {}) if isinstance(d.get("xai"), dict) else {}
            rows.append(
                {
                    "window": d.get("window_size", ""),
                    "stride": r["stride"],
                    "score_mode": d.get("score_mode", "fused"),
                    "score_alpha": f"{score_alpha:.3f}",
                    "target_fpr": f"{target_fpr:.3f}",
                    "auc": m.get("auc", ""),
                    "ap": m.get("ap", ""),
                    "best_f1": m.get("best_f1", ""),
                    "best_precision": m.get("best_precision", ""),
                    "best_recall": m.get("best_recall", ""),
                    "best_fpr": m.get("best_fpr", ""),
                    "calib_threshold": c.get("threshold", ""),
                    "calib_f1": c.get("f1", ""),
                    "calib_precision": c.get("precision", ""),
                    "calib_recall": c.get("recall", ""),
                    "calib_fpr": c.get("fpr", ""),
                    "train_benign_fpr": c.get("train_benign_fpr", ""),
                    "test_benign_fpr": c.get("test_benign_fpr", ""),
                    "ckpt_path": r["ckpt_path"],
                    "out_json": r["out_json"],
                    "xai_report_json": xai.get("report_json", "") if xai.get("enabled") else "",
                    "log_train": r["log_train"],
                    "log_eval": r["log_eval"],
                }
            )

def key(row):
    try:
        return int(row.get("stride", 0))
    except Exception:
        return 0

dedup = {}
for row in rows:
    dedup[str(row.get("stride"))] = row
rows = [dedup[k] for k in sorted(dedup, key=lambda x: int(x))]

if not rows:
    raise SystemExit("没有可合并的 stride 结果")

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for row in rows:
        w.writerow({k: row.get(k, "") for k in cols})

def metric(row):
    try:
        return float(row.get("calib_f1", "nan"))
    except Exception:
        return float("nan")

ranked = sorted(rows, key=metric, reverse=True)
lines = ["# Combined stride sweep", ""]
lines.append(f"- Results CSV: `{out_csv}`")
best = ranked[0]
lines.append("- Best by calib_f1: stride={} calib_f1={}".format(best.get("stride"), best.get("calib_f1")))
lines.append("")
lines.append("| rank | stride | AUC | AP | calib_f1 | ckpt |")
lines.append("| ---: | ---: | ---: | ---: | ---: | --- |")
for i, r in enumerate(ranked, 1):
    lines.append("| {} | {} | {} | {} | {} | `{}` |".format(i, r.get("stride"), r.get("auc"), r.get("ap"), r.get("calib_f1"), r.get("ckpt_path")))
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_csv)
' "$REUSED_RECORDS" "$TRAIN_STRIDE_CSV" "$STRIDE_CSV" "$STRIDE_SCORE_ALPHA" "$TARGET_FPR"

BEST_STRIDE=$(best_stride_field "$STRIDE_CSV" stride)
BEST_STRIDE_CKPT=$(best_stride_field "$STRIDE_CSV" ckpt_path)
BEST_STRIDE_EVAL=$(best_stride_field "$STRIDE_CSV" out_json)

echo "选中的 stride: $BEST_STRIDE"
echo "选中的 checkpoint: $BEST_STRIDE_CKPT"
echo "对应的 stride eval: $BEST_STRIDE_EVAL"
echo "合并后的 stride CSV: $STRIDE_CSV"
echo ""

# ============================================================================
# 2. 在最佳 stride 的 checkpoint 上细扫 alpha
# ============================================================================
echo "[2/4] 固定 window=${WINDOW_SIZE}、stride=${BEST_STRIDE}，细扫 alpha"
echo "=========================================="

"$PY" attack/pipelines/run_tcn_gan_alpha_sweep.py \
  --data-dir "$DATA" \
  --train-files "${TRAIN_FILES[@]}" \
  --test-files "${TEST_FILES[@]}" \
  --window-size "$WINDOW_SIZE" \
  --stride "$BEST_STRIDE" \
  --anomaly-ratio "$ANOMALY_RATIO" \
  --load "$BEST_STRIDE_CKPT" \
  --target-fpr "$TARGET_FPR" \
  --alpha-step "$ALPHA_STEP" \
  --out-dir "$ALPHA_OUT"

BEST_ALPHA=$(alpha_meta_field "$ALPHA_OUT/meta.json" best_alpha)
BEST_ALPHA_F1=$(alpha_meta_field "$ALPHA_OUT/meta.json" best_calib_f1)

echo "选中的 alpha: $BEST_ALPHA (calib_f1=$BEST_ALPHA_F1)"
echo ""

# ============================================================================
# 3. 消融前冻结主模型配置
# ============================================================================
echo "[3/4] 冻结主模型配置"
echo "=========================================="

"$PY" -c '
import json
import sys
from datetime import datetime

out = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "window_size": int(sys.argv[1]),
    "stride": int(sys.argv[2]),
    "score_alpha": float(sys.argv[3]),
    "stride_selection_alpha": float(sys.argv[4]),
    "anomaly_ratio": float(sys.argv[5]),
    "target_fpr": float(sys.argv[6]),
    "best_stride_ckpt": sys.argv[7],
    "best_stride_eval": sys.argv[8],
    "stride_sweep_csv": sys.argv[9],
    "alpha_sweep_meta": sys.argv[10],
    "alpha_sweep_csv": sys.argv[11],
}
with open(sys.argv[12], "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
    f.write("\n")
' \
  "$WINDOW_SIZE" \
  "$BEST_STRIDE" \
  "$BEST_ALPHA" \
  "$STRIDE_SCORE_ALPHA" \
  "$ANOMALY_RATIO" \
  "$TARGET_FPR" \
  "$BEST_STRIDE_CKPT" \
  "$BEST_STRIDE_EVAL" \
  "$STRIDE_CSV" \
  "$ALPHA_OUT/meta.json" \
  "$ALPHA_OUT/alpha_sweep.csv" \
  "$CONFIG_JSON"

echo "冻结配置文件: $CONFIG_JSON"
cat "$CONFIG_JSON"
echo ""

# ============================================================================
# 4. 使用冻结后的 window/stride/alpha 跑消融
# ============================================================================
echo "[4/4] 使用冻结配置做消融 (mean/attn x vanilla/wgan-gp)"
echo "=========================================="

"$PY" attack/pipelines/run_tcn_gan_ablation_2x2.py \
  --python "$PY" \
  --data-dir "$DATA" \
  --train-files "${TRAIN_FILES[@]}" \
  --test-files "${TEST_FILES[@]}" \
  --window-size "$WINDOW_SIZE" \
  --stride "$BEST_STRIDE" \
  --anomaly-ratio "$ANOMALY_RATIO" \
  --epochs "$EPOCHS" \
  --score-alpha "$BEST_ALPHA" \
  --target-fpr "$TARGET_FPR" \
  --xai-samples 128 \
  --xai-batch-size 128 \
  --out-root "$ABLATION_ROOT"

ABLATION_RUN_DIR=$(latest_child_dir "$ABLATION_ROOT")

echo ""
echo "=========================================="
echo "所有主线实验完成"
echo "结束时间: $(date)"
echo "=========================================="
echo ""
echo "冻结后的主模型配置:"
echo "  - $CONFIG_JSON"
echo ""
echo "结果目录:"
echo "  - stride sweep: $STRIDE_RUN_DIR"
echo "  - alpha sweep: $ALPHA_OUT"
echo "  - 消融实验: $ABLATION_RUN_DIR"
echo ""
echo "消融表格:"
echo "  - $ABLATION_RUN_DIR/ablation.csv"
echo "  - $ABLATION_RUN_DIR/ablation_summary.csv"
