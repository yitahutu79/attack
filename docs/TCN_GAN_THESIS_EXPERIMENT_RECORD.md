# TCN-GAN Thesis Experiment Record

更新时间：2026-04-24

这份文档是主模型 `Attentive TCN-WGAN-GP` 实验的 source of truth。外部 baseline（ALoRa / Anomaly-Transformer / Time-Series-Library）的最新状态请看 `docs/RECENT_EXTERNAL_BASELINE_RUNS_CN.md`；最终论文表格以 `paper/main.tex` 为准。

## 1. 最终冻结配置

最终主模型配置记录在：

```text
attack/results/final_experiments/20260414_132416/selected_main_config.json
```

核心参数：

```text
window_size = 128
stride = 16
score_alpha = 0.24
stride_selection_alpha = 0.20
anomaly_ratio = 0.15
target_fpr = 0.05
disc_pooling = attn
gan_loss = wgan-gp
score_mode = fused
```

最终主模型 checkpoint 已复制到本次主线目录内：

```text
attack/results/final_experiments/20260414_132416/reused_checkpoints/ckpt_w128_s16.pt
```

## 2. 必看结果

### Stride Sweep

```text
attack/results/final_experiments/20260414_132416/stride_sweep/combined/tcn_gan_autotune_results.csv
attack/results/final_experiments/20260414_132416/stride_sweep/combined/README.md
```

说明：

- 本次比较的 stride 为 `8,10,12,16,20,24,32`。
- 选择指标为 `calib_f1`。
- 最佳 stride 为 `16`。
- 旧 checkpoint 已复制到 `reused_checkpoints/`，表内路径已经改成本次主线内的路径。

### Alpha Sweep

```text
attack/results/final_experiments/20260414_132416/alpha_sweep/alpha_sweep.csv
attack/results/final_experiments/20260414_132416/alpha_sweep/alpha_sweep.png
attack/results/final_experiments/20260414_132416/alpha_sweep/meta.json
```

结论：

```text
best_alpha = 0.24
calib_f1 = 0.7958
calib_recall = 0.6640
```

这里的 recall 是在低误报标定约束下的结果，不代表模型排序能力上限。阈值分析见下一节。

### Threshold Trade-Off

```text
attack/results/final_experiments/20260414_132416/threshold_curve/threshold_tradeoff.csv
attack/results/final_experiments/20260414_132416/threshold_curve/threshold_tradeoff.png
```

重要结论：

```text
Recall >= 0.90 时，FPR 约为 0.147
Recall >= 0.95 时，FPR 约为 0.158
```

论文里可以表述为：低误报设置下模型较保守；如果部署场景更重视少漏报，可以通过调低阈值把 recall 提升到 90% 以上，但需要接受约 15% 的 FPR。

### 2x2 Ablation

正式消融表：

```text
attack/results/final_experiments/20260414_132416/ablation_2x2/20260414_204232/ablation.csv
attack/results/final_experiments/20260414_132416/ablation_2x2/20260414_204232/ablation_summary.csv
```

这两张表已经清理过，只包含本次冻结配置：

```text
window = 128
stride = 16
score_alpha = 0.24
```

当前结果：

| disc_pooling | gan_loss | calib_f1 | calib_recall | calib_precision | test_benign_fpr |
| --- | --- | ---: | ---: | ---: | ---: |
| mean | vanilla | 0.8258 | 0.7261 | 0.9574 | 0.0237 |
| attn | wgan-gp | 0.7958 | 0.6640 | 0.9928 | 0.0036 |
| mean | wgan-gp | 0.6176 | 0.4851 | 0.8498 | 0.0628 |
| attn | vanilla | 0.4807 | 0.3697 | 0.6868 | 0.1235 |

注意：旧的污染表曾混入 `attack/results/tcn_gan_autotune_runs/` 和 `score_alpha=0.6`，已经修正并归档。以后只看上面两个正式文件。

### XAI

当前保留的 XAI 输出：

```text
attack/results/xai_tcn/eval_w128_s16_fused_w128_s16_fused_a0.24/
```

论文可用图：

```text
xai_time_importance.png
xai_feature_importance.png
```

XAI 用来解释最终模型报警依据，不作为 baseline 分数的一部分。

## 3. Baseline

当前保留窗口级 baseline 目录：

```text
attack/results/baseline_fpr_sweep/20260415_020742/
```

重要提醒：如果 baseline 表中的 `TCN-GAN (Ours)` 仍来自旧基础模型，需要用最终冻结配置重新生成后再写入论文主表。不要把旧 flow-level baseline 和窗口级结果混在同一张主表里。

## 4. 当前项目清理结果

保留在主目录的内容：

- `attack/pipelines/run_final_experiments.sh`
- `attack/pipelines/run_baseline.sh`
- `attack/models/`
- `attack/baselines/`
- `attack/pipelines/`
- `attack/results/final_experiments/20260414_132416/`
- `attack/results/baseline_fpr_sweep/20260415_020742/`
- `attack/results/xai_tcn/eval_w128_s16_fused_w128_s16_fused_a0.24/`
- `attack/paper/`
- `attack/paper/figures/tcn_gan_paper_framework.svg`

归档目录：

```text
attack/_archive/cleanup_20260415/
```

这次归档了：

- 旧 `tcn_gan_autotune_runs/`
- 旧 `alpha_sweeps/`
- 中断的 `final_experiments/20260413_233918`
- 旧 XAI `alpha=0.6` 图
- 重复/误导的 clean 表副本
- LaTeX 编译中间文件
- `.DS_Store` 和 `__pycache__`
- 旧目录索引和历史实验记录

这些文件只是移动，没有删除。

## 5. 后续写论文时的引用顺序

1. 方法框架图：

```text
attack/paper/figures/tcn_gan_paper_framework.svg
attack/paper/figures/tcn_gan_paper_framework.png
```

2. 最终配置：

```text
attack/results/final_experiments/20260414_132416/selected_main_config.json
```

3. 三个 sweep 图/表：

```text
stride_sweep/combined/tcn_gan_autotune_results.csv
alpha_sweep/alpha_sweep.png
threshold_curve/threshold_tradeoff.png
```

4. 消融：

```text
ablation_2x2/20260414_204232/ablation_summary.csv
```

5. XAI：

```text
attack/results/xai_tcn/eval_w128_s16_fused_w128_s16_fused_a0.24/
```
