# `final_experiments/20260414_132416/`

当前论文主线最终实验目录。

## 核心配置

- `selected_main_config.json`
  - 最终冻结配置。
  - 当前论文中的主模型应以这个配置为准。

## 子目录说明

- `stride_sweep/`
  - 固定 `window=128` 后扫描 stride。

- `alpha_sweep/`
  - 固定 checkpoint 后扫描 fused score 的 `alpha`。
  - 最终使用 `alpha=0.24`。

- `threshold_curve/`
  - 阈值 trade-off 分析。
  - 用来查看不同 recall/FPR 之间的取舍。

- `tcn_fpr_sweep/`
  - 最终 Attentive TCN-WGAN-GP 在多个 `target_fpr` 下的评估结果。

- `ablation_2x2/`
  - `mean/attn x vanilla/wgan-gp` 消融实验。

- `seed_runs/`
  - 多 seed 稳定性实验。

- `reused_checkpoints/`
  - stride sweep 中复用的旧 checkpoint。

## 写论文优先入口

这些分散结果已经汇总到：

```text
attack/paper/
```
