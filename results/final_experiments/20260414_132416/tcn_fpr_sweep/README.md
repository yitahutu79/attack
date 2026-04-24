# `tcn_fpr_sweep/`

最终 Attentive TCN-WGAN-GP checkpoint 的多目标误报率评估。

## 文件说明

- `tcn_gan_fpr_sweep.csv`
  - 最重要文件。
  - 同一个最终模型在 `target_fpr=0.01,0.03,0.05,0.10,0.15,0.20,0.30` 下的指标。

- `tcn_gan_fpr_sweep.md`
  - Markdown 版本，方便阅读和复制。

- `eval_w128_s16_fused_fpr*.json`
  - 每个 target-FPR 的详细评估 JSON。

- `eval_w128_s16_fused_fpr*.log`
  - 对应评估日志。

## 论文使用方式

这些结果已经合并到：

```text
attack/paper/tables/fpr_sweep_all_methods.csv
attack/paper/figures/fpr_sweep_f1_recall.png
```
