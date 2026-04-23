# `ablation_2x2/20260414_204232/fpr_sweep/`

四个消融模型的多目标误报率评估。

## 文件说明

- `ablation_fpr_sweep.csv`
  - 最重要文件。
  - 包含 `mean/attn x vanilla/wgan-gp` 四种组合在多个 target-FPR 下的结果。

- `ablation_fpr_sweep.md`
  - Markdown 版本。

- `eval_*_fpr*.json`
  - 每个消融模型、每个 target-FPR 的详细评估结果。

- `eval_*_fpr*.log`
  - 对应评估日志。

## 论文使用方式

这些结果已经整理到：

```text
attack/paper/tables/ablation_target_fpr_0p05.csv
attack/paper/tables/ablation_target_fpr_0p15.csv
attack/paper/tables/ablation_fpr_sweep.csv
```
