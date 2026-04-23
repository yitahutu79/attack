# `ablation_2x2/20260414_204232/`

当前论文使用的 2x2 消融实验结果。

## 顶层表格

- `ablation.csv`
  - 每个消融模型在默认工作点下的结果。

- `ablation_summary.csv`
  - 消融结果摘要。

- `ablation.md`、`ablation_summary.md`
  - Markdown 版本，方便阅读。

## 四个子目录

- `20260414_204232/`
  - `mean + vanilla`

- `20260414_210038/`
  - `attn + vanilla`

- `20260414_212058/`
  - `mean + wgan-gp`

- `20260414_225204/`
  - `attn + wgan-gp`，即最终主模型结构。

每个子目录里都有：

```text
ckpt_w128_s16.pt
train_w128_s16.log
eval_w128_s16_fused.json
eval_w128_s16_fused.log
tcn_gan_autotune_results.csv
```

## FPR sweep

- `fpr_sweep/`
  - 四个消融模型在多个 target-FPR 下的重新评估。
  - 论文当前更推荐引用 target-FPR=0.15 的消融结果，因为它更清楚体现 attention + WGAN-GP 的组合优势。
