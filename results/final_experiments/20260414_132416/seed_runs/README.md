# `seed_runs/`

最终模型多 seed 稳定性实验。

## 文件说明

- `seed_summary.csv`
  - 汇总每个 seed 的核心指标。

- `seed_43/`、`seed_44/`、`seed_45/`、`seed_46/`
  - 各 seed 的训练 checkpoint、日志和评估结果。

默认 seed=42 来自主线最终 checkpoint，因此这里主要保存补跑 seed。

## 论文使用方式

多 seed 结果已经整理为：

```text
attack/paper/tables/seed_results_5seeds.csv
attack/paper/tables/seed_stats_mean_std.csv
attack/paper/figures/seed_f1_bar.png
```
