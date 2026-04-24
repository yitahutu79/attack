# `threshold_curve/`

这里保存最终模型的阈值 trade-off 分析。

## 文件说明

- `scores_w128_s16_a024.csv`
  - 每个测试窗口的 anomaly score 和标签。

- `threshold_tradeoff.csv`
  - 不同阈值下的 Recall、FPR、Precision、F1 等指标。

- `threshold_tradeoff.png`
  - 可视化曲线。
  - 用来回答“如果我想要 90% 以上 recall，要付出多少误报率”。

- `eval_w128_s16_a024.json`
  - 当前阈值设置下的评估摘要。

- `eval_w128_s16_a024.log`
  - 评估日志。

## 论文/汇报图

图已经复制到：

```text
attack/paper/figures/threshold_tradeoff.png
attack/paper/figures/threshold_tradeoff.png
```
