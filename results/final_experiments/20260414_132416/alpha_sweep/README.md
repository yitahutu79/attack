# `alpha_sweep/`

固定最终 checkpoint 后扫描 fused anomaly score 的融合比例 `alpha`。

## 文件说明

- `alpha_sweep.csv`
  - 不同 `alpha` 下的评估指标。

- `alpha_sweep.png`
  - alpha 选择图。

- `meta.json`
  - 本次扫描的配置元信息。

## 结论

当前最终配置使用：

```text
score_alpha = 0.24
```

论文和汇报图已复制到：

```text
attack/paper/figures/alpha_sweep.png
attack/paper/figures/alpha_sweep.png
```
