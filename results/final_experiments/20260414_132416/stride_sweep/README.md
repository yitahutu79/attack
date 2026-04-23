# `stride_sweep/`

固定 `window=128` 后扫描 stride 的结果目录。

## 子目录

- `combined/`
  - 最终合并后的 stride sweep 结果。
  - 包含复用旧 checkpoint 的 stride 和本次补训的 stride。

- `train_missing/`
  - 只补训缺失 stride 的输出目录。

## 论文使用方式

最终选择的 stride 写入：

```text
attack/results/final_experiments/20260414_132416/selected_main_config.json
```

参数选择图表优先看：

```text
attack/paper/
```
