# `reused_checkpoints/`

这里保存 stride sweep 中复用的旧 checkpoint 和训练日志。

## 为什么保留

最终实验脚本为了节省时间，没有重复训练已经完成且口径一致的 stride：

```text
stride = 8, 16, 32
```

这些 checkpoint 被复制到这里，方便追溯最终 stride sweep 的来源。

## 文件说明

- `ckpt_w128_s*.pt`
  - 对应 stride 的模型 checkpoint。

- `train_w128_s*.log`
  - 对应训练日志。

论文主表不直接引用这里的文件，最终汇总结果看：

```text
attack/results/final_experiments/20260414_132416/stride_sweep/combined/
```
