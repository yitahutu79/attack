# `attack/results/`

这里现在只保留论文主线直接需要看的结果。

## 最推荐入口

写论文、做汇报时优先看：

```text
attack/paper/
```

`attack/paper/` 已经把分散在 baseline、final experiments、seed、ablation 和 XAI 目录里的结果整理成论文可用的图和表。

## 当前保留目录

- `baseline_fpr_sweep/20260415_020742/`
  - 当前完整窗口级 baseline 的多误报率扫描结果。
  - 优先看 `baseline_results_with_target_fpr.csv`。

- `final_experiments/20260414_132416/`
  - 当前论文主线最终结果。
  - 包含 stride sweep、alpha sweep、阈值 trade-off、TCN FPR sweep、冻结配置和 2x2 消融。

- `xai_tcn/eval_w128_s16_fused_w128_s16_fused_a0.24/`
  - 最终模型的 XAI 图和报告。

- `run_logs/`
  - 当前目录仅作为未来运行日志的落点。
  - 已有运行日志已归档到 `_archive/cleanup_20260415/legacy_results/run_logs_current_session/`。

详细解释见：

```text
attack/results/RESULTS_GUIDE_CN.md
```

## 不再直接使用

以下旧结果已迁到：

```text
attack/_archive/cleanup_20260415/legacy_results/
```

包括：

- `tcn_gan_autotune_runs/`
- `alpha_sweeps/`
- 中断的 `final_experiments/20260413_233918`
- 旧 XAI `alpha=0.6` 结果
- 运行日志 `run_logs/`
- baseline FPR sweep 的快速试跑目录
- 旧单工作点 `baseline_comparison/`

这些文件只是移动，没有删除。日常写论文和画图时，不要再从归档目录取主结果。

## 论文/汇报资产

论文和汇报用的最终图表不再放在 `results/` 下，统一放到：

```text
attack/paper/figures/
attack/paper/tables/
```
