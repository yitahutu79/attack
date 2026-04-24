# `attack/results/`

这里保留的是当前论文主线仍在使用、或仍有复查价值的结果目录。

如果你的目标是：

- 写论文
- 做组会汇报
- 核对表格与图

优先看：

```text
attack/paper/
```

`attack/paper/` 已经把分散在 `results/` 下的主线结果整理成论文和汇报可直接使用的图表。

## 当前保留目录

- `final_experiments/20260414_132416/`
  - 当前 TCN 主线实验。
  - 包含 `stride sweep`、`alpha sweep`、`threshold curve`、`TCN FPR sweep`、`seed runs`、`2x2 ablation`。

- `cross_dataset_formal_unsup/`
  - 当前跨数据集 formal split 结果。
  - 主要保存 `SWaT`、`TON_IoT`、`UNSW-NB15` 的 TCN 与基础无监督 baseline 结果。

- `sota_deepsvdd_tranad_cicids/`
  - CICIDS2017 上的 `DeepSVDD + TranAD` 主表结果。

- `sota_ganomaly_cicids/`
  - CICIDS2017 上 `GANomaly` 在 `target_fpr=0.05` 的结果。

- `sota_ganomaly_cicids_fpr015/`
  - CICIDS2017 上 `GANomaly` 在 `target_fpr=0.15` 的结果。

- `sota_ganomaly_swat/`
- `sota_ganomaly_toniot/`
- `sota_deepsvdd_swat/`
- `sota_deepsvdd_toniot/`
- `sota_tranad_swat/`
- `sota_tranad_toniot/`
- `sota_unsw_nb15/`
  - 这些目录保存当前论文里仍在使用的现代 baseline 独立结果。

- `sota_rf_mlp_swat_fixed/`
- `sota_rf_mlp_toniot_fixed/`
  - 这是修正后的 `RF/MLP` 结果。
  - 旧的非 `fixed` 版本因为训练标签只有一类而被跳过，已归档。

- `robustness_tests/`
  - 当前保留的轻量噪声鲁棒性结果。

- `xai_tcn/`
  - 最终主模型的 XAI 报告与三张主线图。

- `xai_case_study_shap/`
  - 当前保留的单窗口多视角 XAI 个案结果。

## 已归档内容

以下内容已经移到 `_archive/`，不再作为当前主线结果使用：

- `fixed` 之前的旧 `RF/MLP` 结果
- `v2`、旧版单窗口 XAI 目录
- 明显的测试/中间/历史日志目录
- 外部仓库的 `smoke`、`test_results`、旧 `W=20` 结果

如果要查历史产物，请从 `_archive/cleanup_20260424_*` 下面找。

## 结果使用建议

- 论文表格和汇报图优先使用 `attack/paper/tables/` 和 `attack/paper/figures/`
- 需要追溯原始来源时，再回到 `results/` 对应目录
- 不要再从 `_archive/` 目录取主结果，除非你明确是在查历史版本

详细说明见：

```text
attack/results/RESULTS_GUIDE_CN.md
```
