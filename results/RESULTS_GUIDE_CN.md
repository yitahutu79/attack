# 结果目录说明

本文档说明 `attack/results/` 当前保留目录的用途、哪些目录已经归档，以及论文与汇报应优先从哪里取结果。

## 1. 使用原则

当前建议遵循下面的顺序：

```text
1. attack/paper/
2. attack/results/
3. attack/_archive/
```

也就是说：

- 写论文和做汇报时，优先使用 `attack/paper/`
- 需要追溯原始输出时，再回到 `attack/results/`
- `_archive/` 只用于查历史版本，不再作为主线引用来源

## 2. 当前保留的主线目录

### 2.1 `final_experiments/20260414_132416/`

目录：

```text
attack/results/final_experiments/20260414_132416/
```

用途：当前 Attentive TCN-WGAN-GP 论文主线实验。

包含：

- `selected_main_config.json`
- `stride_sweep/`
- `alpha_sweep/`
- `threshold_curve/`
- `tcn_fpr_sweep/`
- `seed_runs/`
- `ablation_2x2/`

这是当前 TCN 主线最核心的结果目录。

### 2.2 `cross_dataset_formal_unsup/`

目录：

```text
attack/results/cross_dataset_formal_unsup/
```

用途：当前跨数据集 formal split 结果。

子目录：

- `swat_baselines/`
- `swat_tcn/`
- `ton_iot_baselines/`
- `ton_iot_tcn/`
- `unsw_nb15_tcn/`

这些目录主要支撑：

- SWaT 的 formal anomaly-detection split 结果
- TON_IoT 的 limitation / cross-domain 结果
- UNSW-NB15 的 TCN 压力测试结果

### 2.3 现代 baseline 独立结果

当前保留的单独 baseline 结果目录包括：

```text
attack/results/sota_deepsvdd_tranad_cicids/
attack/results/sota_ganomaly_cicids/
attack/results/sota_ganomaly_cicids_fpr015/
attack/results/sota_ganomaly_swat/
attack/results/sota_ganomaly_toniot/
attack/results/sota_deepsvdd_swat/
attack/results/sota_deepsvdd_toniot/
attack/results/sota_tranad_swat/
attack/results/sota_tranad_toniot/
attack/results/sota_unsw_nb15/
attack/results/sota_rf_mlp_swat_fixed/
attack/results/sota_rf_mlp_toniot_fixed/
```

说明：

- `sota_rf_mlp_*_fixed/` 是修正后保留的正式版本
- 对应的旧目录 `sota_rf_mlp_swat/` 和 `sota_rf_mlp_toniot/` 已归档
- 旧版本的问题是训练标签只有单一类别，结果被跳过，不应继续引用

### 2.4 XAI 结果

当前保留两类 XAI 结果：

```text
attack/results/xai_tcn/
attack/results/xai_case_study_shap/
```

区别：

- `xai_tcn/`
  - 主模型标准 XAI 输出
  - 包含 `xai_time_importance.png`、`xai_feature_importance.png`、`attn_weights.png`
  - 当前仍被 `make_paper_ready_assets.py` 和若干主线结果 JSON 引用

- `xai_case_study_shap/`
  - 单窗口多视角 XAI 个案
  - 包含 `xai_case_multiview.png`、`xai_case_top_features.csv`
  - 对应 PPT 中的单窗口案例图

旧的：

- `results/xai_case_study/`
- `results/xai_case_study_v2/`

都已经归档。

### 2.5 `robustness_tests/`

目录：

```text
attack/results/robustness_tests/
```

用途：保存当前保留的轻量噪声鲁棒性结果。

## 3. 当前结果怎么对应到论文

### 3.1 TCN 主表与曲线

优先对应：

```text
attack/results/final_experiments/20260414_132416/
```

### 3.2 SWaT / TON_IoT / UNSW-NB15

优先对应：

```text
attack/results/cross_dataset_formal_unsup/
attack/results/sota_*/
```

### 3.3 XAI 图

优先对应：

```text
attack/results/xai_tcn/
attack/results/xai_case_study_shap/
```

## 4. 已归档的结果类型

目前已经归档的主要类型包括：

- `fixed` 之前的旧 baseline 目录
- `v2` / 旧版 XAI case study
- 旧日志目录
- 空目录和中间目录
- 外部仓库里的 `smoke`、`test_results`
- 外部仓库里旧 `W=20` 结果

这些目录并未删除，只是挪到了 `_archive/cleanup_20260424_*` 下。

## 5. 现在应该避免的用法

- 不要再从 `results/` 中已经归档的旧目录取表格
- 不要继续引用旧的非 `fixed` RF/MLP 结果
- 不要把 `_archive/` 中的旧 XAI 结果当成当前主图
- 不要把外部仓库里的 `smoke/test` 结果继续混入正式表格

## 6. 推荐读取路径

如果你只是想快速定位当前主线结果，建议按这个顺序看：

```text
attack/paper/figures/
attack/paper/tables/
attack/results/final_experiments/20260414_132416/
attack/results/cross_dataset_formal_unsup/
attack/results/xai_tcn/
attack/results/xai_case_study_shap/
```
