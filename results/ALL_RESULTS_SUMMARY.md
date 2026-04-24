# All Results Summary

这份文件现在只作为**人工导航摘要**，不再维护逐表逐数值的大汇总。

原因是：

- 当前论文结果已经分散整理到 `attack/paper/tables/`
- `results/` 目录近期完成过一轮去重和归档
- 旧版“大杂烩总表”容易继续保留历史数值，和当前主线不一致

## 当前主线结果入口

- TCN 主线实验：
  - `attack/results/final_experiments/20260414_132416/`

- 跨数据集 formal 结果：
  - `attack/results/cross_dataset_formal_unsup/`

- 现代 baseline 独立结果：
  - `attack/results/sota_deepsvdd_tranad_cicids/`
  - `attack/results/sota_ganomaly_*`
  - `attack/results/sota_deepsvdd_*`
  - `attack/results/sota_tranad_*`
  - `attack/results/sota_unsw_nb15/`
  - `attack/results/sota_rf_mlp_*_fixed/`

- XAI：
  - `attack/results/xai_tcn/`
  - `attack/results/xai_case_study_shap/`

## 论文/汇报优先入口

如果你的目标是直接取论文或汇报所需图表，请优先看：

```text
attack/paper/figures/
attack/paper/tables/
```

## 不再建议使用

以下类型的内容已经归档或不再作为主线引用来源：

- 非 `fixed` 的旧 `RF/MLP` 目录
- 旧版 `xai_case_study` / `xai_case_study_v2`
- 旧日志目录
- 外部仓库的 `smoke` / `test_results`
- 旧 `W=20` 结果目录

需要追溯历史版本时，请到：

```text
attack/_archive/
```
