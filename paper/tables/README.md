# `attack/paper/tables/`

这里现在只保留**当前最推荐用于论文写作**的数据表。

## 主文优先

- `main_comparison_target_fpr_0p05.csv`
  - CICIDS2017 主结果表，严格低误报工作点。

- `main_comparison_target_fpr_0p15.csv`
  - CICIDS2017 主结果表，高召回工作点。

- `swat_unsup_formal_with_tcn_target_fpr_0p05.csv`
  - SWaT 正式 anomaly-detection 协议主表。
  - 当前最适合放进主文的第二数据集结果。

- `efficiency_summary.csv`
  - 当前 TCN 在 SWaT 与 TON_IoT 上的训练时间 / 每 epoch 时间 / checkpoint 大小汇总。

## 完整补充表

- `swat_formal_full_target_fpr_0p05.csv`
  - SWaT 的完整模型表。
  - 包含 TCN、经典 baseline、TranAD、DeepSVDD。

- `toniot_formal_full_target_fpr_0p05.csv`
  - TON_IoT 的完整模型表。
  - 当前更适合作为补充实验 / challenge-set 表。

- `cross_dataset_formal_full_target_fpr_0p05.csv`
  - 将 SWaT 与 TON_IoT 放在同一张表里，用 `dataset` 列区分。
  - 适合总览跨数据集结果，避免多个零散表看不全模型。

## 现有主线表

- `ablation_target_fpr_0p05.csv`
- `ablation_target_fpr_0p15.csv`
- `ablation_fpr_sweep.csv`
- `fpr_sweep_all_methods.csv`
- `seed_results_5seeds.csv`
- `seed_stats_mean_std.csv`

## 说明

- 已经被替代的 smoke 表、旧的 supervised-mixed SWaT 表、以及重复 cross-dataset 目录，已移动到：
  - `attack/_archive/cleanup_20260419/`

- 如果后续补跑 `CICIDS2017` 上的 `TranAD / DeepSVDD`，推荐直接使用：
  - `attack/pipelines/run_modern_sota_cicids.sh`
