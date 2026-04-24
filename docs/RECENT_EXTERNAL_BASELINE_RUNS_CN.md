# 最近外部模型补跑记录

更新时间：2026-04-24

## 结论

`main.tex` 当前已经纳入主要外部 baseline 结果，包括 `ALoRa`、`TimesNet`、`Anomaly Transformer`、`DLinear`、`Autoformer`。  
`ALoRa @ W=128` 三数据集统一口径重跑已经完成，`CICIDS/SWaT/UNSW` 新结果已落盘，可用于回填主文与汇报页。

## 已保留的外部仓库

| 仓库 | 当前状态 | 说明 |
|---|---:|---|
| `external/ALoRa` | 4/4 已覆盖 + `W=128` 已完成 | 已同时保留旧口径归档结果与 `CICIDS/SWaT/UNSW` 的 `W=128` 统一口径结果 |
| `external/Anomaly-Transformer` | 3/4 已覆盖 | 已补 `CICIDS_FORMAL`、`SWaT_FORMAL`、`UNSW_FORMAL`；`TON_LINUX_FORMAL` 尚未补 |
| `external/Time-Series-Library` | 4/4 已覆盖 | 主要用于 `TimesNet`，另有 `DLinear/Autoformer` 结果作为工程备份 |

已删除/不再纳入：`PaAno`、独立 `TimesNet` 仓库。

## 关键修复

- `ALoRa`：修正 `SWAT_FORMAL` 的 `window_metrics.json` 中误写的 dataset 标签。
- `Anomaly-Transformer`：修复 CPU 环境下写死 `.cuda()` 的问题。
- `Anomaly-Transformer`：新增 `--step`、`--target_fpr`、`--anomaly_ratio`、`--results_path`。
- `Anomaly-Transformer`：新增 calibrated window-level `window_metrics.json` 输出。
- `presentation.html`：补入 ALoRa、TimesNet、Anomaly Transformer 等外部 baseline，并更新汇报日期。

## 可复跑脚本

- `external/Anomaly-Transformer/scripts/CICIDS_FORMAL_fair.sh`
- `external/Anomaly-Transformer/scripts/SWaT_FORMAL_fair.sh`
- `external/Anomaly-Transformer/scripts/UNSW_FORMAL_fair.sh`

## Anomaly-Transformer 已出结果

| Dataset | AUC | AP | F1 | Recall | Precision | Test FPR |
|---|---:|---:|---:|---:|---:|---:|
| `CICIDS_FORMAL` | 0.5978 | 0.5069 | 0.2496 | 0.1571 | 0.6072 | 0.0744 |
| `SWaT_FORMAL` | 0.8661 | 0.9620 | 0.8869 | 0.8098 | 0.9803 | 0.0536 |
| `UNSW_FORMAL` | 0.5845 | 0.6577 | 0.2302 | 0.1361 | 0.7471 | 0.0571 |

结果文件：

- `external/Anomaly-Transformer/test_results/CICIDS_CICIDS_FORMAL_w128_s16/window_metrics.json`
- `external/Anomaly-Transformer/test_results/CICIDS_SWaT_FORMAL_w128_s16/window_metrics.json`
- `external/Anomaly-Transformer/test_results/CICIDS_UNSW_FORMAL_w128_s16/window_metrics.json`

## ALoRa 旧口径归档结果（W=20）

| Dataset | AUC | AP | F1 | Recall | Precision | Test FPR |
|---|---:|---:|---:|---:|---:|---:|
| `CICIDS_FORMAL` | 0.6936 | 0.6239 | 0.4939 | 0.3900 | 0.6733 | 0.1374 |
| `SWAT_FORMAL` | 0.9526 | 0.9879 | 0.9559 | 0.9177 | 0.9975 | 0.0075 |
| `UNSW_FORMAL` | 0.6892 | 0.7032 | 0.2045 | 0.1172 | 0.8030 | 0.0353 |
| `TON_LINUX_FORMAL` | 0.4618 | 0.2861 | 0.2555 | 0.2339 | 0.2816 | 0.2485 |

结果目录：

- `external/ALoRa/Results_archive/`

## ALoRa 统一口径重跑结果（W=128）

| Dataset | target FPR | AUC | AP | F1 | Recall | Precision | Test FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `CICIDS_FORMAL` | 0.05 | 0.5265 | 0.5823 | 0.4902 | 0.3865 | 0.6700 | 0.1394 |
| `SWAT_FORMAL` | 0.05 | 0.9794 | 0.9938 | 0.9822 | 0.9802 | 0.9842 | 0.0518 |
| `UNSW_FORMAL` | 0.15 | 0.7243 | 0.7553 | 0.2180 | 0.1263 | 0.7979 | 0.0397 |

结果文件：

- `external/ALoRa/Results_archive/CICIDS_FORMAL_W128/window_metrics.json`
- `external/ALoRa/Results_archive/SWAT_FORMAL_W128/window_metrics.json`
- `external/ALoRa/Results_archive/UNSW_FORMAL_W128/window_metrics.json`

## Time-Series-Library 补充记录（此前未写入）

以下结果来自 `external/Time-Series-Library/test_results/*/window_metrics.json`。

### 1) TimesNet

| 目标数据集（按目录名） | 结果文件里的 dataset 字段 | 运行口径 | Window | AUC | AP | F1 | Recall | Precision | Test FPR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `CICIDS_FORMAL` | `CICIDS_FORMAL` | `fair` | 128 | 0.3920 | 0.4233 | 0.4755 | 0.3598 | 0.6905 | 0.1124 |
| `SWaT_FORMAL` | `SWaT_FORMAL` | `fair` | 128 | 0.9951 | 0.9973 | 0.9913 | 0.9837 | 0.9991 | 0.0030 |
| `UNSW_FORMAL` | `UNSW_FORMAL` | `fair` | 128 | 0.7133 | 0.7806 | 0.2696 | 0.1579 | 0.9220 | 0.0166 |
| `UNSW_FORMAL` | `SWAT` | `legacy` | 100 | 0.7065 | 0.7721 | 0.5928 | 0.4619 | 0.8269 | 0.1196 |
| `TON_LINUX_FORMAL` | `SWAT` | `legacy` | 100 | 0.4998 | 0.2979 | 0.3166 | 0.3700 | 0.2767 | 0.4079 |
| `SWAT_FORMAL` | `SWAT` | `smoke` | 100 | 0.9942 | 0.9975 | 0.9906 | 0.9821 | 0.9992 | 0.0024 |

说明：

- `TimesNet` 的 `SWaT_FORMAL` 与 `UNSW_FORMAL` 已补齐 `fair (w128,s16)` 口径（2026-04-23）。
- `TimesNet` 旧目录中仍保留了 `UNSW/TON/SWAT` 的 `legacy/smoke (w100)` 结果，且部分文件 `dataset=SWAT`；主表请优先采用上述 fair 行。

### 2) DLinear（fair 口径完整）

| Dataset | 运行口径 | Window | AUC | AP | F1 | Recall | Precision | Test FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `CICIDS_FORMAL` | `fair` | 128 | 0.4024 | 0.4795 | 0.4879 | 0.3695 | 0.7102 | 0.1064 |
| `SWaT_FORMAL` | `fair` | 128 | 0.9945 | 0.9971 | 0.9909 | 0.9828 | 0.9993 | 0.0030 |
| `UNSW_FORMAL` | `fair` | 128 | 0.7400 | 0.8003 | 0.2599 | 0.1512 | 0.8968 | 0.0153 |
| `TON_LINUX_FORMAL` | `fair` | 128 | 0.4463 | 0.2917 | 0.2486 | 0.2093 | 0.3063 | 0.2008 |

### 3) Autoformer（fair 口径完整）

| Dataset | 运行口径 | Window | AUC | AP | F1 | Recall | Precision | Test FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `CICIDS_FORMAL` | `fair` | 128 | 0.4687 | 0.5285 | 0.4985 | 0.3822 | 0.7166 | 0.1107 |
| `SWaT_FORMAL` | `fair` | 128 | 0.9907 | 0.9962 | 0.9909 | 0.9828 | 0.9993 | 0.0030 |
| `UNSW_FORMAL` | `fair` | 128 | 0.7963 | 0.8377 | 0.5353 | 0.3681 | 0.9855 | 0.0092 |
| `TON_LINUX_FORMAL` | `fair` | 128 | 0.3870 | 0.2723 | 0.2540 | 0.2227 | 0.2960 | 0.2250 |

## 还未做

- `Anomaly-Transformer + TON_LINUX_FORMAL` 尚未补。
- `TimesNet + TON_LINUX_FORMAL` 如需进入公平主表，仍需补 `fair (w128,s16)`。
- 论文最终写表前，建议统一核对 `main.tex`、`presentation.html`、各 `window_metrics.json` 的 target FPR 口径，尤其是 UNSW 页面当前主文采用的是压力测试叙事。
