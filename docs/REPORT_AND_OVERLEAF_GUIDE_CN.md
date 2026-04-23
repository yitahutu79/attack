# 当前汇报与 Overleaf 使用指南

更新时间：2026-04-24

本文档回答三个当前版本最常用的问题：

1. 现在汇报优先用哪些图和表；
2. Overleaf 需要上传哪些文件；
3. 哪些文件是当前论文事实来源，哪些只是辅助说明。

## 1. 当前汇报优先使用哪些图

优先目录：

```text
paper/figures/
```

当前最推荐的顺序是：

1. `framework_overview.png`
   - 讲总流程：`flow -> sliding windows -> Attentive TCN-WGAN-GP -> threshold calibration -> online alarm -> XAI audit`

2. `model_architecture.png`
   - 讲模型结构：`TCN + attention pooling + WGAN-GP + fused scoring`

3. `fpr_sweep_f1_recall.png`
   - 讲 target-FPR 工作点变化下的 F1/Recall 取舍

4. `alpha_sweep.png`
   - 讲为什么最终选 `alpha=0.24`

5. `threshold_tradeoff.png`
   - 讲 recall 和 benign false alarms 的关系

6. `seed_f1_bar.png`
   - 讲多 seed 稳定性，尤其低 FPR 点的波动

7. `xai_panel.png` 或 `xai_case_multiview_shap.png`
   - 讲 feature/time attribution 与单窗口案例

如果只准备最小汇报材料，至少保留前 5 张。

## 2. 当前汇报应该优先讲哪些表

当前主文表格已经在 `paper/main.tex` 里，不建议再依赖旧的 `paper/tables/*.csv` 作为唯一真相。  
汇报时建议直接按 `main.tex` 的顺序组织：

1. `CICIDS2017 @ target FPR 0.05`
   - 严格低误报主表

2. `CICIDS2017 modern baselines @ target FPR 0.05`
   - GANomaly、ALoRa、TranAD、TimesNet、DLinear、Autoformer、Anomaly Transformer、DeepSVDD

3. `CICIDS2017 @ target FPR 0.15`
   - 高召回工作点

4. `SWaT @ target FPR 0.05`
   - 工控强验证

5. `UNSW-NB15 @ target FPR 0.15`
   - 压力测试

6. `Ablation @ target FPR 0.15`
   - 解释 attention 与 WGAN-GP 的组合效果

7. `Inference efficiency`
   - 讲 CPU-only feasibility

8. `Noise robustness`
   - 轻量 sanity check

9. `XAI case / top-feature stability`
   - 审计性与解释性

## 3. 当前 Overleaf 需要上传哪些内容

最核心的是：

```text
paper/main.tex
paper/references.bib
paper/figures/*.png
```

如果 Overleaf 工程里已经有 `figures/` 目录，当前推荐保留这些：

```text
framework_overview.png
model_architecture.png
fpr_sweep_f1_recall.png
alpha_sweep.png
threshold_tradeoff.png
seed_f1_bar.png
xai_case_multiview_shap.png
```

补充图可按需要加入：

```text
xai_feature_importance.png
xai_time_importance.png
attn_weights.png
xai_panel.png
```

## 4. 当前写论文时优先参考哪些文件

优先级建议：

1. `paper/main.tex`
2. `paper/references.bib`
3. `docs/CURRENT_PAPER_GUIDE_CN.md`
4. `docs/RECENT_EXTERNAL_BASELINE_RUNS_CN.md`
5. `docs/TCN_GAN_THESIS_EXPERIMENT_RECORD.md`

如果这些文件之间冲突：

> 以 `paper/main.tex` 和最新结果文件为准。

## 5. 当前最需要注意的几个小点

### 5.1 图名已经变了

现在优先使用：

- `framework_overview.png`
- `model_architecture.png`

不要再按旧说法找：

- `tcn_gan_paper_framework.png`
- `fig1.png`
- `fig2.png`

### 5.2 ALoRa 还没完全统一口径

当前主文中的 `ALoRa` 还是 `W=20`。  
`ALoRa@W128` 已启动过但未完成落盘，所以在论文和汇报中还不能假装它已经统一成 `W=128`。

### 5.3 SWaT/UNSW 没有 RF/MLP 是正常的

这是因为 formal one-class split 下训练集主要是 benign，监督模型不公平，不是漏写。

### 5.4 DLinear / Autoformer 已经进入主文

它们不再只是“工程备份结果”，而是已经写进 `main.tex` 的 modern baseline / cross-dataset tables 中。

## 6. 一句话总结

当前汇报和 Overleaf 的工作重点不是再去找旧图旧表，而是围绕 `main.tex` 当前叙事，把 `framework_overview`、`model_architecture`、CICIDS/SWaT/UNSW 主表和 XAI 案例讲清楚。
