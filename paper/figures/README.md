# `attack/paper/figures/`

这里是论文 `main.tex` 直接引用的图片目录。

## 当前图片

- `framework_overview.png`、`framework_overview.svg`
  - 论文主框架图。
  - 汇报时用来讲：flow -> window -> Attentive TCN-WGAN-GP -> fused score -> threshold -> alarm，并说明 Evaluation/XAI 如何支撑论文结论。
  - `svg` 是可编辑矢量版，适合后期微调；`png` 是 Overleaf 最省心的位图版。

- `model_architecture.png`、`model_architecture.svg`
  - 模型结构图。
  - 用来讲 input window、TCN dilated convolution、hidden states、attention pooling、generator、critic、WGAN-GP objective 和 fused scoring 的关系。
  - `svg` 是可编辑矢量版，适合后期微调；`png` 是 Overleaf 最省心的位图版。

- `fpr_sweep_f1_recall.png`
  - 不同目标误报率下的 F1 和 recall 曲线。
  - 汇报时用来说明模型在高召回工作点的优势。

- `alpha_sweep.png`
  - 融合评分参数 `alpha` 扫描图。
  - 汇报时用来说明最终 `alpha=0.24` 的选择来源。

- `threshold_tradeoff.png`
  - 阈值、recall、FPR、precision/F1 的取舍图。
  - 汇报时用来回答“想要 90% recall 时误报会怎样”。

- `seed_f1_bar.png`
  - 多 seed 稳定性图。
  - 汇报时用来说明随机初始化带来的波动。

- `xai_panel.png`
  - XAI 组合图，统一排版展示 feature attribution、temporal attribution 和 attention weights。

- `xai_feature_importance.png`、`xai_time_importance.png`、`attn_weights.png`
  - XAI 原始单图，保留用于汇报或重新生成组合图。

- 旧版 `tcn_gan_paper_framework.*`
  - 已归档到 `attack/_archive/cleanup_20260416/legacy_figures/`。
  - 论文正文和汇报优先使用 `framework_overview.png`。

这是当前唯一的论文/汇报图片集合。旧的重复副本已经归档到 `_archive/`。
