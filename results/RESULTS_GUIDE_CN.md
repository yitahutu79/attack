# 结果目录说明

本文档解释 `attack/results/` 下当前保留结果的用途、论文中应该怎么使用，以及 XAI 图像应该如何解读。

## 当前主线目录

当前只需要关注这些目录：

```text
attack/paper/
attack/results/baseline_fpr_sweep/20260415_020742/
attack/results/final_experiments/20260414_132416/
attack/results/xai_tcn/eval_w128_s16_fused_w128_s16_fused_a0.24/
```

其他旧结果已经移动到：

```text
attack/_archive/cleanup_20260415/legacy_results/
```

归档目录里的文件不要再作为论文主结果使用，除非明确说明是历史实验或补充材料。

## 0. paper

目录：

```text
attack/paper/
```

用途：当前论文写作和汇报优先入口。这里把最终结果整理成可以直接放入论文和 Overleaf 的表格、图片和 tex 文件。

主要表格：

- `tables/main_comparison_target_fpr_0p05.csv`
  - 主对比表，低误报工作点。

- `tables/main_comparison_target_fpr_0p15.csv`
  - 较高召回工作点，可用于补充分析。

- `tables/seed_stats_mean_std.csv`
  - 5 个 seed 的均值和标准差。

- `tables/ablation_target_fpr_0p05.csv`
  - 低误报消融。

- `tables/ablation_target_fpr_0p15.csv`
  - 较高召回消融。该表中 `attn+wgan-gp` 排名第一，可用于解释 attention 在不同工作点下的作用。

主要图片：

- `figures/fpr_sweep_f1_recall.png`
  - Attentive TCN-WGAN-GP 和 baseline 的 F1/Recall 随 target FPR 的变化。

- `figures/seed_f1_bar.png`
  - 多 seed 下 `target_fpr=0.05/0.15` 的 F1 波动。

- `figures/alpha_sweep.png`
- `figures/threshold_tradeoff.png`
- `figures/xai_time_importance.png`
- `figures/xai_feature_importance.png`
- `figures/attn_weights.png`

## 1. baseline_fpr_sweep

目录：

```text
attack/results/baseline_fpr_sweep/20260415_020742/
```

用途：当前正式 baseline 的多误报率扫描结果。这里用于回答论文中的核心问题：

> Attentive TCN-WGAN-GP 相比传统异常检测、监督窗口分类器和深度序列 baseline 是否有优势？

主要文件：

- `baseline_results_with_target_fpr.csv`
  - 推荐作为当前论文 baseline sweep 表格的数据来源。
  - 比原始 `baseline_results.csv` 多一列明确的 `target_fpr`，阅读和画图更清楚。

- `baseline_results.csv`
  - 原始输出表。里面 `train_benign_fpr` 是实际训练 BENIGN FPR，不等同于请求的目标 FPR。

- `baseline_results.json`
  - 原始 JSON 输出。

关键结论：

- RF 不再出现 F1/Recall 全为 0，说明旧结果主要受阈值工作点和标定口径影响。
- 在 `target_fpr=0.05` 下，Attentive TCN-WGAN-GP 相比 RF/MLP 有更高 F1 和极低 test FPR。
- 在 `target_fpr=0.15` 附近，Attentive TCN-WGAN-GP 的 Recall 和 F1 明显提升，适合展示阈值取舍。

## 1.1 旧 baseline_comparison

旧单工作点 `baseline_comparison/` 已归档到：

```text
attack/_archive/cleanup_20260415/legacy_results/baseline_comparison_old/
```

这里的 `TCN-GAN (Ours)` 是旧模型/旧口径，不建议作为最终论文主表直接引用。

## 2. final_experiments/20260414_132416

目录：

```text
attack/results/final_experiments/20260414_132416/
```

用途：当前论文主线最终实验。

该目录的逻辑顺序是：

```text
stride sweep -> alpha sweep -> threshold trade-off -> ablation
```

### 2.1 selected_main_config.json

文件：

```text
attack/results/final_experiments/20260414_132416/selected_main_config.json
```

用途：记录冻结后的主模型配置。

当前冻结配置：

```text
window_size = 128
stride      = 16
score_alpha = 0.24
disc_pooling = attn
gan_loss      = wgan-gp
score_mode    = fused
target_fpr    = 0.05
```

论文中所有“最终模型”都应该优先对齐这份配置。

### 2.2 stride_sweep

目录：

```text
attack/results/final_experiments/20260414_132416/stride_sweep/
```

用途：在固定窗口长度后，比较不同 stride 对模型性能和计算效率的影响。

核心结果：

```text
attack/results/final_experiments/20260414_132416/stride_sweep/combined/tcn_gan_autotune_results.csv
```

论文中可以用于说明：

- stride 越小，窗口重叠越多，实时监测更细，但计算量更大。
- stride 越大，计算更快，但可能损失异常边界信息。
- 当前主线选择 `stride=16`。

### 2.3 alpha_sweep

目录：

```text
attack/results/final_experiments/20260414_132416/alpha_sweep/
```

用途：固定模型 checkpoint 后，扫描 fused anomaly score 的融合比例 `alpha`。

主要文件：

- `alpha_sweep.csv`
  - 每个 alpha 的数值结果。

- `alpha_sweep.png`
  - alpha 对 F1、recall、precision、FPR 等指标的影响图。

- `meta.json`
  - 记录最佳 alpha。

当前选择：

```text
score_alpha = 0.24
```

论文中可以说明：

> alpha 控制判别器概率分数和特征距离分数之间的融合比例。通过 alpha sweep，本文选择在校准集上 F1 表现较优的融合权重。

### 2.4 threshold_curve

目录：

```text
attack/results/final_experiments/20260414_132416/threshold_curve/
```

用途：观察不同阈值下 recall、precision、FPR 的取舍。

主要文件：

- `threshold_tradeoff.csv`
  - 阈值扫描的原始表格。

- `threshold_tradeoff.png`
  - 阈值取舍曲线。

- `scores_w128_s16_a024.csv`
  - 每个窗口的异常分数。

这部分适合用于论文讨论“实时监测系统如何根据误报容忍度选择阈值”。

当前重要观察：

```text
Recall >= 0.90 时，FPR 约为 0.147
Recall >= 0.95 时，FPR 约为 0.158
```

说明如果强行追求 90% 以上召回率，误报率会明显上升。当前最终模型选择更低误报的工作点。

### 2.4.1 tcn_fpr_sweep

目录：

```text
attack/results/final_experiments/20260414_132416/tcn_fpr_sweep/
```

用途：最终 Attentive TCN-WGAN-GP 模型在多个 `target_fpr` 下的阈值工作点扫描。

生成命令：

```bash
bash attack/pipelines/run_tcn_fpr_sweep.sh
```

主要文件：

- `tcn_gan_fpr_sweep.csv`
  - 和 baseline FPR sweep 对齐的 Attentive TCN-WGAN-GP 结果表。

- `tcn_gan_fpr_sweep.md`
  - 方便阅读的 Markdown 表格。

注意：

该实验不重新训练模型，只加载最终 checkpoint 并在多个目标误报率下重新标定阈值。

关键结论：

```text
target_fpr=0.05: F1=0.7958, Recall=0.6640, Precision=0.9928, Test FPR=0.0036
target_fpr=0.15: F1=0.8783, Recall=0.9678, Precision=0.8040, Test FPR=0.1729
```

论文建议：

- 主表可以使用 `target_fpr=0.05`，强调低误报、高 precision。
- 阈值敏感性图可以展示 `target_fpr=0.01~0.30`，说明模型在更高召回工作点下仍优于多数 baseline。

### 2.5 ablation_2x2

目录：

```text
attack/results/final_experiments/20260414_132416/ablation_2x2/20260414_204232/
```

用途：固定 `window=128, stride=16, alpha=0.24` 后，比较：

```text
disc_pooling: mean vs attn
gan_loss:     vanilla vs wgan-gp
```

主要文件：

- `ablation_summary.csv`
  - 推荐作为论文消融表的主要数据来源。

- `ablation_summary.md`
  - Markdown 版本，方便阅读。

- `ablation.csv`
  - 更完整的消融结果。

四个子目录对应四组实验：

```text
20260414_204232 = mean + vanilla
20260414_210038 = attn + vanilla
20260414_212058 = mean + wgan-gp
20260414_225204 = attn + wgan-gp
```

每个子目录中常见文件：

- `ckpt_w128_s16.pt`
  - 当前消融组训练得到的最佳 checkpoint。

- `train_w128_s16.log`
  - 训练日志。

- `eval_w128_s16_fused.log`
  - 普通评估日志。

- `eval_best_xai_w128_s16.log`
  - 带 XAI 的评估日志。

- `eval_w128_s16_fused.json`
  - 机器可读的评估结果，汇总表从这里读取。

- `tcn_gan_autotune_results.csv`
  - 当前消融组的一行摘要。

- `tcn_gan_autotune_rank_plot.png`
  - 自动生成的排序图。因为每个消融子目录通常只有一个配置，所以论文中一般不使用。

当前消融结论需要结合工作点谨慎表述：

- 在严格低误报 `target_fpr=0.05` 附近，`attn + wgan-gp` precision 高、FPR 低，但 recall 偏保守。
- 在高召回 `target_fpr=0.15` 附近，`attn + wgan-gp` 的 F1 和 recall 更适合作为论文消融主结论。
- `attn + vanilla` 表现较差，说明 attention 单独加入不稳定。
- attention 更适合作为与 WGAN-GP 配合的时间证据聚合模块，而不是简单保证所有指标提升。

### 2.5.1 ablation FPR sweep

目录：

```text
attack/results/final_experiments/20260414_132416/ablation_2x2/20260414_204232/fpr_sweep/
```

用途：四个消融模型在多个 `target_fpr` 下的重新评估结果。

生成命令：

```bash
bash attack/pipelines/run_ablation_fpr_sweep.sh
```

主要文件：

- `ablation_fpr_sweep.csv`
- `ablation_fpr_sweep.md`

注意：该实验不重训，只加载四个已有消融 checkpoint。

推荐论文表述：

> 消融结果表明，注意力机制单独加入时并不一定提升整体 F1；但与 WGAN-GP 结合后，模型在较高召回工作点取得更好的 F1/Recall，同时保留 attention 权重用于解释时间证据聚合。这说明 attention 需要稳定的对抗训练目标配合，而不是一个孤立生效的模块。

## 3. xai_tcn

目录：

```text
attack/results/xai_tcn/eval_w128_s16_fused_w128_s16_fused_a0.24/
```

用途：最终主模型的可解释性分析。

当前 XAI 方法：

```text
gradxinput
```

含义：计算输入特征对异常分数的梯度贡献，用于估计哪些时间位置、哪些流量特征对模型判断更重要。

### 3.1 xai_time_importance.png

文件：

```text
xai_time_importance.png
```

含义：窗口内部不同时间位置的重要性。

横轴表示窗口内的时间步，当前窗口长度为 128。纵轴表示该时间位置对异常分数的贡献强度。

怎么看：

- 如果异常样本曲线整体高于正常样本，说明模型在异常窗口中捕捉到了更强的异常证据。
- 如果某些时间位置有明显峰值，说明模型主要依据这些局部时刻做出异常判断。
- 如果曲线比较平缓，说明模型判断更多依赖整个窗口的累积模式，而不是单个尖峰。

论文中可以用于说明：

> 模型不仅输出异常分数，还能定位窗口内部较重要的时间片，从而辅助分析异常发生阶段。

### 3.2 attn_weights.png

文件：

```text
attn_weights.png
```

含义：attention pooling 在不同时间位置上的平均权重。

注意：

attention weight 和 XAI importance 不是完全一样的东西。

- attention weight 表示模型聚合序列特征时更偏向哪些位置。
- gradxinput importance 表示这些位置对最终异常分数的梯度贡献。

两者可以互相印证，但不能简单等同。

论文中可以用于说明：

> 注意力权重提供了模型在时间维度上的关注分布，使检测结果具有一定可解释性。

### 3.3 xai_feature_importance.png

文件：

```text
xai_feature_importance.png
```

含义：不同网络流量特征对异常分数的贡献。

当前异常样本中贡献最高的特征包括：

```text
PSH Flag Count
ACK Flag Count
Flow Packets/s
Fwd Header Length
Fwd Header Length.1
Init_Win_bytes_forward
min_seg_size_forward
Bwd Header Length
Destination Port
Bwd Packet Length Std
```

这些特征大多与 TCP 标志位、包速率、头部长度、窗口大小、端口和包长统计有关，符合网络攻击检测中常见的流量行为变化。

论文中可以用于说明：

> XAI 结果显示，模型的异常判断主要受 TCP 控制标志、包速率、包长统计和端口相关特征影响。这与入侵检测中异常连接行为、突发流量和协议交互异常的经验认识一致。

### 3.4 xai_report.json

文件：

```text
xai_report.json
```

用途：XAI 的数值结果。

里面包含：

- `time_importance`
  - 正常样本和异常样本在每个时间位置上的平均重要性。

- `feature_importance`
  - 每个特征对正常样本和异常样本的平均贡献。

- `attn_weights`
  - attention pooling 的平均时间权重。

- `top_features_by_anomaly_importance`
  - 异常样本中贡献最高的特征排名。

## 论文使用建议

当前结果可以支撑一篇论文初稿，但建议按以下方式写：

1. baseline 对比作为主性能结果。
2. window/stride/alpha sweep 说明参数选择过程。
3. 消融实验说明 attention 与 WGAN-GP 的作用和局限。
4. XAI 作为解释性亮点，说明模型能指出关键时间片和关键流量特征。
5. 不要写成“attention 一定提升所有指标”。当前结果更适合写成“attention + WGAN-GP 带来低误报、高精度和可解释性”。

如果目标是高水平会议或期刊，建议补充：

- 多随机种子实验，报告均值和标准差。
- 使用最终冻结配置重新确认 baseline 表。
- 报告 `Recall@FPR=1%/5%` 或 `FPR@Recall=90%`。
- 对 XAI 做少量案例分析，而不只放平均图。
