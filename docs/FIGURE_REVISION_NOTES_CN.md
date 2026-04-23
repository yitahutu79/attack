# 当前论文图示修改说明

更新时间：2026-04-24

当前论文正文中实际使用的两张核心方法图是：

- `paper/figures/framework_overview.png`
- `paper/figures/model_architecture.png`

对应 `main.tex` 中的：

- `Fig. framework-overview`
- `Fig. model-architecture`

这份文档不再沿用旧的 `fig1.png / fig2.png` 命名，而是直接按当前论文图文件来说明应该怎么修改、怎么讲、哪些地方最值得继续打磨。

## 1. 当前整体判断

两张图现在已经能支撑论文主线，不需要推倒重画。  
当前最值得做的是：

1. 保证图内术语和正文完全一致。
2. 让“低误报告警流程”和“模型内部结构”这两条线更清楚。
3. 为论文版和汇报版分别保留最适合的简洁程度。

## 2. 总体原则

### 2.1 术语统一

图里出现的关键词，最好和 `paper/main.tex` 完全一致：

- `Attentive TCN-WGAN-GP`
- `Sliding Windows`
- `Fused Anomaly Score`
- `Threshold Calibration`
- `Online Alarm`
- `Attention Pooling`
- `WGAN-GP`
- `Feature-Deviation Score`

如果正文写的是 `critic`，图里就尽量别混用 `discriminator`；如果必须写双称呼，也建议写成：

```text
TCN Critic / Discriminator
```

### 2.2 图本体不要自带 caption

LaTeX 已经有 `\caption{...}`，图底部不要再嵌 `Fig. 1` / `Fig. 2` 之类文字。

### 2.3 视觉层级保持克制

建议控制在 2 到 3 个字号层级：

- 外层模块标题
- 子模块标题
- 普通标签 / 公式 / 注释

这篇论文是安全检测和工程实验导向，不需要过度装饰。

## 3. `framework_overview.png` 应该怎么改

这张图对应论文里“从 flow 到告警再到 XAI 审计”的总流程。

### 3.1 这张图现在最该突出什么

当前论文主张已经比旧版更明确，所以总览图建议突出四步：

1. 原始 flow records 按时间排序。
2. 切成 sliding windows。
3. `Attentive TCN-WGAN-GP` 输出 anomaly score。
4. 经过 benign false-positive budget calibration 后变成在线告警，并附带 XAI 审计证据。

### 3.2 建议保留的元素

- `Ordered Flow Records`
- `Sliding Window` 示例，保留 `W=128, stride=16`
- 模型主框
- `Fused Score -> Threshold Calibration -> Online Alarm`
- XAI / audit 输出

### 3.3 建议弱化或移除的元素

- 底部单独列出的 `Claims / Evaluation / XAI` 大块说明，如果已经和右侧流程重复，可以缩小或删掉。
- 与主数据流无关的装饰箭头。
- 太多解释性小字，尤其是汇报版里一页投影看不清的注释。

### 3.4 当前最值得检查的细节

- 拼写有没有残留旧错误，例如 `Ordered`、`Anomalous`、`Online Alarm`。
- 模型名是否已经统一成 `Attentive TCN-WGAN-GP`。
- 右侧是否明确写出“不是直接硬阈值”，而是 `Threshold Calibration`。
- 如果图里还写 `interpretable`、`explainability` 等词，最好和正文里 `audit / XAI evidence` 的说法一致。

### 3.5 汇报时怎么讲这张图

一句话讲法：

> 这张图说明本文不是做单条 flow 分类，而是把按时间排序的流量切成窗口，输出窗口级异常分数，在给定误报率预算下完成阈值标定，再把最终告警交给分析员审计。

## 4. `model_architecture.png` 应该怎么改

这张图对应论文里的方法内部结构。

### 4.1 当前这张图要服务的重点

当前 `main.tex` 的方法部分重点已经很明确，图里最应该服务这三件事：

1. `TCN` 负责提取时间特征。
2. `attention pooling` 负责非均匀地聚合时间步。
3. `WGAN-GP + fused scoring` 共同构成最终检测器。

### 4.2 图里一定要讲清的关系

- `Generator` 生成 benign-like windows。
- `Critic` 对真实窗口和生成窗口打分。
- `Attention Pooling` 是 critic 内部的 temporal aggregation，不是独立 XAI 模块。
- `Fused Score` 在推理阶段把 critic score 和 feature-deviation score 融合。

### 4.3 当前建议统一的符号

和正文一致即可：

- `D(X)`：critic score
- `S_D(X)`：critic unknownness score
- `S_F(X)`：feature-deviation score
- `S(X) = alpha S_D(X) + (1-alpha) S_F(X)`
- `alpha = 0.24`

如果图里出现 loss，建议和正文风格统一为：

- `WGAN-GP Critic Loss`
- `Gradient Penalty`

### 4.4 最容易混乱的地方

1. `attention` 和 `XAI` 混成一回事  
   图里可以画 attention weights，但不要让读者误以为 attention 就是全文唯一解释方法。

2. `training objective` 和 `inference path` 混在一起  
   最好用不同颜色或虚线区分：
   - 训练阶段：generator / critic / gradient penalty
   - 推理阶段：critic evidence + feature deviation -> fused score

3. generator 和 critic 的数据流太拥挤  
   如果箭头交叉很多，阅读负担会很大。建议宁可拉开布局，也不要省这点空间。

### 4.5 当前值得强调的视觉重点

- `Temporal Attention Pooling`
- `WGAN-GP`
- `Fused Score`

因为这三个点最能体现“当前版本论文”与基础 TCN-GAN 的区别。

## 5. 与当前论文叙事要对齐的地方

由于论文现在已经加入 `SWaT`、`UNSW-NB15`、`TON_IoT` 和现代 baseline，对图的期待也和旧版不一样了。图不需要再承担“展示所有实验细节”的任务，而应该服务下面这条主线：

> 用窗口级时间建模和低误报阈值标定，构造一个可审计的网络异常告警器。

所以图里不需要塞进这些内容：

- 所有 baseline 名字
- 所有数据集名字
- 全部实验表格数字
- 太详细的 cross-dataset 解释

这些属于正文表格和 discussion，不属于方法图。

## 6. 论文版和汇报版建议分开

### 论文版

- 术语精确
- 箭头关系完整
- 公式和模块名与正文完全一致
- 信息稍密一些可以接受

### 汇报版

- 缩减细节文字
- 放大关键链路
- 让一眼就能看出 `flow -> window -> score -> calibration -> alarm`
- 避免投影时看不清的小字

## 7. 当前优先级

如果时间有限，建议按这个顺序打磨：

1. `framework_overview.png`  
   先保证总流程和术语统一，因为这是听众最先看到的图。

2. `model_architecture.png`  
   再把 attention pooling、WGAN-GP、fused score 的结构关系讲清。

3. 如果做汇报版，再单独导出更简洁的 presentation 版本。

## 8. 一句话总结

当前两张核心图不需要重画，最重要的是把它们从“旧版项目示意图”打磨成“完全贴合当前 `main.tex` 叙事的论文图”：总览图强调窗口级告警流程，结构图强调 attention、WGAN-GP 和 fused scoring 的组合关系。
