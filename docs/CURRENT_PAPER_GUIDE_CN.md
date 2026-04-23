# 当前论文版本中文导览（零基础版）

更新时间：2026-04-24

这份文档写给“第一次接触这个项目”的同学。  
你不需要先懂 GAN、TCN、FPR，也能看明白我们做了什么、做到哪了、接下来要做什么。

## 1. 这篇论文到底在解决什么问题

一句话：  
我们在做一个网络入侵检测模型，重点不是“离线分类准确率最高”，而是“实际告警时误报别太多、还能解释为什么报警”。

为什么这件事重要：

- 安全运营里最怕“告警风暴”（正常流量被大量误报）。
- 只看 AUC/F1 不够，必须看“误报成本”。
- 只给黑盒分数不够，最好能给分析师可读的证据。

## 2. 我们的方法（不讲数学版）

模型名：`Attentive TCN-WGAN-GP`

可以理解成三步：

1. 把连续流量切成“窗口”（一小段时间序列），不是一条一条独立判断。  
2. 让模型学习“正常流量长什么样”，偏离正常就打高分。  
3. 用阈值把分数变成告警，并做解释（Attention/IG/SHAP）告诉你“为什么报”。

## 3. 你只需要记住的核心结论

当前论文叙事是：

- `CICIDS2017`：主战场，低误报表现最强。
- `SWaT`：跨域验证，结果也很强。
- `UNSW-NB15`：压力测试，结果有竞争力但不是第一。
- `TON_IoT`：明确写成局限，不回避差结果。

这不是“所有数据集都第一”的论文，而是“在低误报告警目标下更实用”的论文。

## 4. Baseline 覆盖（现在已经补齐）

四类对照模型：

- 传统无监督：IsolationForest、OneClassSVM
- 监督窗口模型：RF、MLP
- 深度序列模型：LSTM-AE、LSTM-AD
- 现代异常检测：GANomaly、DeepSVDD、TranAD、ALoRa、Anomaly Transformer、TimesNet、DLinear、Autoformer

说明：

- `DLinear/Autoformer` 来自 `Time-Series-Library`，已跑 `CICIDS/SWaT/UNSW`。
- `ALoRa` 当前主文里窗口长度是 `W=20`；`W=128` 重跑已启动过但未完成落盘，用于统一口径前需要重跑或清理残留进程。

## 5. 当前最重要文件

- 论文主文：[paper/main.tex](/Users/lijie/Desktop/work/attack/paper/main.tex)
- 参考文献：[paper/references.bib](/Users/lijie/Desktop/work/attack/paper/references.bib)
- 已编译论文：[paper/main.pdf](/Users/lijie/Desktop/work/attack/paper/main.pdf)
- 汇报页：[presentation.html](/Users/lijie/Desktop/work/attack/presentation.html)
- 外部 baseline 跑数记录：[docs/RECENT_EXTERNAL_BASELINE_RUNS_CN.md](/Users/lijie/Desktop/work/attack/docs/RECENT_EXTERNAL_BASELINE_RUNS_CN.md)

## 6. 现在做到哪了（进度快照）

已完成：

- 论文主线结构（引言/方法/实验/讨论/结论）基本成型。
- 现代 baseline 已并入主文表格，不再只停留在“计划”。
- Time-Series-Library 引用链已补（TimesNet + TSLib survey）。
- 目录清理已做：明显过时/缓存文件已归档到 `_archive/cleanup_20260424`。

待确认/待处理：

- `ALoRa @ W=128` 三数据集重跑已启动过，但当前未完成落盘；需要清理残留 wrapper 或重新启动后，再决定是否统一去掉表格 `W` 列。

## 7. 为什么有些表里没有 RF/MLP

这是很多人第一次会疑惑的点。

- 在 `SWaT/UNSW` 的 formal one-class split 下，训练集基本是 benign。  
- RF/MLP 这种监督分类模型需要正负样本标签，训练条件不成立或不公平。  
- 所以它们不放在那两张主对比表里是正常且合理的。

而 `CICIDS` 当前划分里有可用于监督训练的攻击窗口，因此能比较 RF/MLP。

## 8. 当前存在的风险（审稿人最可能卡的点）

1. 表格口径混合  
   例如部分模型是 `target_fpr=0.05`，有些表是 `0.15`，如果说明不清会被质疑。

2. 窗口长度不统一  
   例如 `ALoRa` 仍是 `W=20`，其余多是 `W=128`。如果去掉 `W` 列而不统一重跑，风险很高。

3. 贡献边界要写清  
   审稿人会问“你的核心创新到底是结构、训练策略还是评估协议”。主文要一直围绕同一条主线。

4. 引用完整性  
   新引入的 baseline 若无来源引用，会被认为不严谨（这部分已在补齐）。

## 9. 接下来建议按这个顺序推进

1. 清理或重启 `ALoRa@W128`，等它跑完并落盘。  
2. 决定是否“全表统一去掉 W 列”：  
   只有在窗口长度完全统一后才建议去掉。  
3. 固化最终表格与文字解释（尤其 target FPR 口径说明）。  
4. 最后再做语言润色与版式收尾。

## 10. 给新同学的最短阅读路径（15 分钟）

1. 先看 [presentation.html](/Users/lijie/Desktop/work/attack/presentation.html) 前 5 页，理解问题和方法。  
2. 再看 [paper/main.tex](/Users/lijie/Desktop/work/attack/paper/main.tex) 的 `Experimental Setup` 和 `Results`。  
3. 最后看 [docs/RECENT_EXTERNAL_BASELINE_RUNS_CN.md](/Users/lijie/Desktop/work/attack/docs/RECENT_EXTERNAL_BASELINE_RUNS_CN.md)，知道外部模型怎么跑、结果在哪。

如果只看一句总结：

> 这篇工作强调“低误报、可解释、可落地”的窗口级入侵检测，而不是只追求离线指标漂亮。
