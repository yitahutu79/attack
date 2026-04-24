# 实验设计状态指南（2026-04-24 更新）

这份文档原本是从审稿人角度列出的“理想实验清单”。现在项目已经推进到论文收尾阶段，所以这里改成当前状态版：哪些建议已经完成，哪些只做了轻量版本，哪些可以作为局限或后续工作，不再误导你重复跑已经完成的实验。

当前最终事实以以下文件为准：

- `paper/main.tex`
- `docs/CURRENT_PAPER_GUIDE_CN.md`
- `docs/RECENT_EXTERNAL_BASELINE_RUNS_CN.md`
- `docs/TCN_GAN_THESIS_EXPERIMENT_RECORD.md`

## 1. 当前论文实验主线

论文现在不是只证明“模型分数最高”，而是证明：

> `Attentive TCN-WGAN-GP` 在窗口级网络异常检测中，能在低误报约束下取得更实用的告警质量，并且能提供可解释证据。

当前四个数据集的角色：

| 数据集 | 当前定位 | 是否进入主文 |
|---|---|---:|
| `CICIDS2017` | 主战场，低 FPR 主结论 | 是 |
| `SWaT` | 跨域强验证，工业控制场景 | 是 |
| `UNSW-NB15` | 压力测试，说明阈值迁移困难 | 是 |
| `TON_IoT` | 域外局限，不作为胜利点 | 是，简写为 limitation |

## 2. 原建议完成情况

| 审稿人关心点 | 原建议 | 当前状态 | 论文处理 |
|---|---|---|---|
| 多数据集验证 | 至少 2-3 个数据集 | 已完成 | `CICIDS/SWaT/UNSW/TON` |
| 同口径窗口评估 | 固定窗口、步长、指标 | 基本完成 | 主模型与多数 baseline 使用 `W=128, stride=16` |
| 现代 baseline | 加入最新模型 | 已完成 | `GANomaly/DeepSVDD/TranAD/ALoRa/Anomaly Transformer/TimesNet/DLinear/Autoformer` |
| 传统 baseline | IForest/OCSVM/RF/MLP | 已完成 | CICIDS 主表保留；SWaT/UNSW 不放 RF/MLP 是合理的 |
| 消融实验 | attention、WGAN-GP、融合分数 | 已完成主要版本 | 2x2 ablation + alpha sweep |
| 参数敏感性 | window/stride/alpha | 已完成主要版本 | `W=128, stride=16, alpha=0.24` |
| 效率分析 | 推理时间、模型大小 | 已完成轻量版 | 表中报告 windows/s 和 MB |
| 鲁棒性 | 噪声、缺失、漂移 | 部分完成 | 已有 Gaussian noise sanity check；缺失/漂移留作未来工作 |
| XAI | 时间、特征、案例解释 | 已完成 | Attention + Integrated Gradients + SHAP |
| 多随机种子 | mean±std + 显著性检验 | 部分/不足 | 主文谨慎表述，不强写统计显著优势 |

## 3. 现在不建议继续追的旧项

这些不是没价值，而是以当前论文收尾节奏看，继续做的收益不高：

- `NSL-KDD`：数据集偏旧，加入后不一定比 SWaT/UNSW 更有说服力。
- `LOF/GRU-AE/VAE/DAGMM/THOC`：baseline 已经足够多，再加容易稀释主线。
- 完整缺失值/概念漂移实验：适合扩展论文或 revision，不适合现在打乱主表。
- 把 `TON_IoT` 当作主胜利点：当前结果不支持，应继续作为域外局限。

## 4. 仍需注意的真实风险

### 4.1 `ALoRa` 窗口口径

旧结果曾是 `W=20`，而主表里大多数模型是 `W=128`。  
现在 `ALoRa@W128` 已经完成并落盘；如果主文与汇报页都改用新值，就可以再考虑是否删除表格里的 `W` 列。

### 4.2 target FPR 口径

主文目前同时使用两个 operating points：

- `target FPR = 0.05`：严格低误报。
- `target FPR = 0.15`：更高召回压力测试。

写作时要明确每张表的目标 FPR，尤其是 `UNSW-NB15` 的压力测试表。

### 4.3 RF/MLP 的解释

RF/MLP 是监督模型，需要攻击样本训练。  
在 `SWaT/UNSW` formal one-class 设置下，训练集主要是 benign，所以不放 RF/MLP 是合理的；在 `CICIDS` 中有可用于监督训练的窗口标签，因此可以比较。

### 4.4 统计显著性

目前不建议在论文里写“统计显著优于所有 baseline”。  
更稳的写法是强调：

- 低误报约束下的 F1/FPR trade-off；
- 跨数据集的行为差异；
- UNSW/TON 上的局限与阈值迁移风险。

## 5. 当前实验部分推荐结构

论文 `Results` 章节现在适合按问题组织：

1. `RQ1: Low-False-Positive Detection Performance`  
   说明 CICIDS2017 主表、现代 baseline 表、高召回工作点。

2. `RQ2: Cross-Dataset Threshold Transfer`  
   说明 SWaT 强验证、UNSW 压力测试、TON 局限。

3. `RQ3: Hyperparameter Sensitivity`  
   说明 `W=128`、`stride=16`、`alpha=0.24` 的选择依据。

4. `RQ4: Ablation Study`  
   说明 attention 与 WGAN-GP 的组合效果，不夸大 attention 单独收益。

5. `RQ5: Real-Time Monitoring Feasibility`  
   说明 CPU-only 推理吞吐和模型大小。

6. `RQ6: Alarm Auditability`  
   说明 Attention/IG/SHAP 的多视角解释。

## 6. 接下来最有价值的行动

1. 回填后的 `ALoRa@W128` 数值在 `main.tex`、`presentation.html` 和外部 baseline 记录中保持一致。
2. 如果所有主表模型都统一为 `W=128`，再删除表格中的 `W` 列，让版式更干净。
3. 最后统一检查 `target FPR`、`test benign FPR`、`precision/recall/F1` 是否在论文、汇报页、记录文档中一致。
4. 语言上避免绝对化表述，主张控制在“低误报窗口级检测更实用”这一条线上。

## 7. 一句话结论

这份旧指南里的大部分核心建议已经做完了。现在最重要的不是继续无限加实验，而是把已经完成的实验口径统一、表格收干净，并把论文主张写得稳。
