# Results 主表 / 补充表方案与正文起草（当前版）

更新时间：2026-04-24

这份文档已经从“结果写作计划稿”改成当前版说明。它的目标不再是提醒你未来可能写什么，而是告诉你：现在结果结构已经怎么定了、主表和补充表应该如何分工、正文应该围绕什么主张来写，避免再把上周的临时判断误当成当前结论。

当前事实请优先以以下文件为准：

- `paper/main.tex`
- `docs/CURRENT_PAPER_GUIDE_CN.md`
- `docs/RECENT_EXTERNAL_BASELINE_RUNS_CN.md`
- `docs/TCN_GAN_THESIS_EXPERIMENT_RECORD.md`

## 1. 当前结果结构已经基本定型

现在的 `Results` 不应再按“哪个实验刚跑出来就堆哪张表”的方式组织，而应该围绕论文主张来组织：

> `Attentive TCN-WGAN-GP` 在窗口级异常检测中，重点优势是低误报约束下更实用的告警质量，并且能提供可审计的解释证据。

对应到结果层面，当前四个数据集的分工已经比较明确：

| 数据集 | 当前角色 | 写作目标 |
|---|---|---|
| `CICIDS2017` | 主战场 | 证明核心低误报结论 |
| `SWaT` | 跨域强验证 | 证明结论不只在单一 benchmark 成立 |
| `UNSW-NB15` | 压力测试 | 证明方法在更难迁移场景下仍有竞争力 |
| `TON_IoT` | 域外局限 | 诚实展示 dataset shift 与 threshold sensitivity |

因此，当前写作重点不是把所有数据集包装成“全面领先”，而是把不同数据集承担的证据角色写清楚。

## 2. 主文里最值得保留的表

### 2.1 CICIDS2017 主表

这仍然是正文的核心证据，建议继续保留两张 operating-point 表：

1. `target FPR = 0.05`
2. `target FPR = 0.15`

这两张表共同承担的叙事是：

- 在严格低误报预算下，模型具备最强的实用告警质量。
- 当 operating point 放宽到更高召回时，方法仍保持有竞争力的 precision/recall/F1 trade-off。

这里仍然建议保留监督模型和无监督模型共存的完整参照系，但正文必须明确说明：

- `CICIDS2017` 当前划分支持监督窗口模型训练；
- 因此 `RF/MLP` 在这张主表中是合理比较对象。

### 2.2 SWaT 跨数据集主验证表

`SWaT` 现在应该被视为主文里的正式第二证据，而不是补充材料里的可选项。

推荐保留内容：

- `\model{}`
- `OneClassSVM`
- `IsolationForest`
- `LSTM-AE`
- `LSTM-AD`
- `TranAD`（如果当前版本排版允许）

这张表要承担的不是“全面吊打所有 baseline”，而是：

- 说明低误报设定下，方法的优势能迁移到工业控制场景；
- 证明这不是只在 `CICIDS2017` 上成立的偶然结果。

对于 `DeepSVDD` 这类当前结果口径仍可疑、阈值行为异常的模型，不建议为了“看起来 baseline 更多”而强塞进主表。

### 2.3 轻量效率表

当前正文里最适合新增或保留的一张小表，仍然是效率总结表。

推荐列：

- Dataset
- Train windows
- Test windows
- Training time
- Throughput
- Checkpoint size

它的作用不是证明我们是最快模型，而是支撑下面这条写法：

> 该方法不仅在低误报约束下表现强，而且在窗口级在线监测场景中具备现实可部署性。

## 3. 更适合放到补充材料的内容

### 3.1 SWaT mixed split 历史结果

这类表现在更适合用来说明“评估协议会显著影响 anomaly detection 结论”，而不是继续放在正文里抢主线。

### 3.2 TON_IoT 完整对比表

`TON_IoT` 目前最适合的定位是 appendix 或 discussion supporting evidence。

推荐写法：

- 不把它写成优势主表；
- 把它写成更强 domain shift 场景下的 challenge dataset；
- 用来支撑 limitations，而不是胜利结论。

### 3.3 Cross-dataset mini-ablation / seed summary

这些内容有价值，但更适合起到补充证明作用：

- 说明 attention 收益在部分跨数据集场景下可以复现；
- 说明结果不是完全由单次随机初始化决定；
- 但不要让这些附表喧宾夺主。

## 4. Results 章节当前最稳的结构

建议现在把正文稳定在下面这套结构，而不是继续频繁改大纲：

### 4.1 `RQ1: Low-False-Positive Detection Performance`

内容重点：

- `CICIDS2017` 两张主表
- FPR sweep
- 强调 low-FPR operating point 的实用性

### 4.2 `RQ2: Cross-Dataset Validation`

内容重点：

- `SWaT` 作为第二主证据
- `UNSW-NB15` 作为压力测试
- `TON_IoT` 作为 limitation

这部分最重要的是把叙事顺序固定下来：

1. 先讲为什么采用更合理的 anomaly-detection protocol。
2. 再给 `SWaT` 强结果。
3. 然后讲 `UNSW-NB15` 的压力测试表现。
4. 最后诚实说明 `TON_IoT` 的局限。

### 4.3 `RQ3: Hyperparameter Sensitivity`

内容重点：

- `W=128`
- `stride=16`
- `alpha=0.24`

这部分已经足够，不建议继续无限增加调参图。

### 4.4 `RQ4: Ablation Study`

内容重点：

- attention pooling
- `WGAN-GP`
- fused score

写法上要避免过度夸大 attention 单独贡献，更稳的是强调：

- 模块组合共同形成了当前最优行为；
- 不同组件的收益在不同数据集上存在差异。

### 4.5 `RQ5: Real-Time Monitoring Feasibility`

内容重点：

- CPU-only throughput
- checkpoint size
- 训练与推理成本的轻量分析

### 4.6 `RQ6: Alarm Auditability`

内容重点：

- attention
- Integrated Gradients
- SHAP

强调这是“告警审计能力”，不是宣称模型解释已经完美解决。

## 5. 现在最需要统一的写作口径

和旧版计划稿相比，当前真正危险的不是“漏写一张表”，而是不同文档之间说法不一致。

最需要统一的几点：

1. `target FPR = 0.05` 与 `0.15` 的角色区分。
2. 哪些表是 formal one-class split，哪些不是。
3. 为什么 `RF/MLP` 在 `SWaT/UNSW` 主表中不出现。
4. 为什么 `TON_IoT` 属于 limitation 而不是主胜利点。
5. `ALoRa` 仍有窗口长度口径问题，不能在未完成统一前假装无差异。

## 6. 当前版英文写作原则

如果现在要继续改 `paper/main.tex`，建议始终围绕下面几条原则：

1. 不写“our method wins everywhere”。
2. 多写“our method is especially effective under low false-positive constraints”。
3. 不把 `TON_IoT` 包装成成功复现集。
4. 不把 attention 直接写成 faithful explanation。
5. 强调方法的价值是“practical alert quality + auditability”，而不是只看离线 AUC。

## 7. 下一步最值钱的工作

如果按投入产出比排序，接下来最值得做的是：

1. 确认 `ALoRa@W128` 最终状态，并决定是否统一去掉表格中的 `W` 列。
2. 统一 `paper/main.tex`、`presentation.html` 和记录文档中的表格口径说明。
3. 检查各张表的 `precision/recall/F1/test benign FPR` 表述是否完全一致。
4. 最后再做 Results 英文润色和 appendix 收束。

## 8. 一句话结论

这份文档现在应被当成“当前结果写作说明”，不是“未来写作计划”。主表结构已经基本定型：`CICIDS2017` 负责核心结论，`SWaT` 负责跨域强验证，`UNSW-NB15` 负责压力测试，`TON_IoT` 负责 limitation；接下来最重要的是统一口径，而不是继续把结果章节写成不断扩张的实验仓库。
