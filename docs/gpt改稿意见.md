下面这个单子按 **Computers & Security 期刊投稿** 的标准整理，分成“必须补”“强烈建议补”“有时间再补”。你可以直接拿去和 Codex/实验脚本对接。

---

# 一、必须补的实验

## 1. 数据划分与窗口统计实验

**目的：** 证明你的实验没有数据泄漏，且每个数据集的样本规模清楚。

需要统计：

| 数据集        | Train benign windows | Calibration benign windows | Test benign windows | Test anomaly windows | Attack types |
| ---------- | -------------------: | -------------------------: | ------------------: | -------------------: | ------------ |
| CICIDS2017 |                      |                            |                     |                      |              |
| SWaT       |                      |                            |                     |                      |              |
| UNSW-NB15  |                      |                            |                     |                      |              |

同时统计：

* 原始 flow/record 数量；
* 生成 window 数量；
* window size；
* stride；
* anomaly ratio threshold；
* test benign FPR 的分母，例如 `0 / 350`，不要只写 `0.0000`。

**特别注意：**
论文里必须统一窗口标签规则。你现在不能一处写 “any attack flow”，另一处写 “attack ratio 0.15”。建议统一成：

> A window is labeled anomalous if at least 15% of its records are labeled as attacks.

---

## 2. 主结果多随机种子实验

**目的：** GAN 模型随机性较强，单次结果不够可信。

至少做：

| 数据集        | 建议 seed 数 | 模型                      |
| ---------- | --------: | ----------------------- |
| CICIDS2017 |   5 seeds | Proposed + 主要 baselines |
| SWaT       |   3 seeds | Proposed + 主要 baselines |
| UNSW-NB15  |   3 seeds | Proposed + 主要 baselines |

如果算力有限，至少做：

* Proposed；
* MLP；
* RF；
* GANomaly；
* DeepSVDD；
* TranAD；
* OneClassSVM。

输出表格：

| Model | F1 mean ± std | Recall mean ± std | Precision mean ± std | Test FPR mean ± std |
| ----- | ------------: | ----------------: | -------------------: | ------------------: |

---

## 3. TAD Score 评分消融实验

**目的：** 证明 TAD Score 不是简单包装，确实比单独 score 更有效。

至少比较：

| Score mode             | AUC | AP | F1 | Recall | Precision | Test FPR |
| ---------------------- | --: | -: | -: | -----: | --------: | -------: |
| Critic score only      |     |    |    |        |           |          |
| Feature deviation only |     |    |    |        |           |          |
| TAD fusion, α=0.24     |     |    |    |        |           |          |
| TAD fusion, α=0.5      |     |    |    |        |           |          |

建议在 **CICIDS2017** 上先做，SWaT/UNSW 可选做。

这个实验非常关键，因为审稿人会问：

> 提升到底来自 WGAN-GP、attention，还是 feature deviation？

---

## 4. 模型组件消融实验

**目的：** 分清楚 attention、WGAN-GP、TAD fusion 分别有什么贡献。

建议表格：

| Variant             | Pooling   | Loss        | Score  | F1 | Recall | Precision | Test FPR |
| ------------------- | --------- | ----------- | ------ | -: | -----: | --------: | -------: |
| TCN-GAN             | mean      | vanilla GAN | critic |    |        |           |          |
| TCN-GAN + attention | attention | vanilla GAN | critic |    |        |           |          |
| TCN-WGAN-GP         | mean      | WGAN-GP     | critic |    |        |           |          |
| TCN-WGAN-GP + TAD   | attention | WGAN-GP     | fusion |    |        |           |          |

如果你已经有一部分结果，就补齐 score 列和 Test FPR。

---

## 5. XAI 单窗口案例实验

**目的：** 回应“XAI 没有单独 case 不太行”。

需要选择一个典型 anomalous window，最好来自 CICIDS2017，比如：

* DDoS；
* PortScan；
* Web Attack；
* Bot。

对这个窗口输出：

1. 原始 anomaly score；
2. threshold；
3. attack type；
4. attention over time；
5. Integrated Gradients top features；
6. SHAP top features；
7. temporal attribution heatmap；
8. top features 的安全解释。

建议图名：

> Single-window alarm audit case study on CICIDS2017.

不要只放图，要写解释：

> 该窗口被判为异常主要是因为 packet rate、TCP flag、destination port、header length 等特征异常，这与 DDoS/PortScan 的流量行为一致。

---

## 6. XAI faithfulness 遮蔽实验

**目的：** 证明解释不是“画图”，而是和模型决策有关。

至少做一个：

### Feature masking

对 anomalous windows：

1. 找 IG 或 SHAP top-k features；
2. 把 top-k features 替换成 benign mean；
3. 重新计算 anomaly score；
4. 比较随机遮蔽 k 个 features。

表格：

| Masking strategy    | Score drop ↑ | Detection flip rate ↑ |
| ------------------- | -----------: | --------------------: |
| Random-k features   |              |                       |
| IG top-k features   |              |                       |
| SHAP top-k features |              |                       |

如果可以，再做一个：

### Time-step masking

| Masking strategy           | Score drop ↑ | Detection flip rate ↑ |
| -------------------------- | -----------: | --------------------: |
| Random-k time steps        |              |                       |
| Attention top-k time steps |              |                       |
| IG top-k time steps        |              |                       |

这个实验对 Computers & Security 很加分。

---

## 7. 鲁棒性实验加 baseline

**目的：** 回应老师说的“只测自己模型不行”。

你已有：

> CICIDS2017 抽样 12,000 个测试窗口，加入 1% / 3% / 5% Gaussian noise。

现在要补 baseline。

至少加入：

* MLP；
* RF；
* GANomaly；
* DeepSVDD；
* TranAD；
* IsolationForest。

表格：

| Model    | Clean F1 | Noise 1% F1 | Noise 3% F1 | Noise 5% F1 | Avg ΔF1 |
| -------- | -------: | ----------: | ----------: | ----------: | ------: |
| Proposed |          |             |             |             |         |
| MLP      |          |             |             |             |         |
| RF       |          |             |             |             |         |
| GANomaly |          |             |             |             |         |
| TranAD   |          |             |             |             |         |

另加 FPR 表：

| Model | Clean FPR | Noise 1% FPR | Noise 3% FPR | Noise 5% FPR |
| ----- | --------: | -----------: | -----------: | -----------: |

注意：

> threshold 必须固定为 clean benign calibration set 上得到的 threshold，不能每个 noise level 重新调阈值。

---

# 二、强烈建议补的实验

## 8. 攻击类型 breakdown

**目的：** 安全期刊很看重“哪些攻击能检出，哪些攻击检不出”。

在 CICIDS2017 上做：

| Attack type  | # windows | Recall | Precision/FPR | Median score | Top attributed features |
| ------------ | --------: | -----: | ------------: | -----------: | ----------------------- |
| DDoS         |           |        |               |              |                         |
| PortScan     |           |        |               |              |                         |
| Bot          |           |        |               |              |                         |
| Web Attack   |           |        |               |              |                         |
| Infiltration |           |        |               |              |                         |

这个表非常重要，能把你的论文从“机器学习跑分”变成“安全分析”。

---

## 9. α 敏感性实验

**目的：** 证明 α=0.24 不是随便选的。

建议扫：

> α ∈ {0, 0.1, 0.2, 0.24, 0.3, 0.5, 0.7, 1.0}

输出：

|  α | AUC | AP | F1 | Recall | Test FPR |
| -: | --: | -: | -: | -----: | -------: |

解释：

* α=0 表示 feature deviation only；
* α=1 表示 critic score only；
* 中间值表示融合；
* 选择 α=0.24 是因为 low-FPR F1 最优或 recall/FPR trade-off 最好。

---

## 10. 阈值/FPR sweep 实验

**目的：** 证明你的模型在不同误报预算下都比较稳定，而不是只在某个阈值好。

扫 target FPR：

> 0.01 / 0.03 / 0.05 / 0.10 / 0.15 / 0.20

输出：

| Target FPR | Observed Test FPR | Recall | Precision | F1 |
| ---------: | ----------------: | -----: | --------: | -: |
|       0.01 |                   |        |           |    |
|       0.03 |                   |        |           |    |
|       0.05 |                   |        |           |    |
|       0.10 |                   |        |           |    |
|       0.15 |                   |        |           |    |

最好和 2–3 个 baseline 对比。

这个能强化你的 low-false-positive 主线。

---

## 11. SWaT 采样粒度敏感性实验

**目的：** 回应你担心的 SWaT 5sec 只有 1433 windows、模型分数普遍偏高。

不要把 1sec 到 10sec 混合。分别跑：

* 1sec；
* 3sec；
* 5sec；
* 10sec。

表格：

| SWaT granularity | # windows | Proposed F1 | Best baseline F1 | Proposed FPR | Comment |
| ---------------- | --------: | ----------: | ---------------: | -----------: | ------- |
| 1 sec            |           |             |                  |              |         |
| 3 sec            |           |             |                  |              |         |
| 5 sec            |      1433 |             |                  |              |         |
| 10 sec           |           |             |                  |              |         |

如果时间不够，至少跑：

* Proposed；
* OneClassSVM；
* GANomaly；
* TranAD。

这个实验可以说明 SWaT 结果是否只依赖 5 秒聚合版本。

---

## 12. Baseline 超参数和训练协议表

**目的：** 提高公平性和可复现性。

这个不是实验，但必须补。

表格：

| Model           | Training data | Key hyperparameters         | Threshold calibration       |
| --------------- | ------------- | --------------------------- | --------------------------- |
| Proposed        | benign train  | W=128, stride=16, α=0.24... | benign calibration quantile |
| IsolationForest | benign train  | n_estimators...             | benign calibration quantile |
| OneClassSVM     | benign train  | kernel, nu...               | benign calibration quantile |
| MLP             | labeled train | hidden layers...            | validation/calibration      |
| GANomaly        | benign train  | latent dim, epochs...       | benign calibration quantile |
| TranAD          | benign train  | layers, hidden size...      | benign calibration quantile |

---

# 三、有时间再补的实验

## 13. Window size 和 stride 敏感性实验

**目的：** 证明 W=128、stride=16 合理。

扫：

* W ∈ {32, 64, 128, 256}
* stride ∈ {4, 8, 16, 32}

输出：

|  W | Stride | F1 | Recall | Test FPR | Throughput |
| -: | -----: | -: | -----: | -------: | ---------: |

如果你之前已有 window sweep，可以整理进论文。

---

## 14. 训练稳定性实验

**目的：** 证明 WGAN-GP 确实让训练更稳定。

可以比较：

* vanilla GAN；
* WGAN-GP。

指标：

| Loss        | Critic score variance | Gradient norm mean | F1 std over seeds | Test FPR std |
| ----------- | --------------------: | -----------------: | ----------------: | -----------: |
| vanilla GAN |                       |                    |                   |              |
| WGAN-GP     |                       |                    |                   |              |

这个实验可以支撑你论文里的 stability discussion。

---

## 15. 推理效率和资源开销实验

你已经有 throughput 表，可以加强：

| Dataset    | # windows | Seconds | Windows/s | Model size | CPU/GPU |
| ---------- | --------: | ------: | --------: | ---------: | ------- |
| CICIDS2017 |           |         |           |            |         |
| SWaT       |           |         |           |            |         |
| UNSW-NB15  |           |         |           |            |         |

最好补：

* parameter count；
* memory usage；
* per-window latency；
* rolling buffer online scoring说明。

---

# 四、建议删除或弱化的内容

## 1. TON IoT 主结果

如果效果很差，建议从主文结果里删掉。

可以在 Discussion 里一句话写：

> Additional telemetry-domain experiments indicate that direct transfer to heterogeneous IoT telemetry remains challenging.

不要在摘要和贡献点里提 TON IoT。

---

## 2. 没有 baseline 的鲁棒性表

如果来不及补 baseline，就不要把 Gaussian noise robustness 当主结果。否则审稿人会认为证据不足。

---

## 3. 太多不公平的现代 baseline

DLinear、Autoformer、ALoRa 如果实现和调参不充分，可以放附录或删除。主文不要堆太多容易被质疑的 baseline。

---

# 五、最终实验优先级清单

如果你时间有限，按这个顺序做：

| 优先级 | 实验                                    | 是否必须  |
| --- | ------------------------------------- | ----- |
| 1   | 数据划分/window 数量统计                      | 必须    |
| 2   | TAD Score score ablation              | 必须    |
| 3   | XAI 单窗口 case study                    | 必须    |
| 4   | XAI top-k masking faithfulness        | 必须    |
| 5   | Gaussian noise robustness + baseline  | 必须    |
| 6   | 主结果多 seed                             | 强烈建议  |
| 7   | 攻击类型 breakdown                        | 强烈建议  |
| 8   | α sensitivity                         | 强烈建议  |
| 9   | FPR sweep                             | 强烈建议  |
| 10  | SWaT 1sec/3sec/5sec/10sec sensitivity | 建议    |
| 11  | window size / stride sensitivity      | 有时间再补 |
| 12  | WGAN-GP training stability            | 有时间再补 |

---

# 六、给 Codex 的任务可以这样拆

你不要一次性让 Codex 做全部，建议分 5 个任务发。

## 任务 1：数据统计与 split table

让 Codex 统计每个数据集：

* 原始记录数；
* train/calib/test windows；
* benign/anomaly windows；
* attack type 分布；
* window label 规则。

## 任务 2：score ablation + alpha sweep

让 Codex 跑：

* critic only；
* feature deviation only；
* TAD fusion；
* α sweep。

## 任务 3：XAI case + faithfulness

让 Codex 做：

* 选一个典型 anomalous window；
* 输出 attention / IG / SHAP；
* 做 top-k feature masking 和 random masking。

## 任务 4：robustness with baselines

让 Codex 在 12,000 CICIDS2017 windows 上跑：

* Proposed；
* MLP；
* RF；
* GANomaly；
* TranAD；
* DeepSVDD；
* clean / 1% / 3% / 5% Gaussian noise。

## 任务 5：多 seed 和 attack breakdown

让 Codex 跑：

* 5 seeds；
* attack-type recall；
* per-attack score distribution；
* top features by attack type。

---

一句话总结：
**现在最必须补的是：数据划分、防泄漏、TAD 消融、XAI case + faithfulness、鲁棒性 baseline、多 seed 和攻击类型分析。把这些补上，你这篇才比较像能投 Computers & Security 的期刊稿。**
