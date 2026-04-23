# 当前模型结果风险与写法修正

更新时间：2026-04-24

这份文档不再按旧版本“attention 改造后指标下降”来简单下结论，而是结合当前 `paper/main.tex` 的完整结果来说明：

1. 现在真正的论文风险是什么；
2. 哪些说法能稳；
3. 哪些地方仍然要小心；
4. 当前最适合的写法路线是什么。

## 1. 当前最大的真实风险

现在的风险已经不是“只有 CICIDS2017 一个数据集”了，因为论文已经补入：

- `SWaT`
- `UNSW-NB15`
- `TON_IoT`
- 多个现代 baseline：`GANomaly`、`DeepSVDD`、`TranAD`、`Anomaly Transformer`、`TimesNet`、`DLinear`、`Autoformer`、`ALoRa`

当前真正的风险主要有四个：

### 风险 1：把论文写成“全面领先”

当前结果不支持这种写法。

- `CICIDS2017`：主模型强，尤其在低误报工作点。
- `SWaT`：主模型也很强，但不是唯一强的方法。
- `UNSW-NB15`：有竞争力，但不是第一。
- `TON_IoT`：明显是局限。

所以不能写：

```text
Our method consistently outperforms all baselines across all datasets.
```

### 风险 2：把 attention 写成“单独带来全面提升”

当前消融不支持这个说法。

`target FPR=0.15` 下：

- `attn + WGAN-GP`: F1 `0.8783`, Recall `0.9678`
- `mean + vanilla`: F1 `0.8469`, Recall `0.8940`
- `attn + vanilla`: F1 `0.6891`, Recall `0.7099`

这说明：

> attention 不是“单独加上去就一定更好”的万能模块，而是要和 WGAN-GP 及当前评分设计配合起来才有价值。

### 风险 3：外部 baseline 口径混合

当前最典型的是：

- `ALoRa` 还是 `W=20`
- 其他大多数现代模型是 `W=128`
- `UNSW` 某些 external 行带 `dagger`，因为可用 fair run 结果来自 `target FPR=0.05`

这不一定致命，但必须诚实说明，不能装成“完全相同设置下公平比较”。

### 风险 4：把 XAI 写成训练贡献

当前论文里的 XAI 是 post-hoc audit，不是训练阶段提升性能的模块。

所以不能写：

```text
XAI improves detection performance.
```

更稳的说法是：

> XAI 帮助分析员审计告警依据，提高结果可读性与可检查性。

## 2. 当前结果最稳的解释方式

### 2.1 CICIDS2017

`target FPR=0.05`：

- Ours: F1 `0.7958`, Recall `0.6640`, Test FPR `0.0036`
- MLP: F1 `0.7438`, Recall `0.6900`, Test FPR `0.1210`

最稳结论：

> 在严格低误报工作点，本文模型取得最高 F1，并把 observed benign test FPR 压到很低。这比“谁 recall 略高一点”更符合实际安全监控关注点。

`target FPR=0.15`：

- Ours: F1 `0.8783`, Recall `0.9678`

最稳结论：

> 当部署允许更高误报预算时，模型能把 recall 拉到接近 0.97，同时维持最高 F1。

### 2.2 SWaT

- Ours: F1 `0.9950`, Test FPR `0.0000`
- TimesNet / OneClassSVM / DLinear / Autoformer 也很强

最稳结论：

> SWaT 说明本文方法具备很强的跨域迁移能力，但这一数据集对多种时序异常检测方法都相对友好，因此不宜夸写成“碾压式优势”。

### 2.3 UNSW-NB15

- Ours: F1 `0.7928`, Recall `0.9037`, Test FPR `0.4660`

最稳结论：

> 本文方法在 UNSW 上保持竞争力，但并不主导全部 baseline。这里更重要的结论是：阈值跨域迁移困难，不同模型常常通过更高 benign false alarms 来换取高 recall。

### 2.4 TON_IoT

- Ours: AUC `0.5556`, F1 `0.1794`

最稳结论：

> TON_IoT 明确界定了方法当前的域外边界。把它写成 limitation，比硬找优势更可信。

## 3. attention 和 WGAN-GP 现在该怎么写

当前最稳的表述不是：

```text
attention significantly improves detection performance
```

而是：

```text
attention pooling, when paired with WGAN-GP and fused scoring, supports a stronger operating-point trade-off and better alarm auditability
```

中文理解：

> attention pooling 与 WGAN-GP 和 fused scoring 配合后，使模型在实际工作点上的 trade-off 更好，也更利于告警审计；它不是一个“所有指标都会自动提升”的独立增强模块。

## 4. 当前最推荐的论文叙事路线

### 路线 A：当前主文已经采用，也最稳

主张写成：

1. `Attentive TCN-WGAN-GP` 是一个低误报、窗口级、可解释的检测框架。
2. CICIDS2017 是主战场，主要证明低误报工作点的优势。
3. SWaT 是强验证，说明方法能迁移到工控场景。
4. UNSW 是压力测试，显示阈值迁移难题。
5. TON_IoT 是域外局限。
6. XAI 用于告警审计，不包装成提升分数的原因。

这是当前最适合投稿和答辩的路线。

### 路线 B：继续追求统一口径再收口

如果还要进一步补强，当前最有价值的是：

1. 完成 `ALoRa@W128`；
2. 决定是否删除表中的 `W` 列；
3. 统一 external rows 的 target FPR 解释；
4. 再做最终文字收束。

这比继续堆更多 baseline 更值。

## 5. 当前不建议再说的话

不建议：

- `attention 显著提升了所有检测指标`
- `模型在所有数据集上都优于所有 baseline`
- `XAI 提高了模型性能`
- `跨数据集结果证明了通用性`

建议改成：

- `模型在目标低误报场景下表现最强`
- `跨数据集结果展示了强验证、压力测试和域外局限`
- `XAI 用于审计告警依据`
- `阈值迁移仍是开放问题`

## 6. 当前可以直接放进论文或答辩的稳妥说法

英文：

> The ablation results do not support the claim that attention pooling alone universally improves all detection metrics. Instead, the attentive WGAN-GP variant should be interpreted as a low-false-alarm and auditable detector whose advantage becomes most visible at deployment-relevant operating points.

中文：

> 当前结果并不支持“注意力机制单独加入后会全面提升所有检测指标”这一说法。更合适的理解是：attentive WGAN-GP 变体更适合作为一个低误报、可审计的检测器，它的优势主要体现在部署相关的实际工作点上。

英文：

> Across datasets, the method shows three different behaviors: a primary win on CICIDS2017, strong transfer on SWaT, competitive but non-dominant behavior on UNSW-NB15, and an explicit limitation on TON_IoT.

中文：

> 跨数据集结果体现出三种不同角色：CICIDS2017 是主胜利点，SWaT 是强迁移验证，UNSW-NB15 是有竞争力但不绝对领先的压力测试，TON_IoT 则是明确写出的域外局限。

## 7. 一句话总结

当前论文最应该避免的不是“结果不够多”，而是“把当前结果说过头”。只要把主张收在低误报窗口级检测、跨域压力和告警审计这条线上，当前版本是能自洽且有说服力的。
