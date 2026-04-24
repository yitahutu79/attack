# `eval_w128_s16_fused_w128_s16_fused_a0.24/`

最终模型的 XAI 输出目录。

## 对应配置

```text
window = 128
stride = 16
score_mode = fused
score_alpha = 0.24
disc_pooling = attn
gan_loss = wgan-gp
```

## 文件说明

- `xai_report.json`
  - XAI 数值报告。

- `xai_feature_importance.png`
  - 特征重要性图。
  - 说明模型异常判断主要依赖哪些网络流量特征。

- `xai_time_importance.png`
  - 时间位置重要性图。
  - 说明窗口内哪些位置对异常分数贡献更大。

- `attn_weights.png`
  - attention pooling 学到的时间权重。
  - 用于辅助解释模型如何聚合窗口内时间证据。

这些图已经复制到：

```text
attack/paper/figures/
attack/paper/figures/
```
