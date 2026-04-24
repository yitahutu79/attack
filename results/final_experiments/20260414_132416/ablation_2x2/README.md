# `ablation_2x2/`

这里保存最终配置下的 2x2 消融实验。

## 当前有效结果

- `20260414_204232/`
  - 当前论文使用的消融实验目录。

消融维度：

```text
disc_pooling: mean vs attn
gan_loss:     vanilla vs wgan-gp
```

所有消融实验固定：

```text
window = 128
stride = 16
score_mode = fused
score_alpha = 0.24
```
