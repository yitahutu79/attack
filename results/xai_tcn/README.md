## `attack/results/xai_tcn/`

TCN‑GAN 的 XAI 输出目录。当前只保留最终主模型 `window=128, stride=16, alpha=0.24` 的解释结果：

```text
eval_w128_s16_fused_w128_s16_fused_a0.24/
```

常见文件：
- `xai_time_importance*.png`：时间位置的重要性
- `xai_feature_importance*.png`：特征重要性
- `xai_report.json`：数值报告
- `attn_weights.png`：attention 权重可视化

这些一般由：
- `attack/models/tcn_gan_experiment.py --xai-report`
- `attack/pipelines/run_tcn_gan_autotune.py --xai`
触发生成。

旧 `alpha=0.6` 和非最终窗口的 XAI 图已经移动到：

```text
attack/_archive/cleanup_20260415/legacy_results/xai_tcn/
```

如果要看单窗口多视角个案图，请看：

```text
attack/results/xai_case_study_shap/
```
