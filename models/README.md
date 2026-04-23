# `attack/models/`

这里放当前论文主线的模型训练、评估和可视化代码。

## 当前保留文件

### `tcn_gan_experiment.py`

这是当前项目最核心的模型脚本。

它负责：

```text
1. 读取 CICIDS2017 CSV
2. 按 window_size / stride 构造窗口级样本
3. 训练 Attentive TCN-WGAN-GP / TCN-GAN 消融变体
4. 支持 mean pooling 和 attention pooling 判别器
5. 支持 vanilla GAN loss 和 WGAN-GP
6. 计算 prob / feat_l2 / feat_mahal / fused anomaly score
7. 使用 target_fpr 做可部署阈值标定
8. 输出 eval JSON / scores CSV / checkpoint
9. 生成 XAI 报告和图像
```

当前最终主模型配置：

```text
window_size  = 128
stride       = 16
score_alpha  = 0.24
disc_pooling = attn
gan_loss     = wgan-gp
score_mode   = fused
target_fpr   = 0.05
```

最终配置文件：

```text
attack/results/final_experiments/20260414_132416/selected_main_config.json
```

论文中的“本文方法”“主模型”“最终模型”都应该优先指向这个脚本和这份冻结配置。论文正式命名建议使用 `Attentive TCN-WGAN-GP`；`TCN-GAN` 可以作为基础模型或脚本历史名称出现。

### `tcn_gan_visualization.py`

用途：生成 Attentive TCN-WGAN-GP / TCN-GAN 结构示意图或讲解用图。

它不参与训练和评估，只服务论文插图、汇报图或方法框架说明。

## 已归档的旧模型代码

以下文件已经移到：

```text
attack/_archive/cleanup_20260415/legacy_code/models/
```

包括：

```text
tcn_classifier_multiclass.py
tcn_early_warning_multiclass.py
```

归档原因：

- `tcn_classifier_multiclass.py` 是监督式 TCN 多分类支线，不是当前无监督/半监督异常检测主线。
- `tcn_early_warning_multiclass.py` 是早期预警多分类实验支线，和当前最终 Attentive TCN-WGAN-GP 论文叙事不一致。

这些文件没有删除，只是不再放在当前主模型目录里，避免和最终论文主线混淆。

## 当前建议

训练、评估、XAI 都从下面两个入口间接调用本目录代码：

```bash
bash attack/pipelines/run_final_experiments.sh
bash attack/pipelines/run_baseline.sh
```

如果单独调试主模型，再直接运行：

```bash
/Users/lijie/miniforge3/envs/attack/bin/python attack/models/tcn_gan_experiment.py --help
```
