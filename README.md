# Attentive TCN-WGAN-GP 网络入侵检测实验目录

这个目录现在围绕论文主线来整理：

> Attentive TCN-WGAN-GP 是一个适合窗口级实时网络异常检测的模型，并且支持 attention 与 XAI 解释。

当前主线配置已经冻结为：

- `window_size = 128`
- `stride = 16`
- 主表工作点：`target_fpr = 0.05`
- 高召回工作点：`target_fpr = 0.15`
- `anomaly_ratio = 0.15`
- `disc_pooling = attn`
- `gan_loss = wgan-gp`
- `score_mode = fused`
- `score_alpha = 0.24`

最终配置文件：

```text
results/final_experiments/20260414_132416/selected_main_config.json
```

## 主入口

- `pipelines/run_final_experiments.sh`
  - 当前论文主线脚本。
  - 流程是：固定 window 后扫 stride，再扫 alpha，冻结配置，最后跑 2x2 消融。

- `pipelines/run_baseline_fpr_sweep.sh`
  - 完整窗口级 baseline 的目标误报率扫描脚本。
  - 用于观察不同 `target_fpr` 下各 baseline 的 recall / precision / F1 变化。

- `pipelines/run_tcn_fpr_sweep.sh`
  - 最终 Attentive TCN-WGAN-GP checkpoint 的目标误报率扫描脚本。
  - 不重新训练，只在多个 `target_fpr` 下重新评估阈值工作点。

- `pipelines/run_ablation_fpr_sweep.sh`
  - 四个 2x2 消融 checkpoint 的目标误报率扫描脚本。
  - 不重新训练，用于比较不同误报预算下的消融结论。

- `pipelines/run_tcn_seed_runs.sh`
  - 最终 Attentive TCN-WGAN-GP 配置的额外 seed 重跑脚本。
  - 默认补跑 seed=43,44。

- `pipelines/run_baseline.sh`
  - 单工作点 baseline 脚本。
  - 现在一般不作为论文主表入口，优先用 `pipelines/run_baseline_fpr_sweep.sh`。

## 核心代码

- `models/tcn_gan_experiment.py`
  - Attentive TCN-WGAN-GP 训练/评估脚本。
  - 支持 attention pooling、WGAN-GP、fused score、target-FPR 标定和 XAI 报告。

- `pipelines/run_tcn_gan_autotune.py`
  - window/stride sweep。

- `pipelines/run_tcn_gan_alpha_sweep.py`
  - 不重训的 alpha sweep，使用固定 checkpoint。

- `pipelines/run_tcn_gan_ablation_2x2.py`
  - 2x2 消融：`mean/attn x vanilla/wgan-gp`。

- `pipelines/make_tcn_gan_ablation_table.py`
  - 从 eval JSON 生成消融表。

- `baselines/window_baselines.py`
  - 完整窗口级 baseline，对应当前 `results/baseline_fpr_sweep/20260415_020742/` 结果。

## 重点结果目录

- `results/final_experiments/`
  - 当前论文主线输出。现在以 `20260414_132416/` 为准。

- `paper/`
  - 汇报和论文写作优先入口。
  - 已集中主对比表、FPR sweep 图、多 seed 图表、消融表和 XAI 图。

- `results/xai_tcn/`
  - XAI 报告和图片。当前保留最终模型 `alpha=0.24` 的解释结果。

## 文档

- `docs/CURRENT_PAPER_GUIDE_CN.md`
  - 当前论文主线、数据集角色、主要风险和阅读入口。

- `docs/README.md`
  - `docs/` 文档导航，区分当前事实来源、当前版说明文档和历史参考。

- `docs/TCN_GAN_THESIS_EXPERIMENT_RECORD.md`
  - 当前论文主线实验记录和重跑说明。

- `docs/A_TIER_REVISION_PLAN_CN.md`
  - 当前版的高标准投稿补强评估，重点看哪些已经完成、哪些仍是风险。

- `docs/RESULTS_WRITEUP_PLAN_CN.md`
  - 当前版的 Results 写作说明，帮助统一主表、补充表和正文叙事。

- `results/RESULTS_GUIDE_CN.md`
  - 解释 baseline、最终实验和 XAI 结果文件。

- `paper/README.md`
  - 说明 Overleaf 需要复制哪些 tex、bib 和图片文件。

- `docs/EXPERIMENT_DESIGN_GUIDE.md`
  - 当前实验设计状态说明，区分已完成、部分完成和不建议继续追的项。

## 归档

旧脚本、旧 flow-level 论文对齐结果、重复图片、失败/半成品运行和系统文件已经移动到：

- `_archive/cleanup_20260413/`
- `_archive/cleanup_20260415/`

这些文件只是移动，没有删除。
