# `attack/pipelines/`

这里放当前论文主线的自动化实验脚本。

一句话区分：

```text
models    负责“训练/评估模型本身”
pipelines 负责“按论文实验流程反复调用 models”
```

## 当前 bash 入口

这些是你平时直接运行的实验入口。

### `run_final_experiments.sh`

主线 TCN 实验入口。

流程：

```text
stride sweep -> alpha sweep -> threshold trade-off -> 2x2 ablation
```

运行：

```bash
bash attack/pipelines/run_final_experiments.sh
```

### `run_baseline_fpr_sweep.sh`

当前推荐 baseline 入口。

每个 baseline 训练一次，然后在多个 target-FPR 下重新标定阈值并评估。

运行：

```bash
bash attack/pipelines/run_baseline_fpr_sweep.sh
```

### `run_tcn_fpr_sweep.sh`

不重新训练，加载最终 Attentive TCN-WGAN-GP checkpoint，扫描多个 target-FPR。

### `run_ablation_fpr_sweep.sh`

不重新训练，加载 2x2 消融 checkpoint，扫描多个 target-FPR。

### `run_tcn_seed_runs.sh`

最终模型多 seed 补跑入口。

### `run_baseline.sh`

单工作点 baseline 重跑脚本。现在一般不作为论文主表入口，优先用 `run_baseline_fpr_sweep.sh`。

## 当前 Python pipeline

### `run_tcn_gan_autotune.py`

用途：扫描 window 或 stride。

当前主线里主要用于：

```text
固定 window=128 后扫描 stride
补训缺失 stride
复用已有 checkpoint
汇总 tcn_gan_autotune_results.csv
```

会调用：

```text
attack/models/tcn_gan_experiment.py
```

### `run_tcn_gan_alpha_sweep.py`

用途：固定 checkpoint，不重新训练，只扫描 fused score 的 `alpha`。

当前主线选择：

```text
score_alpha = 0.24
```

输出位置：

```text
attack/results/final_experiments/20260414_132416/alpha_sweep/
```

### `run_tcn_gan_ablation_2x2.py`

用途：固定最终参数后做 2x2 消融。

当前比较：

```text
disc_pooling: mean vs attn
gan_loss:     vanilla vs wgan-gp
```

输出位置：

```text
attack/results/final_experiments/20260414_132416/ablation_2x2/20260414_204232/
```

### `make_tcn_gan_ablation_table.py`

用途：从当前消融目录下的 `eval_*.json` 汇总表格。

输出：

```text
ablation.csv
ablation_summary.csv
ablation.md
ablation_summary.md
```

注意：

这个脚本已经修正过，指定 `--glob` 时只读取当前路径，不再混入旧实验结果。

## 当前辅助 pipeline

### `make_paper_ready_assets.py`

用途：把 baseline FPR sweep、最终 TCN FPR sweep、seed、消融和 XAI 结果整理到 `attack/paper/`。

输出：

```text
attack/paper/figures/
attack/paper/tables/
attack/paper/asset_manifest.json
attack/paper/ASSETS.md
```

当前论文主表和汇报图优先使用这里生成的文件。

### `plot_tcn_gan_autotune_viz.py`

用途：对 sweep 目录做后处理可视化。

它不训练模型，只读取已有 `csv/json/xai_report`，用于生成 sweep 图或对齐后的 XAI 时间曲线。

## 推荐入口

日常不要一个个手动调用 pipeline，优先使用项目根入口：

```bash
bash attack/pipelines/run_final_experiments.sh
```

这个入口的主线顺序是：

```text
1. stride sweep
2. alpha sweep
3. threshold trade-off
4. 2x2 ablation
```

baseline 对比使用：

```bash
bash attack/pipelines/run_baseline_fpr_sweep.sh
```

## 已归档的旧 pipeline

旧 flow-level paper pipeline、finishline pipeline、旧 paper table/figure 脚本已经移动到：

```text
attack/_archive/cleanup_20260415/legacy_code/pipelines/
```

当前论文主线不要再从这些旧脚本取结果。
