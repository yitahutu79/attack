# `attack/baselines/`

这里现在只保留当前论文主线真正使用的完整窗口级 baseline。

## 当前保留文件

### `window_baselines.py`

用途：完整窗口级 baseline 对比。

这是当前窗口级 baseline 结果的主要来源，包含：

```text
IsolationForest
OneClassSVM
RF (Window)
MLP (Window)
LSTM-AE
LSTM-AD
```

当前入口：

```bash
bash attack/pipelines/run_baseline.sh
```

如果要扫描多个误报预算：

```bash
bash attack/pipelines/run_baseline_fpr_sweep.sh
```

论文用途：

> 在相同窗口级数据划分下，对比传统无监督方法、监督窗口分类器和深度序列异常检测方法。

当前默认设置：

```text
window_size = 128
stride = 16
anomaly_ratio = 0.15
target_fpr = 0.05
methods = iforest ocsvm rf mlp lstm_ae lstm_ad
```

正式 FPR sweep 输出：

```text
attack/results/baseline_fpr_sweep/20260415_020742/
```

常见输出：

```text
baseline_results.json
baseline_results.csv
baseline.log
```

## 已归档的旧 baseline 代码

以下支线已经移到：

```text
attack/_archive/cleanup_20260415/legacy_code/baselines/
```

包括：

```text
compare_models.py
xai_anomaly_score.py
supervised_multiclass.py
gan_tabular_augment.py
xai_openset.py
```

归档原因：

- `compare_models.py` 自己又实现了一套轻量 baseline，对当前正式 baseline 来说重复且容易混淆。
- `xai_anomaly_score.py` 只服务 `compare_models.py` 的旧可选分数融合。
- `supervised_multiclass.py` 是旧 flow-level / 多分类支线，和当前窗口级 TCN-GAN 主线不是同一实验口径。
- `gan_tabular_augment.py` 只服务旧 supervised baseline 的 GAN 数据增强。
- `xai_openset.py` 属于旧 open-set 解释性支线，当前论文主线不再直接使用。

## 后续如果要做 FPR sweep

现在已经在 `window_baselines.py` 内部支持 `--target-fpr-grid`，同一批已训练模型会在多个阈值下评估。这样：

```text
模型来源一致
数据划分一致
不会重复维护两套 baseline
论文口径也更干净
```
