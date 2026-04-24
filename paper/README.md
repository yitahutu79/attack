# `attack/paper/`

这里放当前论文正文、引用、图表和 Overleaf 需要的整理后资产。

## 当前论文文件

- `main.tex`
  - 当前最新版英文论文正文。
  - 当前论文结论、表格口径和叙事主线都应以这里为准。

- `references.bib`
  - BibTeX 参考文献库。

- `attack.pdf`
  - 当前可阅读 PDF。
  - 若 `main.tex` 更新后需要重新编译，应同步更新该文件。

## 图片目录

- `figures/`
  - `main.tex` 直接引用的图片。
  - 上传 Overleaf 时需要保持目录名为 `figures`，否则图片路径要同步修改。

当前图片统一保存在 `figures/`，不再在 `results/` 下保留重复副本。

## 表格目录

- `tables/`
  - 当前论文和汇报优先使用的整理后表格。
  - 这里的 CSV/MD 是从 `results/` 主线目录汇总后的可直接引用版本。

## 资产清单

- `asset_manifest.json`
  - 当前推荐图表及其来源文件清单。

- `ASSETS.md`
  - 当前推荐图表与来源摘要。

## 放到 Overleaf 要复制什么

最小需要复制这些内容：

```text
attack/paper/main.tex
attack/paper/references.bib
attack/paper/figures/*.png
```

`tables/` 一般不用上传到 Overleaf，除非你要在 Overleaf 里手动查表。

在 Overleaf 里建议结构：

```text
main.tex
references.bib
figures/
  alpha_sweep.png
  attn_weights.png
  fpr_sweep_f1_recall.png
  seed_f1_bar.png
  framework_overview.png
  model_architecture.png
  xai_case_multiview_shap.png
  threshold_tradeoff.png
  xai_feature_importance.png
  xai_time_importance.png
```

## 当前读取顺序

推荐按下面的顺序看：

```text
attack/paper/
attack/results/
attack/_archive/
```

其中：

- `paper/` 是论文和汇报的直接入口
- `results/` 用于追溯当前主线原始结果
- `_archive/` 只用于查历史版本
