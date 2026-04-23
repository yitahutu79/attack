# `attack/paper/`

这里放当前论文初稿和 Overleaf 需要的文件。

## 当前论文文件

- `main.tex`
  - 当前最新版英文论文初稿。
  - 模型名已经统一为 `Attentive TCN-WGAN-GP`。
  - 已包含 Introduction、Related Work、Method、Experiments、Results、XAI、Discussion、Conclusion。

- `references.bib`
  - BibTeX 参考文献库。
  - Overleaf 编译时需要和 `main.tex` 放在同一项目里。

- `attack.pdf`
  - 当前编译后的论文 PDF。
  - 如果这是从 Overleaf 下载/同步回来的版本，就可以作为当前可阅读 PDF。
  - 后续如果修改 `main.tex`，需要重新在 Overleaf 编译并更新这个 PDF。

## 图片目录

- `figures/`
  - `main.tex` 直接引用的图片。
  - 上传 Overleaf 时需要保持目录名为 `figures`，否则图片路径要同步修改。

当前图片统一保存在 `figures/`，不再在 `results/` 下保留重复副本。

## 表格目录

- `tables/`
  - 论文和汇报优先使用的整理后表格。
  - 包含 baseline 主表、FPR sweep、消融表和多 seed 表。

## 资产清单

- `asset_manifest.json`
  - 图表来源文件和推荐图表列表。

- `ASSETS.md`
  - 由 `attack/pipelines/make_paper_ready_assets.py` 自动生成的图表清单。

## 放到 Overleaf 要复制什么

最小需要复制这些内容：

```text
attack/paper/main.tex
attack/paper/references.bib
attack/paper/figures/*.png
```

`tables/` 一般不用上传到 Overleaf，除非你想在 Overleaf 里手动查表或后续改成从 CSV 自动生成 LaTeX 表。

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
  xai_panel.png
  threshold_tradeoff.png
  xai_feature_importance.png
  xai_time_importance.png
```

## 汇报优先看

如果只是做组会汇报，不需要直接从这里找原始结果，优先看：

```text
attack/paper/
```

这里已经把论文/汇报用 tex、bib、图和表集中好了。
