# `attack/docs/` 文档导航（2026-04-24）

这份索引用于区分：

- 哪些文档是**当前事实来源**（可直接用于写论文）
- 哪些文档是**当前版说明文档**（可直接用于理解当前状态，但不等同于最终论文正文）
- 哪些文档是**阶段性/历史参考**（可读，但不要直接当最终结论）

## 一、当前事实来源（优先）

1. `CURRENT_PAPER_GUIDE_CN.md`  
用途：论文当前主线、当前风险、当前待办的总入口。

2. `RECENT_EXTERNAL_BASELINE_RUNS_CN.md`  
用途：外部 baseline（ALoRa / Anomaly-Transformer / Time-Series-Library）跑数状态和结果口径。

3. `TCN_GAN_THESIS_EXPERIMENT_RECORD.md`  
用途：主模型实验冻结配置与核心结果路径。

4. `EXPERIMENT_DESIGN_GUIDE.md`  
用途：审稿人实验建议的最新完成状态，区分已完成、部分完成和不建议继续追的实验。

5. `../paper/main.tex` + `../paper/references.bib`  
用途：最终论文内容与引用真相。

## 二、当前版说明文档（次优先）

- `A_TIER_REVISION_PLAN_CN.md`
- `RESULTS_WRITEUP_PLAN_CN.md`
- `REPORT_AND_OVERLEAF_GUIDE_CN.md`

这些文档已经同步到当前状态，适合用来理解“论文现在怎么讲、下一步怎么收尾”。如果和主文冲突，仍然以 `paper/main.tex` 和最新结果文件为准。

## 三、阶段性说明文档（可能过期）

- `PAPER_READING_GUIDE_CN.md`
- `PRESENTATION_CLOSING_PLAN_CN.md`
- `FIGURE_REVISION_NOTES_CN.md`
- `MODEL_RESULT_RISK_AND_FIX_CN.md`

这几份写作时点较早，仍有参考价值，但阅读前请先以 `CURRENT_PAPER_GUIDE_CN.md` 对齐当前状态。

## 四、使用建议

- 写论文时：先看 `CURRENT_PAPER_GUIDE_CN.md`，再对照 `paper/main.tex`。
- 查外部 baseline：看 `RECENT_EXTERNAL_BASELINE_RUNS_CN.md`，不要只看旧汇报稿。
- 查当前结果目录结构：优先看 `../results/README.md` 和 `../results/RESULTS_GUIDE_CN.md`。
- 若文档之间冲突：以 `paper/main.tex` 和最新结果文件为准。
