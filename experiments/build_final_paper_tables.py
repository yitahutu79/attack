#!/usr/bin/env python3
"""Build final paper tables from strict CICIDS result artifacts (no retraining)."""

from __future__ import annotations

import csv
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def f4(x: float) -> str:
    return f"{float(x):.4f}"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def md_table(rows: list[dict], cols: list[str], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        vals = [str(r.get(c, "")) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def build_table1(out_dir: Path) -> tuple[Path, Path, list[dict]]:
    src = ROOT / "results/current_paper_compact_baselines/cicids2017_compact_baselines.csv"
    df = pd.read_csv(src)

    want = ["Ours", "MLP (Window)", "GANomaly", "TranAD", "IsolationForest", "OneClassSVM"]
    display = {
        "Ours": "Ours",
        "MLP (Window)": "MLP",
        "GANomaly": "GANomaly",
        "TranAD": "TranAD",
        "IsolationForest": "IsolationForest",
        "OneClassSVM": "OneClassSVM",
    }

    sub = df[df["method"].isin(want)].copy()
    sub = sub.sort_values(by="method", key=lambda s: s.map({m: i for i, m in enumerate(want)}))

    rows: list[dict] = []
    for _, r in sub.iterrows():
        rows.append(
            {
                "method": display[str(r["method"])],
                "AUC": f4(r["auc"]),
                "AP": f4(r["ap"]),
                "observed_test_benign_fpr": f4(r["observed_test_benign_fpr"]),
                "precision": f4(r["precision"]),
                "recall": f4(r["recall"]),
                "F1": f4(r["f1"]),
            }
        )

    csv_path = out_dir / "table1_cicids_compact_baselines.csv"
    md_path = out_dir / "table1_cicids_compact_baselines.md"
    fields = ["method", "AUC", "AP", "observed_test_benign_fpr", "precision", "recall", "F1"]
    write_csv(csv_path, rows, fields)
    md_text = "# 表1：CICIDS2017 严格协议下代表性 Baseline 对比\n\n"
    md_text += "说明：所有指标来自 strict no-leakage 结果，数值保留 4 位小数。\n\n"
    md_text += md_table(rows, fields, ["方法", "AUC", "AP", "测试良性误报率", "Precision", "Recall", "F1"])
    md_path.write_text(md_text, encoding="utf-8")
    return csv_path, md_path, rows


def build_table2(out_dir: Path) -> tuple[Path, Path, list[dict]]:
    src = ROOT / "results/cicids_strict_mps/posthoc_threshold_sweep.csv"
    df = pd.read_csv(src)
    tfprs = [0.15, 0.20, 0.25, 0.30]
    sub = df[df["target_fpr"].round(2).isin(tfprs)].copy()
    sub = sub.sort_values("target_fpr")

    rows: list[dict] = []
    for _, r in sub.iterrows():
        tfpr = float(r["target_fpr"])
        rows.append(
            {
                "target_fpr": f4(tfpr),
                "observed_test_benign_fpr": f4(r["observed_test_benign_fpr"]),
                "precision": f4(r["precision"]),
                "recall": f4(r["recall"]),
                "F1": f4(r["F1"]),
                "selected_operating_point": "Yes" if abs(tfpr - 0.25) < 1e-9 else "No",
            }
        )

    csv_path = out_dir / "table2_threshold_sweep.csv"
    md_path = out_dir / "table2_threshold_sweep.md"
    fields = ["target_fpr", "observed_test_benign_fpr", "precision", "recall", "F1", "selected_operating_point"]
    write_csv(csv_path, rows, fields)
    md_rows = [dict(r) for r in rows]
    for r in md_rows:
        if r["selected_operating_point"] == "Yes":
            r["target_fpr"] = f"{r['target_fpr']} ★"
    md_text = "# 表2：阈值扫描与工作点选择\n\n"
    md_text += "说明：选定工作点为 target_fpr=0.2500（★）。\n\n"
    md_text += md_table(md_rows, fields, ["target_fpr", "测试良性误报率", "Precision", "Recall", "F1", "是否选定"]) 
    md_path.write_text(md_text, encoding="utf-8")
    return csv_path, md_path, rows


def build_table3(out_dir: Path) -> tuple[Path, Path, list[dict]]:
    src = ROOT / "results/cicids_strict_mps/posthoc_threshold_sweep_attack_type_breakdown.csv"
    df = pd.read_csv(src)
    sub = df[(df["target_fpr"].round(2) == 0.25) & (df["attack_type"].isin(["Bot", "DDoS", "PortScan"]))].copy()

    order = {"Bot": 0, "DDoS": 1, "PortScan": 2}
    sub["_o"] = sub["attack_type"].map(order)
    sub = sub.sort_values("_o")

    rows: list[dict] = []
    for _, r in sub.iterrows():
        atk = str(r["attack_type"])
        rows.append(
            {
                "attack_type": atk,
                "total": int(r["total_attack_windows"]),
                "detected": int(r["detected_attack_windows"]),
                "recall": f4(r["recall"]),
                "mean_score": f4(r["mean_score"]),
                "note": "局限性：Bot 召回偏低" if atk == "Bot" else "",
            }
        )

    csv_path = out_dir / "table3_attack_type_breakdown.csv"
    md_path = out_dir / "table3_attack_type_breakdown.md"
    fields = ["attack_type", "total", "detected", "recall", "mean_score", "note"]
    write_csv(csv_path, rows, fields)
    md_text = "# 表3：攻击类型分解（selected operating point, target_fpr=0.2500）\n\n"
    md_text += "说明：Bot 在当前设置下召回率显著偏低，是当前方法局限性。\n\n"
    md_text += md_table(rows, fields, ["攻击类型", "总窗口", "检出窗口", "召回率", "均值分数", "备注"])
    md_path.write_text(md_text, encoding="utf-8")
    return csv_path, md_path, rows


def build_table4(out_dir: Path) -> tuple[Path, Path, list[dict]]:
    src = ROOT / "results/cicids_strict_mps/posthoc_tad_score_ablation_e8.csv"
    df = pd.read_csv(src)
    sub = df[df["target_fpr"].round(2) == 0.25].copy()
    sub = sub[sub["score_mode"].isin(["critic_only", "feature_deviation_only", "fused"])].copy()

    order = {"critic_only": 0, "feature_deviation_only": 1, "fused": 2}
    sub["_o"] = sub["score_mode"].map(order)
    sub = sub.sort_values("_o")

    rows: list[dict] = []
    for _, r in sub.iterrows():
        mode = str(r["score_mode"])
        if mode == "fused":
            mode_label = "fused (alpha=0.24)"
        elif mode == "critic_only":
            mode_label = "critic_only"
        else:
            mode_label = "feature_deviation_only"
        rows.append(
            {
                "score_mode": mode_label,
                "target_fpr": f4(r["target_fpr"]),
                "observed_test_benign_fpr": f4(r["observed_test_benign_fpr"]),
                "precision": f4(r["precision"]),
                "recall": f4(r["recall"]),
                "F1": f4(r["F1"]),
                "AUC": f4(r["AUC"]),
                "AP": f4(r["AP"]),
            }
        )

    csv_path = out_dir / "table4_tad_score_ablation.csv"
    md_path = out_dir / "table4_tad_score_ablation.md"
    fields = ["score_mode", "target_fpr", "observed_test_benign_fpr", "precision", "recall", "F1", "AUC", "AP"]
    write_csv(csv_path, rows, fields)
    md_text = "# 表4：TAD Score 消融（target_fpr=0.2500）\n\n"
    md_text += "说明：对比 critic-only、feature-deviation-only 与 fused(alpha=0.24) 的标定后表现。\n\n"
    md_text += md_table(rows, fields, ["评分模式", "target_fpr", "测试良性误报率", "Precision", "Recall", "F1", "AUC", "AP"])
    md_path.write_text(md_text, encoding="utf-8")
    return csv_path, md_path, rows


def build_table5(out_dir: Path) -> tuple[Path, Path, list[dict]]:
    src = ROOT / "results/cicids_strict_mps/xai_faithfulness_e8/xai_faithfulness_ig_masking_e8_summary.csv"
    df = pd.read_csv(src).sort_values("k")
    rows: list[dict] = []
    for _, r in df.iterrows():
        rows.append(
            {
                "k": int(r["k"]),
                "original_mean_fused_score": f4(r["original_mean_fused_score"]),
                "ig_mean_score_drop": f4(r["ig_mean_score_drop"]),
                "random_mean_score_drop": f4(r["random_mean_score_drop"]),
                "delta_drop": f4(r["delta_drop"]),
                "ig_beats_random_fraction": f4(r["ig_beats_random_fraction"]),
            }
        )

    csv_path = out_dir / "table5_xai_faithfulness.csv"
    md_path = out_dir / "table5_xai_faithfulness.md"
    fields = ["k", "original_mean_fused_score", "ig_mean_score_drop", "random_mean_score_drop", "delta_drop", "ig_beats_random_fraction"]
    write_csv(csv_path, rows, fields)
    md_text = "# 表5：XAI Faithfulness（IG 掩蔽 vs 随机掩蔽）\n\n"
    md_text += "说明：delta_drop>0 表示 IG 选中特征对异常分数更关键。\n\n"
    md_text += md_table(rows, fields, ["k", "原始均值分数", "IG 均值降幅", "随机均值降幅", "delta_drop", "IG 优于随机比例"])
    md_path.write_text(md_text, encoding="utf-8")
    return csv_path, md_path, rows


def build_table6(out_dir: Path) -> tuple[Path, Path, list[dict]]:
    src = ROOT / "results/cicids_strict_mps/robustness_noise_e8/robustness_gaussian_noise_e8.csv"
    df = pd.read_csv(src)
    keep = [0.00, 0.01, 0.03, 0.05]
    sub = df[df["noise_std"].round(2).isin(keep)].copy().sort_values("noise_std")

    rows: list[dict] = []
    for _, r in sub.iterrows():
        rows.append(
            {
                "noise_std": f4(r["noise_std"]),
                "AUC": f4(r["AUC"]),
                "AP": f4(r["AP"]),
                "observed_test_benign_fpr": f4(r["observed_test_benign_fpr"]),
                "precision": f4(r["precision"]),
                "recall": f4(r["recall"]),
                "F1": f4(r["F1"]),
            }
        )

    csv_path = out_dir / "table6_noise_sensitivity.csv"
    md_path = out_dir / "table6_noise_sensitivity.md"
    fields = ["noise_std", "AUC", "AP", "observed_test_benign_fpr", "precision", "recall", "F1"]
    write_csv(csv_path, rows, fields)
    md_text = "# 表6：噪声敏感性（固定阈值）\n\n"
    md_text += "说明：在固定 threshold=0.4254 下，观察噪声扰动对性能与误报率的影响。\n\n"
    md_text += md_table(rows, fields, ["噪声强度", "AUC", "AP", "测试良性误报率", "Precision", "Recall", "F1"])
    md_path.write_text(md_text, encoding="utf-8")
    return csv_path, md_path, rows


def build_summary(out_dir: Path) -> Path:
    p = out_dir / "final_results_summary_for_paper.md"
    lines = [
        "# 论文主文结果总结（Strict Protocol）",
        "",
        "## 1. CICIDS2017 主表（表1）",
        "- 在 strict no-leakage 协议下，Ours 在 AUC、AP、F1 与 observed_test_benign_fpr 的整体折中上优于代表性 baseline。",
        "- 该结论基于同口径窗口级评估，避免了旧结果中的协议不一致问题。",
        "",
        "## 2. 阈值扫描（表2）",
        "- target_fpr=0.2500 被选为当前工作点。",
        "- 在该点 observed_test_benign_fpr=0.0460，precision/recall/F1 达到较优平衡。",
        "",
        "## 3. 攻击类型分解（表3）",
        "- DDoS 与 PortScan 在 selected operating point 下召回较高。",
        "- Bot 召回显著偏低，属于当前方法局限性（limitation）。",
        "",
        "## 4. TAD Score 消融（表4）",
        "- critic_only 与 feature_deviation_only 各有偏向。",
        "- fused(alpha=0.24) 在标定后综合指标上呈现最佳 calibrated trade-off。",
        "",
        "## 5. XAI Faithfulness（表5）",
        "- IG top-k 掩蔽导致的分数下降显著大于随机掩蔽（delta_drop 为正）。",
        "- 该结果支撑了“告警可审计（auditable）”的主张。",
        "",
        "## 6. 噪声敏感性（表6）",
        "- 轻微噪声下 AUC/AP 仍较稳定，排序能力保持。",
        "- 但在强扰动下，固定阈值对误报率敏感，需要后续考虑自适应阈值或部署侧鲁棒校准。",
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def main() -> None:
    out_dir = ROOT / "results/cicids_strict_mps/final_paper_tables"
    ensure_dir(out_dir)

    artifacts: list[Path] = []
    artifacts += list(build_table1(out_dir)[:2])
    artifacts += list(build_table2(out_dir)[:2])
    artifacts += list(build_table3(out_dir)[:2])
    artifacts += list(build_table4(out_dir)[:2])
    artifacts += list(build_table5(out_dir)[:2])
    artifacts += list(build_table6(out_dir)[:2])
    artifacts.append(build_summary(out_dir))

    print("Generated files:")
    for p in artifacts:
        print(p)


if __name__ == "__main__":
    main()
