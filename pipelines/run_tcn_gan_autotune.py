#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    """
    When executed as `python attack/pipelines/xxx.py`, sys.path[0] becomes `attack/pipelines`,
    which makes `import attack.*` fail. Add repo root so `attack/` is importable.
    """
    root = Path(__file__).resolve().parents[2]
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


@dataclass
class RunResult:
    window: int
    stride: int
    score_mode: str
    score_alpha: float
    target_fpr: float | None
    ckpt_path: str
    out_json: str
    log_train: str
    log_eval: str
    xai_report_json: str | None
    metrics: dict[str, object]
    calibrated: dict[str, object] | None


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run(argv: list[str], log_path: str) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        p = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise SystemExit(f"Command failed ({p.returncode}): {' '.join(argv)}\nSee log: {log_path}")


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _metric_key(d: dict, prefer: str) -> float:
    """
    prefer:
      - 'calib_f1'
      - 'calib_recall'
      - 'auc'
    """
    if prefer == "calib_f1":
        return float(d.get("calibrated", {}).get("f1", float("nan")))
    if prefer == "calib_recall":
        return float(d.get("calibrated", {}).get("recall", float("nan")))
    return float(d.get("metrics", {}).get("auc", float("nan")))


def _write_summary(
    out_dir: Path,
    results: list[RunResult],
    *,
    metric_prefer: str,
) -> None:
    # CSV
    csv_path = out_dir / "tcn_gan_autotune_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "window",
                "stride",
                "score_mode",
                "score_alpha",
                "target_fpr",
                "auc",
                "ap",
                "best_f1",
                "best_precision",
                "best_recall",
                "best_fpr",
                "calib_threshold",
                "calib_f1",
                "calib_precision",
                "calib_recall",
                "calib_fpr",
                "train_benign_fpr",
                "test_benign_fpr",
                "ckpt_path",
                "out_json",
                "xai_report_json",
                "log_train",
                "log_eval",
            ]
        )
        for r in results:
            m = r.metrics or {}
            calib = r.calibrated or {}
            w.writerow(
                [
                    r.window,
                    r.stride,
                    r.score_mode,
                    f"{r.score_alpha:.3f}",
                    "" if r.target_fpr is None else f"{r.target_fpr:.3f}",
                    m.get("auc"),
                    m.get("ap"),
                    m.get("best_f1"),
                    m.get("best_precision"),
                    m.get("best_recall"),
                    m.get("best_fpr"),
                    calib.get("threshold"),
                    calib.get("f1"),
                    calib.get("precision"),
                    calib.get("recall"),
                    calib.get("fpr"),
                    calib.get("train_benign_fpr"),
                    calib.get("test_benign_fpr"),
                    r.ckpt_path,
                    r.out_json,
                    r.xai_report_json or "",
                    r.log_train,
                    r.log_eval,
                ]
            )

    # Markdown summary
    md_path = out_dir / "README.md"
    ranked = sorted(results, key=lambda rr: _metric_key(_load_json(rr.out_json), metric_prefer), reverse=True)
    best = ranked[0] if ranked else None
    lines: list[str] = []
    lines.append("# TCN-GAN autotune run")
    lines.append("")
    lines.append(f"- Results CSV: `{csv_path}`")
    if best is not None:
        d = _load_json(best.out_json)
        lines.append(f"- Best (by `{metric_prefer}`): window={best.window} stride={best.stride} score={best.score_mode}")
        if best.score_mode == "fused":
            lines.append(f"  - score_alpha={best.score_alpha:.2f}")
        lines.append(f"  - out_json: `{best.out_json}`")
        if d.get("xai", {}).get("enabled"):
            lines.append(f"  - xai_report: `{d.get('xai', {}).get('report_json','')}`")
            lines.append(f"  - xai_time_plot: `{d.get('xai', {}).get('plot_time_importance','')}`")
            lines.append(f"  - xai_feat_plot: `{d.get('xai', {}).get('plot_feature_importance','')}`")
    lines.append("")
    lines.append("## All runs (sorted)")
    lines.append("")
    lines.append("| rank | window | stride | score | alpha | AUC | AP | calib_f1 | calib_rec | logs |")
    lines.append("| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for i, r in enumerate(ranked, start=1):
        d = _load_json(r.out_json)
        m = d.get("metrics", {})
        c = d.get("calibrated", {})
        auc = m.get("auc")
        ap = m.get("ap")
        calib_f1 = c.get("f1")
        calib_rec = c.get("recall")
        alpha = r.score_alpha if r.score_mode == "fused" else ""
        lines.append(
            f"| {i} | {r.window} | {r.stride} | {r.score_mode} | {alpha} | {auc} | {ap} | {calib_f1} | {calib_rec} | `{r.log_eval}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Optional quick plot (best-effort)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        xs = list(range(1, len(ranked) + 1))
        ys = []
        labels = []
        for r in ranked:
            d = _load_json(r.out_json)
            if metric_prefer.startswith("calib_"):
                ys.append(float(d.get("calibrated", {}).get(metric_prefer.replace("calib_", ""), float("nan"))))
            else:
                ys.append(float(d.get("metrics", {}).get(metric_prefer, float("nan"))))
            labels.append(f"w{r.window}s{r.stride}")
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(xs, ys, marker="o")
        ax.set_title(f"TCN-GAN autotune: {metric_prefer} by rank")
        ax.set_xlabel("rank (higher is better)")
        ax.set_ylabel(metric_prefer)
        ax.grid(True, alpha=0.25)
        for x, y, lab in zip(xs, ys, labels, strict=True):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        fig.tight_layout()
        fig_path = out_dir / "tcn_gan_autotune_rank_plot.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
    except Exception:
        pass


def _train_and_eval_one(
    *,
    py: str,
    data_dir: str,
    train_files: list[str],
    test_files: list[str],
    window: int,
    stride: int,
    anomaly_ratio: float,
    epochs: int,
    batch_size: int,
    test_batch_size: int,
    lr: float,
    seed: int,
    disc_pooling: str,
    gan_loss: str,
    gp_lambda: float,
    n_critic: int,
    score_mode: str,
    score_alpha: float,
    target_fpr: float | None,
    xai: bool,
    xai_samples: int,
    xai_batch_size: int,
    out_dir: Path,
) -> RunResult:
    _ensure_repo_root_on_path()

    ckpt = out_dir / f"ckpt_w{window}_s{stride}.pt"
    out_json = out_dir / f"eval_w{window}_s{stride}_{score_mode}.json"
    log_train = out_dir / f"train_w{window}_s{stride}.log"
    log_eval = out_dir / f"eval_w{window}_s{stride}_{score_mode}.log"

    # Train (save best)
    train_argv = [
        py,
        "attack/models/tcn_gan_experiment.py",
        "--data-dir",
        data_dir,
        "--train-files",
        *train_files,
        "--test-files",
        *test_files,
        "--window-size",
        str(window),
        "--stride",
        str(stride),
        "--anomaly-ratio",
        str(anomaly_ratio),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--test-batch-size",
        str(test_batch_size),
        "--lr",
        str(lr),
        "--seed",
        str(seed),
        "--disc-pooling",
        str(disc_pooling),
        "--gan-loss",
        str(gan_loss),
        "--gp-lambda",
        str(gp_lambda),
        "--n-critic",
        str(n_critic),
        "--save-best",
        str(ckpt),
    ]
    _run(train_argv, str(log_train))

    # Eval (with score + optional calibration + optional XAI)
    eval_argv = [
        py,
        "attack/models/tcn_gan_experiment.py",
        "--data-dir",
        data_dir,
        "--train-files",
        *train_files,
        "--test-files",
        *test_files,
        "--window-size",
        str(window),
        "--stride",
        str(stride),
        "--anomaly-ratio",
        str(anomaly_ratio),
        "--load",
        str(ckpt),
        "--eval-only",
        "--disc-pooling",
        str(disc_pooling),
        "--gan-loss",
        str(gan_loss),
        "--gp-lambda",
        str(gp_lambda),
        "--n-critic",
        str(n_critic),
        "--score-mode",
        str(score_mode),
        "--score-alpha",
        str(score_alpha),
        "--out-json",
        str(out_json),
    ]
    if target_fpr is not None:
        eval_argv += ["--target-fpr", str(target_fpr)]
    if xai:
        eval_argv += ["--xai-report", "--xai-samples", str(xai_samples), "--xai-batch-size", str(xai_batch_size)]
    _run(eval_argv, str(log_eval))

    d = _load_json(str(out_json))
    xai_report_json = None
    if isinstance(d.get("xai"), dict) and d.get("xai", {}).get("enabled"):
        xai_report_json = str(d.get("xai", {}).get("report_json") or "")

    return RunResult(
        window=int(window),
        stride=int(stride),
        score_mode=str(score_mode),
        score_alpha=float(score_alpha),
        target_fpr=float(target_fpr) if target_fpr is not None else None,
        ckpt_path=str(ckpt),
        out_json=str(out_json),
        log_train=str(log_train),
        log_eval=str(log_eval),
        xai_report_json=xai_report_json or None,
        metrics=dict(d.get("metrics", {})),
        calibrated=dict(d.get("calibrated", {})) if isinstance(d.get("calibrated"), dict) else None,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Auto-run TCN-GAN sweeps with logging (window sweep + optional stride sweep) and optional XAI reports."
    )
    ap.add_argument("--python", default=sys.executable, help="Python executable (default: current)")
    ap.add_argument("--data-dir", default="attack/dataset/CICIDS2017")
    ap.add_argument("--train-files", nargs="+", required=True)
    ap.add_argument("--test-files", nargs="+", required=True)

    ap.add_argument(
        "--window-grid",
        nargs="+",
        default=None,
        help="(optional) comma list (spaces ok), e.g. 16,32,64,128. "
        "If omitted, defaults to 16,32,64,128 for window sweep. "
        "If --fixed-window is set, the default window sweep is skipped unless you explicitly pass --window-grid.",
    )
    ap.add_argument(
        "--stride-ratio",
        type=float,
        default=0.125,
        help="for window sweep: stride = max(1, round(window*ratio)). Default 0.125 (window/8)",
    )
    ap.add_argument(
        "--stride-grid",
        nargs="+",
        default=[""],
        help="optional: for fixed-window stride sweep, comma list, e.g. 1,2,4,8,16 (requires --fixed-window)",
    )
    ap.add_argument("--fixed-window", type=int, default=0, help="if >0, run stride sweep at this window")

    ap.add_argument("--anomaly-ratio", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--test-batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--disc-pooling", choices=["mean", "attn"], default="mean")
    ap.add_argument("--gan-loss", choices=["vanilla", "wgan-gp"], default="vanilla")
    ap.add_argument("--gp-lambda", type=float, default=10.0)
    ap.add_argument("--n-critic", type=int, default=5)

    ap.add_argument("--score-mode", default="fused", choices=["prob", "feat_l2", "feat_mahal", "fused"])
    ap.add_argument("--score-alpha", type=float, default=0.6)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--metric-prefer", default="calib_f1", choices=["calib_f1", "calib_recall", "auc"])

    ap.add_argument("--xai", action="store_true", help="generate XAI report for each run (slow)")
    ap.add_argument("--xai-only-best", action="store_true", help="only generate XAI for the best run (recommended)")
    ap.add_argument("--xai-samples", type=int, default=512)
    ap.add_argument("--xai-batch-size", type=int, default=128)

    ap.add_argument("--out-root", default="attack/results/final_experiments/manual_autotune_runs")
    ap.add_argument("--skip-existing", action="store_true", help="skip runs whose eval JSON already exists")
    ap.add_argument(
        "--post-viz",
        action="store_true",
        help="after runs, generate extra figures (aligned XAI time plot + window-metrics plot) if matplotlib is available",
    )

    args = ap.parse_args()

    _ensure_repo_root_on_path()

    out_dir = Path(args.out_root) / _now_tag()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decide whether to run window sweep.
    # - If user explicitly provides --window-grid: always run it (even with --fixed-window).
    # - If not provided:
    #   - run default window sweep only when --fixed-window is not set
    #   - otherwise skip the window sweep and only do the fixed-window stride sweep.
    windows: list[int] = []
    if args.window_grid is not None:
        window_grid_text = " ".join([str(x) for x in (args.window_grid or [])]).replace(" ", "")
        windows = [int(s.strip()) for s in window_grid_text.split(",") if s.strip()]
        if not windows:
            raise SystemExit("--window-grid 不能为空")
    elif int(args.fixed_window) <= 0:
        windows = [16, 32, 64, 128]

    # ---- window sweep (train each window, stride derived by ratio) ----
    results: list[RunResult] = []
    planned: list[tuple[int, int]] = []
    if windows:
        for w in windows:
            stride = max(1, int(round(float(w) * float(args.stride_ratio))))
            planned.append((int(w), int(stride)))

    # Decide whether to run XAI per run or only for best. If only-best, we first run without XAI.
    want_xai_each = bool(args.xai) and not bool(args.xai_only_best)
    for w, s in planned:
        out_json = out_dir / f"eval_w{w}_s{s}_{str(args.score_mode)}.json"
        if bool(args.skip_existing) and out_json.exists():
            d = _load_json(str(out_json))
            results.append(
                RunResult(
                    window=w,
                    stride=s,
                    score_mode=str(args.score_mode),
                    score_alpha=float(args.score_alpha),
                    target_fpr=float(args.target_fpr) if args.target_fpr is not None else None,
                    ckpt_path=str(out_dir / f"ckpt_w{w}_s{s}.pt"),
                    out_json=str(out_json),
                    log_train=str(out_dir / f"train_w{w}_s{s}.log"),
                    log_eval=str(out_dir / f"eval_w{w}_s{s}_{str(args.score_mode)}.log"),
                    xai_report_json=str(d.get("xai", {}).get("report_json") or "") if isinstance(d.get("xai"), dict) else None,
                    metrics=dict(d.get("metrics", {})),
                    calibrated=dict(d.get("calibrated", {})) if isinstance(d.get("calibrated"), dict) else None,
                )
            )
            continue

        results.append(
            _train_and_eval_one(
                py=str(args.python),
                data_dir=str(args.data_dir),
                train_files=list(args.train_files),
                test_files=list(args.test_files),
                window=w,
                stride=s,
                anomaly_ratio=float(args.anomaly_ratio),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                test_batch_size=int(args.test_batch_size),
                lr=float(args.lr),
                seed=int(args.seed),
                disc_pooling=str(args.disc_pooling),
                gan_loss=str(args.gan_loss),
                gp_lambda=float(args.gp_lambda),
                n_critic=int(args.n_critic),
                score_mode=str(args.score_mode),
                score_alpha=float(args.score_alpha),
                target_fpr=float(args.target_fpr) if args.target_fpr is not None else None,
                xai=want_xai_each,
                xai_samples=int(args.xai_samples),
                xai_batch_size=int(args.xai_batch_size),
                out_dir=out_dir,
            )
        )

    # ---- optional stride sweep at a fixed window ----
    if int(args.fixed_window) > 0:
        stride_grid_text = " ".join([str(x) for x in (args.stride_grid or [])]).replace(" ", "")
        if not stride_grid_text.strip().strip(","):
            raise SystemExit("使用 --fixed-window 需要同时提供 --stride-grid，例如 1,2,4,8,16")
        strides = [int(s.strip()) for s in stride_grid_text.split(",") if s.strip()]
        if not strides:
            raise SystemExit("--stride-grid 不能为空")
        fw = int(args.fixed_window)
        for s in strides:
            out_json = out_dir / f"eval_w{fw}_s{s}_{str(args.score_mode)}.json"
            if bool(args.skip_existing) and out_json.exists():
                d = _load_json(str(out_json))
                results.append(
                    RunResult(
                        window=fw,
                        stride=int(s),
                        score_mode=str(args.score_mode),
                        score_alpha=float(args.score_alpha),
                        target_fpr=float(args.target_fpr) if args.target_fpr is not None else None,
                        ckpt_path=str(out_dir / f"ckpt_w{fw}_s{s}.pt"),
                        out_json=str(out_json),
                        log_train=str(out_dir / f"train_w{fw}_s{s}.log"),
                        log_eval=str(out_dir / f"eval_w{fw}_s{s}_{str(args.score_mode)}.log"),
                        xai_report_json=str(d.get("xai", {}).get("report_json") or "")
                        if isinstance(d.get("xai"), dict)
                        else None,
                        metrics=dict(d.get("metrics", {})),
                        calibrated=dict(d.get("calibrated", {})) if isinstance(d.get("calibrated"), dict) else None,
                    )
                )
                continue
            results.append(
                _train_and_eval_one(
                    py=str(args.python),
                    data_dir=str(args.data_dir),
                    train_files=list(args.train_files),
                    test_files=list(args.test_files),
                    window=fw,
                    stride=int(s),
                    anomaly_ratio=float(args.anomaly_ratio),
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    test_batch_size=int(args.test_batch_size),
                    lr=float(args.lr),
                    seed=int(args.seed),
                    disc_pooling=str(args.disc_pooling),
                    gan_loss=str(args.gan_loss),
                    gp_lambda=float(args.gp_lambda),
                    n_critic=int(args.n_critic),
                    score_mode=str(args.score_mode),
                    score_alpha=float(args.score_alpha),
                    target_fpr=float(args.target_fpr) if args.target_fpr is not None else None,
                    xai=want_xai_each,
                    xai_samples=int(args.xai_samples),
                    xai_batch_size=int(args.xai_batch_size),
                    out_dir=out_dir,
                )
            )

    # If xai-only-best, re-run eval for the best run with --xai-report enabled (no retrain).
    if bool(args.xai) and bool(args.xai_only_best) and results:
        ranked = sorted(results, key=lambda rr: _metric_key(_load_json(rr.out_json), str(args.metric_prefer)), reverse=True)
        best = ranked[0]
        d0 = _load_json(best.out_json)
        if not (isinstance(d0.get("xai"), dict) and d0.get("xai", {}).get("enabled")):
            # Run eval again with XAI, overwrite JSON to include xai field (more convenient for downstream tables).
            eval_argv = [
                str(args.python),
                "attack/models/tcn_gan_experiment.py",
                "--data-dir",
                str(args.data_dir),
                "--train-files",
                *list(args.train_files),
                "--test-files",
                *list(args.test_files),
                "--window-size",
                str(best.window),
                "--stride",
                str(best.stride),
                "--anomaly-ratio",
                str(args.anomaly_ratio),
                "--load",
                str(best.ckpt_path),
                "--eval-only",
                "--disc-pooling",
                str(args.disc_pooling),
                "--gan-loss",
                str(args.gan_loss),
                "--gp-lambda",
                str(args.gp_lambda),
                "--n-critic",
                str(args.n_critic),
                "--score-mode",
                str(best.score_mode),
                "--score-alpha",
                str(best.score_alpha),
                "--out-json",
                str(best.out_json),
                "--target-fpr",
                str(args.target_fpr),
                "--xai-report",
                "--xai-samples",
                str(args.xai_samples),
                "--xai-batch-size",
                str(args.xai_batch_size),
            ]
            _run(eval_argv, str(out_dir / f"eval_best_xai_w{best.window}_s{best.stride}.log"))
            d1 = _load_json(best.out_json)
            best.metrics = dict(d1.get("metrics", {}))
            best.calibrated = dict(d1.get("calibrated", {})) if isinstance(d1.get("calibrated"), dict) else None
            if isinstance(d1.get("xai"), dict) and d1.get("xai", {}).get("enabled"):
                best.xai_report_json = str(d1.get("xai", {}).get("report_json") or "")

    _write_summary(out_dir, results, metric_prefer=str(args.metric_prefer))

    if bool(args.post_viz):
        try:
            subprocess.run(
                [str(args.python), "attack/pipelines/plot_tcn_gan_autotune_viz.py", "--run-dir", str(out_dir)],
                check=False,
            )
        except Exception:
            pass
    print(f"Done. See: {out_dir}")


if __name__ == "__main__":
    main()
