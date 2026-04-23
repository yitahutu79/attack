#!/usr/bin/env python3
"""
Post-process a run directory from `run_tcn_gan_autotune.py` and generate:
1) Aligned XAI time-importance comparison plot across windows
2) Window vs (calib_f1 / calib_recall / test_benign_fpr) plot

This script does NOT retrain anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def _safe_float(x: object, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if not s:
            return default
        return float(s)
    except Exception:
        return default


def _interp(xs: list[float], ys: list[float], new_xs: list[float]) -> list[float]:
    # Simple linear interpolation without numpy.
    if not xs or not ys or len(xs) != len(ys):
        return [float("nan")] * len(new_xs)
    out: list[float] = []
    j = 0
    n = len(xs)
    for x in new_xs:
        while j + 1 < n and xs[j + 1] < x:
            j += 1
        if j + 1 >= n:
            out.append(float(ys[-1]))
            continue
        x0, x1 = xs[j], xs[j + 1]
        y0, y1 = ys[j], ys[j + 1]
        if x1 <= x0:
            out.append(float(y0))
            continue
        t = (x - x0) / (x1 - x0)
        out.append(float(y0 + t * (y1 - y0)))
    return out


def plot_xai_time_aligned(run_dir: Path, *, out_path: Path, points: int = 256) -> None:
    # Find eval jsons and their xai reports
    eval_jsons = sorted(run_dir.glob("eval_w*_s*_*.json"))
    items: list[tuple[int, int, Path]] = []
    for p in eval_jsons:
        d = _load_json(p)
        xai = d.get("xai", {})
        if isinstance(xai, dict) and xai.get("enabled") and xai.get("report_json"):
            rp = Path(str(xai["report_json"]))
            if rp.exists():
                # parse window/stride from filename
                stem = p.stem
                # eval_w128_s16_fused
                try:
                    w = int(stem.split("_w", 1)[1].split("_", 1)[0])
                    s = int(stem.split("_s", 1)[1].split("_", 1)[0])
                except Exception:
                    continue
                items.append((w, s, rp))
    if not items:
        raise SystemExit(f"No XAI reports found in {run_dir}")

    # Sort by window ascending for nicer legend
    items.sort(key=lambda x: x[0])

    new_x = [i / (points - 1) for i in range(points)]
    curves: list[dict[str, object]] = []
    for w, s, rp in items:
        r = _load_json(rp)
        an = [float(v) for v in r["time_importance"]["anomaly"]]
        be = [float(v) for v in r["time_importance"]["benign"]]
        xs = [i / (len(an) - 1) for i in range(len(an))] if len(an) > 1 else [0.0]
        an_i = _interp(xs, an, new_x)
        be_i = _interp(xs, be, new_x)
        diff_i = [a - b for a, b in zip(an_i, be_i, strict=True)]
        curves.append({"window": w, "stride": s, "an": an_i, "be": be_i, "diff": diff_i})

    # Plot (matplotlib optional)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"matplotlib not available: {e}") from e

    fig = plt.figure(figsize=(11.5, 6.5))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

    for c in curves:
        w = int(c["window"])
        s = int(c["stride"])
        label = f"w{w}s{s}"
        ax1.plot(new_x, c["an"], label=f"ANOM {label}")
        ax1.plot(new_x, c["be"], linestyle="--", alpha=0.6, label=f"BENIGN {label}")
        ax2.plot(new_x, c["diff"], label=label)

    ax1.set_title("XAI time importance aligned by relative position (|grad*input|)")
    ax1.set_ylabel("importance")
    ax1.grid(True, alpha=0.25)
    ax1.legend(ncol=2, fontsize=8)

    ax2.set_title("ANOM - BENIGN (aligned)")
    ax2.set_xlabel("relative t within window (0..1)")
    ax2.set_ylabel("delta importance")
    ax2.grid(True, alpha=0.25)
    ax2.legend(ncol=4, fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_window_metrics(run_dir: Path, *, out_path: Path) -> None:
    csv_path = run_dir / "tcn_gan_autotune_results.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing CSV: {csv_path}")
    rows = _read_csv_rows(csv_path)
    if not rows:
        raise SystemExit(f"Empty CSV: {csv_path}")

    # Prefer reading test_benign_fpr from eval json (more precise); fall back to CSV if present.
    eval_map: dict[tuple[int, int], dict] = {}
    for p in run_dir.glob("eval_w*_s*_*.json"):
        d = _load_json(p)
        try:
            w = int(d.get("window_size"))
            s = int(d.get("stride"))
        except Exception:
            continue
        eval_map[(w, s)] = d

    pts: list[tuple[int, int, float, float, float]] = []
    for r in rows:
        w = int(float(r["window"]))
        s = int(float(r["stride"]))
        calib_f1 = _safe_float(r.get("calib_f1"))
        calib_rec = _safe_float(r.get("calib_recall"))
        test_benign_fpr = _safe_float(r.get("test_benign_fpr"))
        d = eval_map.get((w, s))
        if d and isinstance(d.get("calibrated"), dict):
            test_benign_fpr = _safe_float(d["calibrated"].get("test_benign_fpr"), test_benign_fpr)
        pts.append((w, s, calib_f1, calib_rec, test_benign_fpr))

    pts.sort(key=lambda x: x[0])
    xs = [p[0] for p in pts]
    strides = [p[1] for p in pts]
    f1s = [p[2] for p in pts]
    recs = [p[3] for p in pts]
    fprs = [p[4] for p in pts]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"matplotlib not available: {e}") from e

    fig = plt.figure(figsize=(11, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax2 = ax.twinx()

    ax.plot(xs, f1s, marker="o", label="calib_f1 (target_fpr=0.05)")
    ax.plot(xs, recs, marker="o", label="calib_recall (target_fpr=0.05)")
    ax2.plot(xs, fprs, marker="s", color="tab:red", label="test_benign_fpr (at calibrated threshold)")

    for x, s in zip(xs, strides, strict=True):
        ax.annotate(f"s={s}", (x, 0), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    ax.set_title("Window sweep metrics (fixed target_fpr=0.05 calibration)")
    ax.set_xlabel("window size")
    ax.set_ylabel("F1 / Recall")
    ax2.set_ylabel("Test BENIGN FPR")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0.0, top=1.02)
    ax2.set_ylim(bottom=0.0, top=1.02)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot extra figures for a TCN-GAN autotune run directory.")
    ap.add_argument("--run-dir", required=True, help="e.g. attack/results/final_experiments/20260414_132416/stride_sweep/combined")
    ap.add_argument("--points", type=int, default=256, help="resample points for aligned time plot")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run-dir not found: {run_dir}")

    plot_xai_time_aligned(run_dir, out_path=run_dir / "xai_time_aligned.png", points=int(args.points))
    plot_window_metrics(run_dir, out_path=run_dir / "window_metrics.png")
    print(f"Wrote: {run_dir / 'xai_time_aligned.png'}")
    print(f"Wrote: {run_dir / 'window_metrics.png'}")


if __name__ == "__main__":
    main()
