#!/usr/bin/env python3
"""
Run a 2x2 ablation for TCN-GAN:
  disc_pooling: mean vs attn
  gan_loss:     vanilla vs wgan-gp

It trains+evals each cell (same window/stride/etc), then generates ablation tables.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run(argv: list[str]) -> None:
    p = subprocess.run(argv)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="TCN-GAN 2x2 ablation runner (mean/attn × vanilla/wgan-gp).")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--data-dir", default="attack/dataset/CICIDS2017")
    ap.add_argument("--train-files", nargs="+", required=True)
    ap.add_argument("--test-files", nargs="+", required=True)
    ap.add_argument("--window-size", type=int, default=128)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--anomaly-ratio", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--score-alpha", type=float, default=0.6)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--xai-samples", type=int, default=256)
    ap.add_argument("--xai-batch-size", type=int, default=128)
    ap.add_argument("--out-root", default="attack/results/tcn_gan_ablation_runs")
    args = ap.parse_args()

    out_dir = Path(args.out_root) / _now_tag()
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = [
        ("mean", "vanilla"),
        ("attn", "vanilla"),
        ("mean", "wgan-gp"),
        ("attn", "wgan-gp"),
    ]
    for pooling, gan_loss in combos:
        tag = f"{pooling}_{gan_loss}".replace("-", "")
        ckpt = out_dir / f"ckpt_{tag}.pt"
        out_json = out_dir / f"eval_{tag}.json"
        argv = [
            str(args.python),
            "attack/pipelines/run_tcn_gan_autotune.py",
            "--data-dir",
            str(args.data_dir),
            "--train-files",
            *args.train_files,
            "--test-files",
            *args.test_files,
            "--window-grid",
            str(args.window_size),
            "--stride-ratio",
            str(float(args.stride) / float(args.window_size)),
            "--epochs",
            str(args.epochs),
            "--disc-pooling",
            pooling,
            "--gan-loss",
            gan_loss,
            "--score-mode",
            "fused",
            "--score-alpha",
            str(args.score_alpha),
            "--target-fpr",
            str(args.target_fpr),
            "--metric-prefer",
            "calib_f1",
            "--xai",
            "--xai-only-best",
            "--xai-samples",
            str(args.xai_samples),
            "--xai-batch-size",
            str(args.xai_batch_size),
            "--out-root",
            str(out_dir),
        ]
        _run(argv)

    # Generate tables from this run directory
    _run(
        [
            str(args.python),
            "attack/pipelines/make_tcn_gan_ablation_table.py",
            "--glob",
            str(out_dir / "*" / "eval_w*_s*_*.json"),
            "--out-md",
            str(out_dir / "ablation.md"),
            "--out-csv",
            str(out_dir / "ablation.csv"),
            "--out-md-summary",
            str(out_dir / "ablation_summary.md"),
            "--out-csv-summary",
            str(out_dir / "ablation_summary.csv"),
        ]
    )
    print(f"Done. See: {out_dir}")


if __name__ == "__main__":
    main()
