#!/usr/bin/env python3
"""Measure inference runtime for saved TCN checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["cicids2017", "swat", "unsw_nb15", "ton_iot"])
    p.add_argument("--data-dir", required=True)
    p.add_argument("--load", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--disable-weight-norm", action="store_true")
    p.add_argument("--train-files", nargs="*", default=[])
    p.add_argument("--test-files", nargs="*", default=[])
    p.add_argument("--unsupervised-formal-split", action="store_true")
    p.add_argument("--chrono-unsupervised-split", action="store_true")
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--anomaly-ratio", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--score-mode", default="fused")
    p.add_argument("--score-alpha", type=float, default=0.24)
    p.add_argument("--ref-max", type=int, default=20000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.disable_weight_norm:
        os.environ["TCN_GAN_DISABLE_WEIGHT_NORM"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(root.parent) not in sys.path:
        sys.path.insert(0, str(root.parent))

    import numpy as np
    import torch

    from attack.dataset_loaders import (
        load_windowed_chrono_unsupervised_split,
        load_windowed_dataset,
        load_windowed_unsupervised_split,
    )
    from attack.models.tcn_gan_experiment import TCNDiscriminator, compute_anomaly_scores

    ckpt = torch.load(args.load, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    hidden = list(ckpt_args.get("hidden_channels", [128, 128]))
    dropout = float(ckpt_args.get("dropout", 0.2))
    pooling = str(ckpt_args.get("disc_pooling", "attn"))

    if args.unsupervised_formal_split:
        ds = load_windowed_unsupervised_split(
            dataset=args.dataset,
            data_dir=args.data_dir,
            benign_file="",
            attack_file="",
            window_size=args.window_size,
            stride=args.stride,
            anomaly_ratio=args.anomaly_ratio,
            progress=False,
        )
    elif args.chrono_unsupervised_split:
        ds = load_windowed_chrono_unsupervised_split(
            dataset=args.dataset,
            data_dir=args.data_dir,
            mixed_file="",
            window_size=args.window_size,
            stride=args.stride,
            anomaly_ratio=args.anomaly_ratio,
            train_fraction=0.6,
            calib_fraction=0.1,
            progress=False,
        )
    else:
        ds = load_windowed_dataset(
            dataset=args.dataset,
            data_dir=args.data_dir,
            train_files=args.train_files,
            test_files=args.test_files,
            window_size=args.window_size,
            stride=args.stride,
            anomaly_ratio=args.anomaly_ratio,
            train_benign_only=True,
            progress=False,
        )

    disc = TCNDiscriminator(
        seq_len=args.window_size,
        feat_dim=int(ds.x_train.shape[2]),
        hidden_channels=hidden,
        dropout=dropout,
        pooling=pooling,
    )
    disc.load_state_dict(ckpt["discriminator"])
    device = torch.device("cpu")
    disc.to(device).eval()

    x_ref = ds.x_calib if ds.x_calib is not None and len(ds.x_calib) else ds.x_train
    if len(x_ref) > int(args.ref_max):
        x_ref = x_ref[: int(args.ref_max)]
    # Warm up once to avoid counting one-time kernel/setup cost.
    _ = compute_anomaly_scores(
        disc,
        ds.x_test[: min(len(ds.x_test), args.batch_size)],
        device,
        args.batch_size,
        score_mode=args.score_mode,
        score_alpha=args.score_alpha,
        x_ref_benign=x_ref,
    )
    t0 = time.perf_counter()
    scores = compute_anomaly_scores(
        disc,
        ds.x_test,
        device,
        args.batch_size,
        score_mode=args.score_mode,
        score_alpha=args.score_alpha,
        x_ref_benign=x_ref,
    )
    t1 = time.perf_counter()
    if len(scores) != len(ds.x_test):
        raise RuntimeError("score count mismatch")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "dataset": args.label or args.dataset,
        "test_windows": int(len(ds.x_test)),
        "eval_seconds": float(t1 - t0),
        "windows_per_sec": float(len(ds.x_test) / max(t1 - t0, 1e-12)),
        "checkpoint_mb": float(Path(args.load).stat().st_size / (1024 * 1024)),
        "feature_dim": int(ds.x_train.shape[2]),
        "score_mode": str(args.score_mode),
    }
    exists = out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
