#!/usr/bin/env python3
"""Lightweight Gaussian-noise robustness for the final TCN detector."""

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
    p.add_argument("--data-dir", required=True)
    p.add_argument("--train-files", nargs="+", required=True)
    p.add_argument("--test-files", nargs="+", required=True)
    p.add_argument("--load", required=True)
    p.add_argument("--out-csv", default="paper/tables/noise_robustness.csv")
    p.add_argument("--out-md", default="paper/tables/noise_robustness.md")
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--anomaly-ratio", type=float, default=0.15)
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--score-alpha", type=float, default=0.24)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--ref-max", type=int, default=20000)
    p.add_argument("--calib-max", type=int, default=50000)
    p.add_argument("--test-max", type=int, default=12000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
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

    from attack.dataset_loaders import load_windowed_dataset
    from attack.models.tcn_gan_experiment import (
        TCNDiscriminator,
        embed_sequences,
        feature_deviation_scores,
        metrics_at_threshold,
        threshold_from_benign_fpr,
    )

    rng = np.random.default_rng(int(args.seed))
    ds = load_windowed_dataset(
        dataset="cicids2017",
        data_dir=args.data_dir,
        train_files=args.train_files,
        test_files=args.test_files,
        window_size=args.window_size,
        stride=args.stride,
        anomaly_ratio=args.anomaly_ratio,
        train_benign_only=True,
        scaler="minmax",
        clip_minmax=True,
        progress=False,
    )
    if int(args.test_max) > 0 and len(ds.x_test) > int(args.test_max):
        y = np.asarray(ds.y_test).reshape(-1)
        idx_b = np.where(y == 0)[0]
        idx_a = np.where(y == 1)[0]
        n_a = min(len(idx_a), int(args.test_max) // 2)
        n_b = min(len(idx_b), int(args.test_max) - n_a)
        sel = np.concatenate([
            rng.choice(idx_a, size=n_a, replace=False),
            rng.choice(idx_b, size=n_b, replace=False),
        ])
        rng.shuffle(sel)
        x_test = ds.x_test[sel]
        y_test = ds.y_test[sel]
    else:
        x_test = ds.x_test
        y_test = ds.y_test
    ckpt = torch.load(args.load, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    disc = TCNDiscriminator(
        seq_len=int(args.window_size),
        feat_dim=int(ds.x_train.shape[2]),
        hidden_channels=list(ckpt_args.get("hidden_channels", [128, 128])),
        dropout=float(ckpt_args.get("dropout", 0.2)),
        pooling=str(ckpt_args.get("disc_pooling", "attn")),
    )
    disc.load_state_dict(ckpt["discriminator"])
    device = torch.device("cpu")
    disc.to(device).eval()

    ref = ds.x_train[: min(len(ds.x_train), int(args.ref_max))]
    calib = ds.x_train[: min(len(ds.x_train), int(args.calib_max))]
    emb_ref, unk_ref = embed_sequences(disc, ref, device, int(args.batch_size))
    dev_ref = feature_deviation_scores(emb_ref, emb_ref, method="l2")

    def fused_scores(x: np.ndarray) -> np.ndarray:
        emb, unk = embed_sequences(disc, x, device, int(args.batch_size))
        dev = feature_deviation_scores(emb, emb_ref, method="l2")
        dev_n = np.clip((dev - float(dev_ref.min())) / max(float(dev_ref.max() - dev_ref.min()), 1e-12), 0.0, 1.0)
        unk_n = np.clip((unk - float(unk_ref.min())) / max(float(unk_ref.max() - unk_ref.min()), 1e-12), 0.0, 1.0)
        return (float(args.score_alpha) * unk_n + (1.0 - float(args.score_alpha)) * dev_n).astype(np.float32)

    calib_scores = fused_scores(calib)
    threshold = threshold_from_benign_fpr(calib_scores, float(args.target_fpr))

    scenarios = [("Clean", 0.0), ("Noise 1%", 0.01), ("Noise 3%", 0.03), ("Noise 5%", 0.05)]
    rows: list[dict[str, object]] = []
    t0 = time.perf_counter()
    for name, sigma in scenarios:
        if sigma > 0:
            x = np.clip(x_test + rng.normal(0.0, sigma, x_test.shape).astype(np.float32), 0.0, 1.0)
        else:
            x = x_test
        scores = fused_scores(x)
        m = metrics_at_threshold(y_test, scores, threshold)
        rows.append(
            {
                "scenario": name,
                "sigma": sigma,
                "f1": float(m["f1"]),
                "recall": float(m["recall"]),
                "precision": float(m["precision"]),
                "test_fpr": float(m["fpr"]),
            }
        )
    elapsed = time.perf_counter() - t0

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "sigma", "f1", "recall", "precision", "test_fpr"])
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# Gaussian Noise Robustness",
        "",
        f"Target FPR={float(args.target_fpr):.2f}; threshold calibrated on clean benign windows. Eval seconds={elapsed:.2f}.",
        "",
        "| Scenario | Sigma | F1 | Recall | Precision | Test FPR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            "| {scenario} | {sigma:.2f} | {f1:.4f} | {recall:.4f} | {precision:.4f} | {test_fpr:.4f} |".format(**r)
        )
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_json = out_csv.with_suffix(".json")
    out_json.write_text(
        json.dumps(
            {
                "rows": rows,
                "eval_seconds": elapsed,
                "target_fpr": args.target_fpr,
                "sampled_test_windows": int(len(y_test)),
                "sampled_anomaly_ratio": float(np.asarray(y_test).mean()) if len(y_test) else 0.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"rows": rows, "eval_seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()
