#!/usr/bin/env python3
"""Compute top-feature stability over anomalous windows."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="dataset/CICIDS2017")
    p.add_argument("--load", required=True)
    p.add_argument("--out-csv", default="paper/tables/xai_stability_top_features.csv")
    p.add_argument("--out-md", default="paper/tables/xai_stability_top_features.md")
    p.add_argument("--n-windows", type=int, default=64)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--score-mode", default="fused")
    p.add_argument("--score-alpha", type=float, default=0.24)
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
        _anomaly_score_torch,
        _prepare_ref_stats,
    )

    ckpt = torch.load(args.load, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    train_files = ckpt_args.get(
        "train_files",
        [
            "Tuesday-WorkingHours.pcap_ISCX.csv",
            "Wednesday-workingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        ],
    )
    test_files = ckpt_args.get(
        "test_files",
        [
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        ],
    )
    ds = load_windowed_dataset(
        dataset="cicids2017",
        data_dir=args.data_dir,
        train_files=train_files,
        test_files=test_files,
        window_size=int(ckpt_args.get("window_size", 128)),
        stride=int(ckpt_args.get("stride", 16)),
        anomaly_ratio=float(ckpt_args.get("anomaly_ratio", 0.15)),
        train_benign_only=True,
        progress=False,
    )

    disc = TCNDiscriminator(
        seq_len=int(ckpt_args.get("window_size", 128)),
        feat_dim=int(ds.x_train.shape[2]),
        hidden_channels=list(ckpt_args.get("hidden_channels", [128, 128])),
        dropout=float(ckpt_args.get("dropout", 0.2)),
        pooling=str(ckpt_args.get("disc_pooling", "attn")),
    )
    disc.load_state_dict(ckpt["discriminator"])
    device = torch.device("cpu")
    disc.to(device).eval()

    rng = np.random.default_rng(args.seed)
    idx = np.where(np.asarray(ds.y_test).reshape(-1) == 1)[0]
    idx = rng.choice(idx, size=min(int(args.n_windows), len(idx)), replace=False)
    x = np.asarray(ds.x_test[idx], dtype=np.float32)
    ref_stats = _prepare_ref_stats(
        disc,
        ds.x_train[: min(len(ds.x_train), 20000)],
        device,
        batch_size=256,
        score_mode=args.score_mode,
    )

    top_k = int(args.top_k)
    counts = np.zeros(len(ds.feature_names), dtype=np.int64)
    rank_sum = np.zeros(len(ds.feature_names), dtype=np.float64)
    attr_sum = np.zeros(len(ds.feature_names), dtype=np.float64)
    n = 0
    for start in range(0, len(x), int(args.batch_size)):
        xb = torch.from_numpy(x[start:start + int(args.batch_size)]).to(device).float()
        xb.requires_grad_(True)
        disc.zero_grad(set_to_none=True)
        scores = _anomaly_score_torch(
            disc,
            xb,
            score_mode=args.score_mode,
            score_alpha=float(args.score_alpha),
            ref_stats=ref_stats,
        )
        scores.mean().backward()
        if xb.grad is None:
            continue
        attrs = (xb.grad * xb).abs().sum(dim=1).detach().cpu().numpy()
        for row in attrs:
            n += 1
            order = np.argsort(-row)
            attr_sum += row
            for rank, feat_idx in enumerate(order[:top_k], start=1):
                counts[int(feat_idx)] += 1
                rank_sum[int(feat_idx)] += rank

    rows = []
    for i, name in enumerate(ds.feature_names):
        if counts[i] <= 0:
            continue
        rows.append(
            {
                "feature": str(name),
                "windows_in_topk": int(counts[i]),
                "coverage": float(counts[i] / max(n, 1)),
                "mean_rank_when_topk": float(rank_sum[i] / max(counts[i], 1)),
                "mean_abs_attr": float(attr_sum[i] / max(n, 1)),
            }
        )
    rows.sort(key=lambda r: (-float(r["coverage"]), float(r["mean_rank_when_topk"])))
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "windows_in_topk", "coverage", "mean_rank_when_topk", "mean_abs_attr"])
        w.writeheader()
        w.writerows(rows[:10])

    out_md = Path(args.out_md)
    lines = [
        f"# XAI Top-Feature Stability (N={n}, top-k={top_k})",
        "",
        "| Feature | Windows in top-k | Coverage | Mean rank | Mean abs attr |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows[:10]:
        lines.append(
            "| {feature} | {windows_in_topk} | {coverage:.3f} | {mean_rank_when_topk:.2f} | {mean_abs_attr:.3e} |".format(**r)
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
