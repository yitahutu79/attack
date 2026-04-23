#!/usr/bin/env python3
"""Minimal TCN-GAN runner for cross-dataset smoke validation.

This intentionally avoids the large experiment CLI so the second-dataset
bring-up path stays simple and robust.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TCN_GAN_DISABLE_WEIGHT_NORM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ATTACK = ROOT / "attack"
if str(ATTACK) not in sys.path:
    sys.path.insert(0, str(ATTACK))

from attack.dataset_loaders import (  # noqa: E402
    load_windowed_chrono_unsupervised_split,
    load_windowed_dataset,
    load_windowed_mixed_split,
    load_windowed_unsupervised_split,
)
from models.tcn_gan_experiment import (  # noqa: E402
    TCNDiscriminator,
    TCNGenerator,
    SequenceDataset,
    compute_anomaly_scores,
    metrics_at_threshold,
    set_seed,
    threshold_from_benign_fpr,
    train_one_epoch,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal cross-dataset TCN-GAN runner")
    ap.add_argument("--dataset", default="swat", choices=["swat", "ton_iot", "unsw_nb15", "generic"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-files", nargs="*", default=[])
    ap.add_argument("--test-files", nargs="*", default=[])
    ap.add_argument("--supervised-mixed-split", action="store_true")
    ap.add_argument("--mixed-files", nargs="*", default=[])
    ap.add_argument("--mixed-train-fraction", type=float, default=0.5)
    ap.add_argument("--unsupervised-formal-split", action="store_true")
    ap.add_argument("--unsup-benign-file", default="")
    ap.add_argument("--unsup-attack-file", default="")
    ap.add_argument("--unsup-train-fraction", type=float, default=0.6)
    ap.add_argument("--unsup-calib-fraction", type=float, default=0.2)
    ap.add_argument("--chrono-unsupervised-split", action="store_true")
    ap.add_argument("--chrono-file", default="")
    ap.add_argument("--chrono-train-fraction", type=float, default=0.6)
    ap.add_argument("--chrono-calib-fraction", type=float, default=0.1)
    ap.add_argument("--window-size", type=int, default=128)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--anomaly-ratio", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--test-batch-size", type=int, default=256)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--hidden-channels", nargs="+", type=int, default=[128, 128])
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--disc-pooling", choices=["attn", "mean"], default="attn")
    ap.add_argument("--score-mode", choices=["prob", "fused", "feat_l2", "feat_mahal"], default="prob")
    ap.add_argument("--score-alpha", type=float, default=0.24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--save-best", default="")
    args = ap.parse_args()

    t0 = time.perf_counter()
    set_seed(int(args.seed))
    torch.set_num_threads(1)
    device = torch.device("cpu")

    if args.unsupervised_formal_split:
        ds = load_windowed_unsupervised_split(
            dataset=args.dataset,
            data_dir=args.data_dir,
            benign_file=args.unsup_benign_file or None,
            attack_file=args.unsup_attack_file or None,
            window_size=args.window_size,
            stride=args.stride,
            anomaly_ratio=args.anomaly_ratio,
            benign_train_fraction=args.unsup_train_fraction,
            benign_calib_fraction=args.unsup_calib_fraction,
            scaler="minmax",
            clip_minmax=True,
            progress=True,
        )
        x_train_benign = np.asarray(ds.x_train)
        x_calib_benign = np.asarray(ds.x_calib) if ds.x_calib is not None else None
    elif args.chrono_unsupervised_split:
        ds = load_windowed_chrono_unsupervised_split(
            dataset=args.dataset,
            data_dir=args.data_dir,
            mixed_file=args.chrono_file or None,
            window_size=args.window_size,
            stride=args.stride,
            anomaly_ratio=args.anomaly_ratio,
            train_fraction=args.chrono_train_fraction,
            calib_fraction=args.chrono_calib_fraction,
            scaler="minmax",
            clip_minmax=True,
            progress=True,
        )
        x_train_benign = np.asarray(ds.x_train)
        x_calib_benign = np.asarray(ds.x_calib) if ds.x_calib is not None else None
    elif args.supervised_mixed_split:
        mixed = load_windowed_mixed_split(
            dataset=args.dataset,
            data_dir=args.data_dir,
            files=args.mixed_files,
            window_size=args.window_size,
            stride=args.stride,
            anomaly_ratio=args.anomaly_ratio,
            train_fraction=args.mixed_train_fraction,
            scaler="standard",
            clip_minmax=False,
            progress=True,
        )
        benign_mask = np.asarray(mixed.y_train) == 0
        if not np.any(benign_mask):
            raise ValueError("mixed split 训练集里没有 benign 窗口，无法训练 TCN 异常检测器")
        ds = mixed
        x_train_benign = np.asarray(mixed.x_train)[benign_mask]
        x_calib_benign = None
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
            scaler="minmax",
            clip_minmax=True,
            progress=True,
        )
        x_train_benign = np.asarray(ds.x_train)
        x_calib_benign = None

    print(
        f"Loaded {args.dataset}: train_windows={len(ds.x_train)}, "
        f"test_windows={len(ds.x_test)}, test_anom_ratio={float(ds.y_test.mean()):.4f}, "
        f"features={len(ds.feature_names)}",
        flush=True,
    )
    if args.supervised_mixed_split:
        print(
            f"Using mixed split: train_anom_ratio={float(ds.y_train.mean()):.4f}, "
            f"benign_train_windows={len(x_train_benign)}, train_fraction={float(args.mixed_train_fraction):.2f}",
            flush=True,
        )
    elif args.unsupervised_formal_split:
        print(
            f"Using unsupervised formal split: benign_train_windows={len(x_train_benign)}, "
            f"benign_calib_windows={0 if x_calib_benign is None else len(x_calib_benign)}, "
            f"train_fraction={float(args.unsup_train_fraction):.2f}, calib_fraction={float(args.unsup_calib_fraction):.2f}",
            flush=True,
        )
    elif args.chrono_unsupervised_split:
        print(
            f"Using chrono unsupervised split: benign_train_windows={len(x_train_benign)}, "
            f"benign_calib_windows={0 if x_calib_benign is None else len(x_calib_benign)}, "
            f"train_fraction={float(args.chrono_train_fraction):.2f}, calib_fraction={float(args.chrono_calib_fraction):.2f}",
            flush=True,
        )

    train_loader = DataLoader(
        SequenceDataset(x_train_benign), batch_size=args.batch_size, shuffle=True, drop_last=False
    )

    generator = TCNGenerator(
        args.window_size, x_train_benign.shape[2], args.latent_dim, args.hidden_channels, args.dropout
    ).to(device)
    discriminator = TCNDiscriminator(
        args.window_size, x_train_benign.shape[2], args.hidden_channels, args.dropout, pooling=args.disc_pooling
    ).to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        e0 = time.perf_counter()
        d_loss, g_loss = train_one_epoch(
            generator,
            discriminator,
            train_loader,
            device,
            g_optimizer,
            d_optimizer,
            criterion,
            args.latent_dim,
            gan_loss="vanilla",
            progress=True,
            desc=f"train ep{epoch}/{args.epochs}",
        )
        seconds = time.perf_counter() - e0
        history.append({"epoch": float(epoch), "d_loss": float(d_loss), "g_loss": float(g_loss), "seconds": seconds})
        print(
            f"epoch {epoch}/{args.epochs}: d_loss={d_loss:.4f}, g_loss={g_loss:.4f}, seconds={seconds:.2f}",
            flush=True,
        )

    y_true = np.asarray(ds.y_test, dtype=np.int64)
    y_score = compute_anomaly_scores(
        discriminator,
        ds.x_test,
        device,
        args.test_batch_size,
        score_mode=args.score_mode,
        score_alpha=args.score_alpha,
        x_ref_benign=x_train_benign if str(args.score_mode).lower().strip() != "prob" else None,
    )
    calib_source = x_calib_benign if x_calib_benign is not None and len(x_calib_benign) else x_train_benign
    train_scores = compute_anomaly_scores(
        discriminator,
        calib_source,
        device,
        args.test_batch_size,
        score_mode=args.score_mode,
        score_alpha=args.score_alpha,
        x_ref_benign=x_train_benign if str(args.score_mode).lower().strip() != "prob" else None,
    )
    threshold = threshold_from_benign_fpr(train_scores, args.target_fpr)
    calibrated = metrics_at_threshold(y_true, y_score, threshold)
    train_benign_fpr = float(np.mean(train_scores >= threshold))

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        auc = float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size > 1 else None
        ap_score = float(average_precision_score(y_true, y_score)) if float(y_true.sum()) > 0 else 0.0
    except Exception:
        auc = None
        ap_score = 0.0

    payload = {
        "task": "cross_dataset_minimal_tcn",
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "train_files": ds.train_files,
        "test_files": ds.test_files,
        "calib_files": ds.calib_files,
        "supervised_mixed_split": bool(args.supervised_mixed_split),
        "mixed_train_fraction": float(args.mixed_train_fraction) if args.supervised_mixed_split else None,
        "unsupervised_formal_split": bool(args.unsupervised_formal_split),
        "unsup_train_fraction": float(args.unsup_train_fraction) if args.unsupervised_formal_split else None,
        "unsup_calib_fraction": float(args.unsup_calib_fraction) if args.unsupervised_formal_split else None,
        "chrono_unsupervised_split": bool(args.chrono_unsupervised_split),
        "chrono_train_fraction": float(args.chrono_train_fraction) if args.chrono_unsupervised_split else None,
        "chrono_calib_fraction": float(args.chrono_calib_fraction) if args.chrono_unsupervised_split else None,
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "anomaly_ratio": float(args.anomaly_ratio),
        "feature_dim": int(x_train_benign.shape[2]),
        "n_train_windows": int(len(x_train_benign)),
        "n_train_windows_raw": int(len(ds.x_train)),
        "n_calib_windows": 0 if x_calib_benign is None else int(len(x_calib_benign)),
        "n_test_windows": int(len(ds.x_test)),
        "train_anomaly_ratio": float(ds.y_train.mean()) if len(ds.y_train) else 0.0,
        "test_anomaly_ratio": float(y_true.mean()) if len(y_true) else 0.0,
        "model": {
            "disc_pooling": str(args.disc_pooling),
            "gan_loss": "vanilla",
            "weight_norm_disabled": True,
        },
        "score_mode": str(args.score_mode),
        "score_alpha": float(args.score_alpha),
        "history": history,
        "metrics": {
            "auc": auc,
            "ap": ap_score,
        },
        "calibrated": {
            **calibrated,
            "target_fpr": float(args.target_fpr),
            "train_benign_fpr": train_benign_fpr,
            "test_benign_fpr": None if int((y_true == 0).sum()) == 0 else float(calibrated["fpr"]),
        },
        "timing": {
            "total_seconds": float(time.perf_counter() - t0),
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.save_best:
        torch.save(
            {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "args": vars(args),
                "payload": payload,
            },
            args.save_best,
        )
    print(f"Saved JSON: {out_json}", flush=True)
    print(
        "Calibrated: F1={f1:.4f}, recall={recall:.4f}, precision={precision:.4f}, train_FPR={train_benign_fpr:.4f}".format(
            **payload["calibrated"]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
