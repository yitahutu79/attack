#!/usr/bin/env python3
"""
Fine-grained alpha sweep for TCN-GAN fused score WITHOUT retraining.

Alpha only affects scoring:
  fused = alpha * norm(prob) + (1-alpha) * norm(feature_dev)

This script:
  - loads an existing checkpoint
  - rebuilds train/test windows (same as tcn_gan_experiment.py)
  - precomputes embeddings once
  - evaluates many alphas efficiently
  - outputs CSV + plot
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

# Make `attack/` importable when running as `python attack/pipelines/xxx.py`
_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT_S = str(_REPO_ROOT)
if _REPO_ROOT_S not in sys.path:
    sys.path.insert(0, _REPO_ROOT_S)

# Reuse core utilities from the main script for consistent preprocessing/windowing.
from attack.models.tcn_gan_experiment import (  # type: ignore
    TCNDiscriminator,
    _minmax_norm,
    _sample_rows,
    build_sequences,
    embed_sequences,
    feature_deviation_scores,
    fit_scaler_from_train,
    preprocess_raw_frame,
    threshold_from_benign_fpr,
)

from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, th: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float32)
    y_pred = (y_score >= float(th)).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    eps = 1e-12
    prec = float(tp / (tp + fp + eps))
    rec = float(tp / (tp + fn + eps))
    f1 = float(2 * prec * rec / (prec + rec + eps))
    fpr = float(fp / (fp + tn + eps))
    return {"threshold": float(th), "precision": prec, "recall": rec, "f1": f1, "fpr": fpr}


def _benign_fpr(y_score_benign: np.ndarray, th: float) -> float:
    y = np.asarray(y_score_benign, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return float("nan")
    return float((y >= float(th)).mean())


@dataclass
class Precomp:
    y_true: np.ndarray
    # normalized components (0..1)
    unk_train_n: np.ndarray
    dev_train_n: np.ndarray
    unk_test_n: np.ndarray
    dev_test_n: np.ndarray


def _prepare_precomp(
    *,
    discriminator: TCNDiscriminator,
    device,
    train_seqs: np.ndarray,
    test_seqs: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int,
    feat_ref_max: int,
    calib_max: int,
    seed: int,
    dev_method: str,
) -> Precomp:
    # Reference for normalization
    x_ref = _sample_rows(train_seqs, int(feat_ref_max), int(seed) + 11)
    emb_ref, unk_ref = embed_sequences(discriminator, x_ref, device, batch_size)
    if emb_ref.size == 0:
        raise ValueError("empty emb_ref (check train_seqs)")
    dev_ref = feature_deviation_scores(emb_ref, emb_ref, method=dev_method)

    # Calibration benign windows (subset of train benign windows)
    calib = _sample_rows(train_seqs, int(calib_max), int(seed))
    emb_train, unk_train = embed_sequences(discriminator, calib, device, batch_size)
    dev_train = feature_deviation_scores(emb_train, emb_ref, method=dev_method)

    # Test windows
    emb_test, unk_test = embed_sequences(discriminator, test_seqs, device, batch_size)
    dev_test = feature_deviation_scores(emb_test, emb_ref, method=dev_method)

    dev_train_n = _minmax_norm(dev_train, float(np.min(dev_ref)), float(np.max(dev_ref)))
    unk_train_n = _minmax_norm(unk_train, float(np.min(unk_ref)), float(np.max(unk_ref)))
    dev_test_n = _minmax_norm(dev_test, float(np.min(dev_ref)), float(np.max(dev_ref)))
    unk_test_n = _minmax_norm(unk_test, float(np.min(unk_ref)), float(np.max(unk_ref)))

    return Precomp(
        y_true=np.asarray(test_labels, dtype=np.int64),
        unk_train_n=unk_train_n,
        dev_train_n=dev_train_n,
        unk_test_n=unk_test_n,
        dev_test_n=dev_test_n,
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "alpha",
        "auc",
        "ap",
        "calib_threshold",
        "calib_precision",
        "calib_recall",
        "calib_f1",
        "calib_fpr",
        "test_benign_fpr",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])


def _plot(out_png: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception:
        return

    xs = [float(r["alpha"]) for r in rows]
    f1 = [float(r["calib_f1"]) for r in rows]
    rec = [float(r["calib_recall"]) for r in rows]
    fpr = [float(r["test_benign_fpr"]) for r in rows]

    fig = plt.figure(figsize=(10.5, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax2 = ax.twinx()

    ax.plot(xs, f1, marker="o", label="calib_f1 (target_fpr)")
    ax.plot(xs, rec, marker="o", label="calib_recall (target_fpr)")
    ax2.plot(xs, fpr, marker="s", color="tab:red", label="test_benign_fpr")

    ax.set_xlabel("alpha")
    ax.set_ylabel("F1 / Recall")
    ax2.set_ylabel("Test BENIGN FPR")
    ax.set_ylim(0, 1.02)
    ax2.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Alpha sweep for TCN-GAN fused score (no retraining).")
    ap.add_argument("--data-dir", default="attack/dataset/CICIDS2017")
    ap.add_argument("--train-files", nargs="+", required=True)
    ap.add_argument("--test-files", nargs="+", required=True)
    ap.add_argument("--load", required=True, help="checkpoint .pt (with discriminator weights)")
    ap.add_argument("--window-size", type=int, required=True)
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--anomaly-ratio", type=float, default=0.15)

    ap.add_argument("--disc-pooling", choices=["mean", "attn"], default="mean")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--dev-method", choices=["l2", "mahal"], default="l2")
    ap.add_argument("--feat-ref-max", type=int, default=20000)
    ap.add_argument("--calib-max", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--alpha-step", type=float, default=0.02, help="grid step in [0,1]")
    ap.add_argument("--alpha-min", type=float, default=0.0)
    ap.add_argument("--alpha-max", type=float, default=1.0)

    ap.add_argument("--out-dir", default="", help="default: attack/results/final_experiments/manual_alpha_sweeps/<timestamp>/")
    args = ap.parse_args()

    # device
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load data (same logic as tcn_gan_experiment day-split mode)
    from attack.models.tcn_gan_experiment import load_cicids_files  # type: ignore

    files = load_cicids_files(str(args.data_dir))
    train_frames = []
    train_labels = []
    test_frames = []
    test_labels = []
    for n in args.train_files:
        numeric, y = preprocess_raw_frame(files[str(n)])
        train_frames.append(numeric)
        train_labels.append(y)
    for n in args.test_files:
        numeric, y = preprocess_raw_frame(files[str(n)])
        test_frames.append(numeric)
        test_labels.append(y)

    scaler = fit_scaler_from_train(train_frames, train_labels, benign_only=True)
    train_seq_parts = []
    for numeric, y in zip(train_frames, train_labels, strict=True):
        x = scaler.transform(numeric.astype(np.float32)).astype(np.float32)
        x = np.clip(x, 0.0, 1.0)
        seqs, seq_y = build_sequences(x, y, int(args.window_size), int(args.stride), float(args.anomaly_ratio))
        keep = seq_y == 0
        seqs = seqs[keep]
        if len(seqs):
            train_seq_parts.append(seqs)
    if not train_seq_parts:
        raise SystemExit("train_seqs empty (check window/stride/anomaly_ratio)")
    train_seqs = np.concatenate(train_seq_parts, axis=0)

    test_seq_parts = []
    test_y_parts = []
    for numeric, y in zip(test_frames, test_labels, strict=True):
        x = scaler.transform(numeric.astype(np.float32)).astype(np.float32)
        x = np.clip(x, 0.0, 1.0)
        seqs, seq_y = build_sequences(x, y, int(args.window_size), int(args.stride), float(args.anomaly_ratio))
        test_seq_parts.append(seqs)
        test_y_parts.append(seq_y.astype(np.uint8))
    test_seqs = np.concatenate(test_seq_parts, axis=0)
    y_true = np.concatenate(test_y_parts, axis=0).astype(np.int64)

    # Build discriminator from checkpoint args
    ckpt = torch.load(str(args.load), map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    hidden_channels = list(ckpt_args.get("hidden_channels", [128, 128]))
    dropout = float(ckpt_args.get("dropout", 0.2))
    disc_pooling = str(ckpt_args.get("disc_pooling", args.disc_pooling))
    if disc_pooling not in ("mean", "attn"):
        disc_pooling = str(args.disc_pooling)

    feat_dim = int(train_seqs.shape[2])
    disc = TCNDiscriminator(int(args.window_size), feat_dim, hidden_channels, dropout, pooling=disc_pooling).to(device)

    if isinstance(ckpt, dict) and "discriminator" in ckpt:
        disc.load_state_dict(ckpt["discriminator"])
    else:
        disc.load_state_dict(ckpt)
    disc.eval()

    pre = _prepare_precomp(
        discriminator=disc,
        device=device,
        train_seqs=train_seqs,
        test_seqs=test_seqs,
        test_labels=y_true,
        batch_size=int(args.batch_size),
        feat_ref_max=int(args.feat_ref_max),
        calib_max=int(args.calib_max),
        seed=int(args.seed),
        dev_method=str(args.dev_method),
    )

    # alpha grid
    a0 = float(args.alpha_min)
    a1 = float(args.alpha_max)
    step = float(args.alpha_step)
    if not (0.0 <= a0 <= 1.0 and 0.0 <= a1 <= 1.0 and step > 0):
        raise SystemExit("alpha params invalid")
    if a1 < a0:
        a0, a1 = a1, a0

    grid = []
    x = a0
    while x <= a1 + 1e-9:
        grid.append(round(min(max(x, 0.0), 1.0), 6))
        x += step
    # ensure endpoints
    if grid and grid[0] != round(a0, 6):
        grid.insert(0, round(a0, 6))
    if grid and grid[-1] != round(a1, 6):
        grid.append(round(a1, 6))

    rows: list[dict[str, object]] = []
    for a in grid:
        fused_train = float(a) * pre.unk_train_n + (1.0 - float(a)) * pre.dev_train_n
        fused_test = float(a) * pre.unk_test_n + (1.0 - float(a)) * pre.dev_test_n

        th = threshold_from_benign_fpr(fused_train, float(args.target_fpr))
        m = _metrics_at_threshold(pre.y_true, fused_test, th)

        # overall AUC/AP for reference
        if np.unique(pre.y_true).size < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(pre.y_true, fused_test))
        apv = float(average_precision_score(pre.y_true, fused_test)) if float(pre.y_true.sum()) > 0 else 0.0
        test_benign_fpr = _benign_fpr(fused_test[pre.y_true == 0], th)

        rows.append(
            {
                "alpha": float(a),
                "auc": auc,
                "ap": apv,
                "calib_threshold": float(th),
                "calib_precision": float(m["precision"]),
                "calib_recall": float(m["recall"]),
                "calib_f1": float(m["f1"]),
                "calib_fpr": float(m["fpr"]),
                "test_benign_fpr": float(test_benign_fpr),
            }
        )

    # choose best alpha by calib_f1
    best = max(rows, key=lambda r: float(r.get("calib_f1", float("-inf"))))

    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else Path("attack/results/final_experiments/manual_alpha_sweeps") / _now_tag()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "alpha_sweep.csv"
    png_path = out_dir / "alpha_sweep.png"
    meta_path = out_dir / "meta.json"

    _write_csv(csv_path, rows)
    _plot(png_path, rows)

    meta = {
        "best_alpha": float(best["alpha"]),
        "best_calib_f1": float(best["calib_f1"]),
        "best_calib_recall": float(best["calib_recall"]),
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "disc_pooling": str(disc_pooling),
        "dev_method": str(args.dev_method),
        "target_fpr": float(args.target_fpr),
        "checkpoint": str(args.load),
        "csv": str(csv_path),
        "plot": str(png_path) if png_path.exists() else "",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote: {csv_path}")
    if png_path.exists():
        print(f"Wrote: {png_path}")
    print(f"Wrote: {meta_path}")
    print(f"Best alpha={meta['best_alpha']:.3f} calib_f1={meta['best_calib_f1']:.4f} calib_rec={meta['best_calib_recall']:.4f}")


if __name__ == "__main__":
    main()
