#!/usr/bin/env python3
"""
Robustness tests for the A-tier revision plan.
Tests:
1. Gaussian noise on numeric features (1%, 3%, 5%).
2. Missing features (random mask 5%, 10%, 20%).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ATTACK = ROOT / "attack"
if str(ATTACK) not in sys.path:
    sys.path.insert(0, str(ATTACK))

from attack.dataset_loaders import load_windowed_dataset
from attack.models.tcn_gan_experiment import (
    TCNDiscriminator,
    compute_anomaly_scores,
    metrics_at_threshold,
    threshold_from_benign_fpr,
    _sample_rows
)

def add_gaussian_noise(x, sigma_ratio=0.01):
    """Add Gaussian noise relative to the feature range (assuming [0,1] scaling)."""
    noise = np.random.normal(0, sigma_ratio, x.shape).astype(np.float32)
    return np.clip(x + noise, 0.0, 1.0)

def mask_features(x, mask_ratio=0.05):
    """Randomly mask features with 0."""
    mask = np.random.choice([0, 1], size=x.shape, p=[mask_ratio, 1 - mask_ratio]).astype(np.float32)
    return x * mask

def main():
    parser = argparse.ArgumentParser(description="Robustness tests for TCN-GAN")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--train-files", nargs="+", required=True)
    parser.add_argument("--test-files", nargs="+", required=True)
    parser.add_argument("--load", required=True)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--anomaly-ratio", type=float, default=0.15)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--score-mode", default="fused")
    parser.add_argument("--score-alpha", type=float, default=0.24)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-log", help="Path to save execution log")
    args = parser.parse_args()

    if args.out_log:
        log_path = Path(args.out_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        class Logger(object):
            def __init__(self, filename):
                self.terminal = sys.stdout
                self.log = open(filename, "a", encoding='utf-8')
            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)
                self.log.flush()
            def flush(self):
                self.terminal.flush()
                self.log.flush()
        sys.stdout = Logger(args.out_log)
        print(f"Logging to {args.out_log}")
        print(f"Run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # 1. Load Data
    print("Loading data...")
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
        progress=True
    )

    # 2. Load Model
    print(f"Loading model from {args.load}...")
    ckpt = torch.load(args.load, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    hidden_channels = ckpt_args.get("hidden_channels", [128, 128])
    dropout = float(ckpt_args.get("dropout", 0.2))
    pooling = str(ckpt_args.get("disc_pooling", "attn"))
    
    disc = TCNDiscriminator(args.window_size, ds.x_train.shape[2], hidden_channels, dropout, pooling=pooling).to(device)
    disc.load_state_dict(ckpt["discriminator"])
    disc.eval()

    # 3. Calibrate Threshold on Clean Train Benign
    print("Calibrating threshold...")
    calib_seqs = _sample_rows(ds.x_train, 20000, 42)
    x_ref = _sample_rows(ds.x_train, 10000, 43)
    
    benign_scores = compute_anomaly_scores(
        disc, calib_seqs, device, batch_size=256,
        score_mode=args.score_mode, score_alpha=args.score_alpha, x_ref_benign=x_ref
    )
    threshold = threshold_from_benign_fpr(benign_scores, args.target_fpr)
    print(f"Calibrated threshold: {threshold:.6f} (target FPR: {args.target_fpr})")

    # 4. Run robustness tests
    test_scenarios = [
        ("Clean", lambda x: x),
        ("Noise 1%", lambda x: add_gaussian_noise(x, 0.01)),
        ("Noise 3%", lambda x: add_gaussian_noise(x, 0.03)),
        ("Noise 5%", lambda x: add_gaussian_noise(x, 0.05)),
        ("Mask 5%", lambda x: mask_features(x, 0.05)),
        ("Mask 10%", lambda x: mask_features(x, 0.10)),
        ("Mask 20%", lambda x: mask_features(x, 0.20)),
    ]

    results = []
    for name, perturb_fn in test_scenarios:
        print(f"\nRunning scenario: {name}")
        x_test_perturbed = perturb_fn(ds.x_test)
        
        y_score = compute_anomaly_scores(
            disc, x_test_perturbed, device, batch_size=256,
            score_mode=args.score_mode, score_alpha=args.score_alpha, x_ref_benign=x_ref
        )
        
        m = metrics_at_threshold(ds.y_test, y_score, threshold)
        print(f"F1: {m['f1']:.4f}, Recall: {m['recall']:.4f}, FPR: {m['fpr']:.4f}")
        
        res = {
            "scenario": name,
            "f1": m["f1"],
            "recall": m["recall"],
            "fpr": m["fpr"],
            "precision": m["precision"]
        }
        results.append(res)

    # 5. Save results
    out_path = os.path.join(args.out_dir, "robustness_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
