#!/usr/bin/env python3
"""Generate IG + SHAP + attention case study for a saved TCN checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ATTACK = ROOT / "attack"
if str(ATTACK) not in sys.path:
    sys.path.insert(0, str(ATTACK))

try:
    import shap  # type: ignore
except Exception:
    shap = None

from attack.dataset_loaders import load_windowed_dataset
from attack.models.tcn_gan_experiment import (
    TCNDiscriminator,
    _anomaly_score_torch,
    _prepare_ref_stats,
)


def integrated_gradients(
    model: TCNDiscriminator,
    x: torch.Tensor,
    *,
    baseline: torch.Tensor,
    score_mode: str,
    score_alpha: float,
    ref_stats: dict[str, object] | None,
    steps: int = 32,
) -> np.ndarray:
    grads = []
    for a in torch.linspace(0.0, 1.0, steps, device=x.device):
        xi = baseline + a * (x - baseline)
        xi.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        score = _anomaly_score_torch(model, xi, score_mode=score_mode, score_alpha=score_alpha, ref_stats=ref_stats).sum()
        score.backward()
        grads.append(xi.grad.detach().cpu().numpy())
    avg_grad = np.mean(np.stack(grads, axis=0), axis=0)
    return (x.detach().cpu().numpy() - baseline.detach().cpu().numpy()) * avg_grad


def _as_case_array(values: object) -> np.ndarray:
    """Normalize SHAP output to (n_cases, window, features)."""
    if isinstance(values, list):
        if not values:
            return np.empty((0, 0, 0), dtype=np.float32)
        values = values[0]
    arr = np.asarray(values)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        return np.empty((0, 0, 0), dtype=np.float32)
    return arr.astype(np.float32)


def _safe_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    vmax = float(np.nanmax(np.abs(values))) if values.size else 0.0
    if vmax <= 1e-12:
        return np.zeros_like(values)
    return values / vmax


def _top_feature_rows(
    feature_names: list[str],
    ig_feature: np.ndarray,
    shap_feature: np.ndarray | None,
    *,
    top_k: int = 8,
) -> list[dict[str, object]]:
    if shap_feature is not None and shap_feature.size == len(ig_feature):
        rank_score = _safe_norm(ig_feature) + _safe_norm(shap_feature)
    else:
        rank_score = np.asarray(ig_feature, dtype=np.float32)
    order = np.argsort(-rank_score)[:top_k]
    rows: list[dict[str, object]] = []
    for rank, idx in enumerate(order, start=1):
        rows.append(
            {
                "rank": rank,
                "feature": feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
                "ig_abs_mean": float(ig_feature[idx]),
                "shap_abs_mean": float(shap_feature[idx]) if shap_feature is not None and idx < len(shap_feature) else "",
            }
        )
    return rows


def _plot_case_study(
    out_path: Path,
    *,
    anomaly_score: float,
    benign_score: float,
    ig_heatmap: np.ndarray,
    attention: np.ndarray,
    ig_time: np.ndarray,
    shap_time: np.ndarray | None,
    top_rows: list[dict[str, object]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)

    ax = axes[0, 0]
    ax.bar(["Benign case", "Anomaly case"], [benign_score, anomaly_score], color=["#6699cc", "#cc5a49"])
    ax.set_ylabel("Anomaly score")
    ax.set_title("Single-window score contrast")
    ax.set_ylim(0, max(1.05, anomaly_score * 1.1))
    for i, value in enumerate([benign_score, anomaly_score]):
        ax.text(i, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    ax = axes[0, 1]
    shown = np.abs(ig_heatmap)
    im = ax.imshow(shown.T, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Feature index")
    ax.set_title("Integrated Gradients time-feature attribution")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    x = np.arange(len(ig_time))
    ax.plot(x, _safe_norm(ig_time), label="IG time attribution", linewidth=2.0, color="#cc5a49")
    if len(attention):
        ax.plot(x, _safe_norm(attention), label="Attention weight", linewidth=1.8, color="#3d7f5f")
    if shap_time is not None and len(shap_time) == len(ig_time):
        ax.plot(x, _safe_norm(shap_time), label="SHAP time attribution", linewidth=1.8, color="#4d5db8")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Normalized evidence")
    ax.set_title("Attention vs post-hoc attribution")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    labels = [str(row["feature"]) for row in reversed(top_rows)]
    ig_vals = [float(row["ig_abs_mean"]) for row in reversed(top_rows)]
    y = np.arange(len(labels))
    ax.barh(y, ig_vals, color="#cc5a49", alpha=0.85, label="IG")
    shap_vals = [row["shap_abs_mean"] for row in reversed(top_rows)]
    if all(v != "" for v in shap_vals):
        ax.barh(y, [float(v) for v in shap_vals], color="#4d5db8", alpha=0.55, label="SHAP")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean absolute attribution")
    ax.set_title("Top contributing features")
    ax.legend(fontsize=8)

    fig.suptitle("Multi-view XAI case study: score, attribution, attention, and features", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-files", nargs="+", required=True)
    ap.add_argument("--test-files", nargs="+", required=True)
    ap.add_argument("--load", required=True)
    ap.add_argument("--window-size", type=int, default=128)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--anomaly-ratio", type=float, default=0.15)
    ap.add_argument("--score-mode", default="fused")
    ap.add_argument("--score-alpha", type=float, default=0.24)
    ap.add_argument("--case-index", type=int, default=-1, help="anomaly sample index in test anomalies; -1 means highest-score anomaly")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-log", help="Path to save execution log")
    ap.add_argument("--no-shap", action="store_true", help="Skip SHAP and generate the case figure from IG + attention only")
    ap.add_argument("--shap-background", type=int, default=64, help="Number of benign background windows for SHAP")
    ap.add_argument("--shap-benign-case", action="store_true", help="Also explain the benign reference case with SHAP")
    args = ap.parse_args()

    if args.out_log:
        log_path = Path(args.out_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Simple tee implementation
        class Logger(object):
            def __init__(self, filename):
                self.terminal = sys.stdout
                self.log = open(filename, "a")

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

    print(f"Loading dataset 'cicids2017' from {args.data_dir}...")
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
        progress=True,
    )

    print(f"Loading checkpoint from {args.load}...")
    ckpt = torch.load(args.load, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    hidden_channels = ckpt_args.get("hidden_channels", [128, 128])
    dropout = float(ckpt_args.get("dropout", 0.2))
    pooling = str(ckpt_args.get("disc_pooling", "attn"))
    disc = TCNDiscriminator(args.window_size, ds.x_train.shape[2], hidden_channels, dropout, pooling=pooling).cpu()
    disc.load_state_dict(ckpt["discriminator"])
    disc.eval()

    x_ref = np.asarray(ds.x_train[: min(len(ds.x_train), 1024)], dtype=np.float32)
    ref_stats = _prepare_ref_stats(disc, x_ref, torch.device("cpu"), 256, args.score_mode) if args.score_mode != "prob" else None

    print("Computing anomaly scores for the test set...")
    x_test = np.asarray(ds.x_test, dtype=np.float32)
    y_test = np.asarray(ds.y_test, dtype=np.int64)
    scores = []
    with torch.no_grad():
        for i in range(0, len(x_test), 256):
            xb = torch.from_numpy(x_test[i:i+256])
            s = _anomaly_score_torch(disc, xb, score_mode=args.score_mode, score_alpha=args.score_alpha, ref_stats=ref_stats)
            scores.append(s.cpu().numpy())
            if (i // 256) % 10 == 0:
                print(f"  Processed {i}/{len(x_test)} windows...")
    scores = np.concatenate(scores)

    anom_idx = np.where(y_test == 1)[0]
    benign_idx = np.where(y_test == 0)[0]
    if len(anom_idx) == 0 or len(benign_idx) == 0:
        raise ValueError("test set needs both benign and anomaly windows for case study")
    case_idx = int(anom_idx[np.argmax(scores[anom_idx])]) if args.case_index < 0 else int(anom_idx[min(args.case_index, len(anom_idx)-1)])
    benign_case_idx = int(benign_idx[np.argmin(scores[benign_idx])])

    x_case = torch.from_numpy(x_test[case_idx:case_idx+1]).float()
    x_benign = torch.from_numpy(x_test[benign_case_idx:benign_case_idx+1]).float()
    
    print(f"Explaining anomaly case (test index {case_idx}) with Integrated Gradients...")
    baseline = torch.zeros_like(x_case)
    ig_anom = integrated_gradients(disc, x_case, baseline=baseline, score_mode=args.score_mode, score_alpha=args.score_alpha, ref_stats=ref_stats)
    
    print(f"Explaining benign case (test index {benign_case_idx}) with Integrated Gradients...")
    ig_benign = integrated_gradients(disc, x_benign, baseline=baseline, score_mode=args.score_mode, score_alpha=args.score_alpha, ref_stats=ref_stats)

    attn_anom = disc.forward_attn_weights(x_case).detach().cpu().numpy().reshape(-1).tolist() if disc.forward_attn_weights(x_case) is not None else []
    attn_benign = disc.forward_attn_weights(x_benign).detach().cpu().numpy().reshape(-1).tolist() if disc.forward_attn_weights(x_benign) is not None else []

    shap_summary = {"enabled": False}
    shap_case = np.empty((0, 0, 0), dtype=np.float32)
    if args.no_shap:
        print("Skipping SHAP because --no-shap was set.")
    elif shap is None:
        print("Skipping SHAP because the shap package is not available.")
    else:
        print("Running SHAP GradientExplainer (this may take a minute)...")
        class Wrap(torch.nn.Module):
            def __init__(self, disc, score_mode, score_alpha, ref_stats):
                super().__init__()
                self.disc = disc
                self.score_mode = score_mode
                self.score_alpha = score_alpha
                self.ref_stats = ref_stats

            def forward(self, x):
                return _anomaly_score_torch(self.disc, x, score_mode=self.score_mode, score_alpha=self.score_alpha, ref_stats=self.ref_stats).unsqueeze(1)

        wrapper = Wrap(disc, args.score_mode, args.score_alpha, ref_stats)
        bg_n = max(1, min(int(args.shap_background), len(x_ref)))
        background = torch.from_numpy(x_ref[:bg_n]).float()
        explainer = shap.GradientExplainer(wrapper, background)
        explain_np = x_case.numpy()
        if args.shap_benign_case:
            explain_np = np.concatenate([explain_np, x_benign.numpy()], axis=0)
        shap_vals = explainer.shap_values(torch.from_numpy(explain_np).float())
        vals = _as_case_array(shap_vals)
        shap_case = vals
        shap_summary = {
            "enabled": True,
            "background_windows": int(bg_n),
            "explained_cases": int(len(vals)),
            "anomaly_abs_mean": float(np.abs(vals[0]).mean()) if len(vals) >= 1 else 0.0,
            "benign_abs_mean": float(np.abs(vals[1]).mean()) if len(vals) >= 2 else 0.0,
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ig_anom_abs = np.abs(ig_anom[0])
    ig_benign_abs = np.abs(ig_benign[0])
    shap_anom_abs = np.abs(shap_case[0]) if len(shap_case) >= 1 else None
    shap_benign_abs = np.abs(shap_case[1]) if len(shap_case) >= 2 else None
    top_rows = _top_feature_rows(
        ds.feature_names,
        ig_anom_abs.mean(axis=0),
        shap_anom_abs.mean(axis=0) if shap_anom_abs is not None else None,
    )

    case_plot_path = out_dir / "xai_case_multiview.png"
    _plot_case_study(
        case_plot_path,
        anomaly_score=float(scores[case_idx]),
        benign_score=float(scores[benign_case_idx]),
        ig_heatmap=ig_anom_abs,
        attention=np.asarray(attn_anom, dtype=np.float32),
        ig_time=ig_anom_abs.mean(axis=1),
        shap_time=shap_anom_abs.mean(axis=1) if shap_anom_abs is not None else None,
        top_rows=top_rows,
    )

    with (out_dir / "xai_case_top_features.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "feature", "ig_abs_mean", "shap_abs_mean"])
        writer.writeheader()
        writer.writerows(top_rows)

    payload = {
        "note": "Attention weights show temporal pooling inside the discriminator and should not be interpreted as a faithful explanation by themselves. Integrated gradients and SHAP are post-hoc attribution views over the final anomaly score.",
        "score_mode": args.score_mode,
        "score_alpha": args.score_alpha,
        "feature_names": ds.feature_names,
        "anomaly_case": {
            "test_index": case_idx,
            "score": float(scores[case_idx]),
            "attention": attn_anom,
            "ig_abs_time_mean": ig_anom_abs.mean(axis=1).tolist(),
            "ig_abs_feature_mean": ig_anom_abs.mean(axis=0).tolist(),
            "shap_abs_time_mean": shap_anom_abs.mean(axis=1).tolist() if shap_anom_abs is not None else [],
            "shap_abs_feature_mean": shap_anom_abs.mean(axis=0).tolist() if shap_anom_abs is not None else [],
            "top_features": top_rows,
        },
        "benign_case": {
            "test_index": benign_case_idx,
            "score": float(scores[benign_case_idx]),
            "attention": attn_benign,
            "ig_abs_time_mean": ig_benign_abs.mean(axis=1).tolist(),
            "ig_abs_feature_mean": ig_benign_abs.mean(axis=0).tolist(),
            "shap_abs_time_mean": shap_benign_abs.mean(axis=1).tolist() if shap_benign_abs is not None else [],
            "shap_abs_feature_mean": shap_benign_abs.mean(axis=0).tolist() if shap_benign_abs is not None else [],
        },
        "shap": shap_summary,
        "artifacts": {
            "case_plot": str(case_plot_path),
            "top_features_csv": str(out_dir / "xai_case_top_features.csv"),
        },
    }
    (out_dir / "xai_upgrade_case_study.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "xai_upgrade_case_study.md").write_text(
        "# XAI Upgrade Case Study\n\n"
        "Attention weights are reported as temporal pooling signals, not as stand-alone explanations.\n\n"
        f"- anomaly_case_index: `{case_idx}`\n"
        f"- benign_case_index: `{benign_case_idx}`\n"
        f"- anomaly_score: `{float(scores[case_idx]):.6f}`\n"
        f"- benign_score: `{float(scores[benign_case_idx]):.6f}`\n"
        f"- shap_enabled: `{bool(shap_summary.get('enabled', False))}`\n",
        encoding="utf-8",
    )
    print(f"Saved multi-view XAI plot to {case_plot_path}")


if __name__ == "__main__":
    main()
