#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('/Users/lijie/Desktop/work/attack')
V1 = ROOT / 'results/cicids_strict_mps/xai_case_trace_e8'
V2 = ROOT / 'results/cicids_strict_mps/xai_case_trace_e8_v2'
OUT = ROOT / 'paper_acsac/figures'
SENS_CSV = ROOT / 'results/cicids_strict_mps/final_paper_tables/table6_noise_sensitivity.csv'


def build_case_heatmap_figure() -> None:
    ig_long = pd.read_csv(V1 / 'case_ig_time_feature_matrix_e8.csv')
    top_nonred = pd.read_csv(V2 / 'case_top_features_nonredundant_e8_v2.csv').sort_values('nonredundant_rank')
    # Metadata is reported in caption/text to avoid clutter and overlap on the figure.

    ordered_features = top_nonred['feature_name'].astype(str).tolist()
    hm = (
        ig_long[ig_long['feature_name'].isin(ordered_features)]
        .pivot_table(index='feature_name', columns='time_step', values='abs_ig_attribution', aggfunc='mean')
        .reindex(index=ordered_features)
        .reindex(columns=list(range(128)))
        .fillna(0.0)
    )
    raw = hm.to_numpy(dtype=np.float64)
    transformed = np.log1p(raw)
    vmax = float(np.percentile(transformed, 99))
    if vmax <= 0:
        vmax = max(float(np.max(transformed)), 1e-6)
    transformed = np.clip(transformed, 0.0, vmax)
    # Improve contrast without changing attribution values: use a low-percentile floor.
    vmin = float(np.percentile(transformed, 8))
    if not np.isfinite(vmin) or vmin < 0 or vmin >= vmax:
        vmin = 0.0

    mean_ig = top_nonred['mean_abs_ig_attribution'].to_numpy(dtype=np.float64)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })

    fig = plt.figure(figsize=(7.5, 3.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.7, 1.4])

    ax_hm = fig.add_subplot(gs[0, 0])
    im = ax_hm.imshow(
        transformed,
        origin='upper',
        aspect='auto',
        interpolation='nearest',
        cmap='Blues',
        vmin=vmin,
        vmax=vmax,
    )
    ax_hm.set_xlabel('Time step')
    ax_hm.set_ylabel('Correlation-pruned feature')
    ax_hm.set_xticks([0, 32, 64, 96, 127])
    ax_hm.set_yticks(np.arange(len(ordered_features)))
    ax_hm.set_yticklabels(ordered_features)

    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.04, pad=0.02)
    cbar.set_label('|IG| (log1p, p99 clipped)')

    ax_bar = fig.add_subplot(gs[0, 1])
    y = np.arange(len(ordered_features))
    ax_bar.barh(y, mean_ig, color='#2563eb')
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('Mean |IG|')
    ax_bar.grid(axis='x', alpha=0.3)

    fig.savefig(OUT / 'case_trace_attribution_heatmap_mainpaper.pdf')
    fig.savefig(OUT / 'case_trace_attribution_heatmap_mainpaper.png', dpi=300)
    plt.close(fig)


def build_case_masking_figure() -> None:
    mask = pd.read_csv(V1 / 'case_masking_verification_e8.csv').sort_values('k')
    base = float(mask['original_fused_score'].iloc[0])
    thr = float(mask['threshold'].iloc[0])

    labels = ['Original']
    vals = [base]
    colors = ['#1d4ed8']
    for _, r in mask.iterrows():
        k = int(r['k'])
        labels.extend([f'IG@{k}', f'Rand@{k}'])
        vals.extend([float(r['ig_masked_fused_score']), float(r['random_masked_fused_score_mean'])])
        colors.extend(['#ea580c', '#6b7280'])

    x = np.arange(len(vals))

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })

    fig, ax = plt.subplots(figsize=(3.45, 2.6), constrained_layout=True)
    ax.bar(x, vals, color=colors, width=0.72)
    ax.axhline(thr, color='#dc2626', linestyle='--', linewidth=1.2, label=r'Threshold $\tau$')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha='right')
    ax.set_ylabel('Fused score')
    ax.set_title('Masking verification')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(frameon=False, loc='upper left', fontsize=7)

    fig.savefig(OUT / 'case_trace_masking_mainpaper.pdf')
    fig.savefig(OUT / 'case_trace_masking_mainpaper.png', dpi=300)
    plt.close(fig)


def build_sensitivity_figure() -> None:
    df = pd.read_csv(SENS_CSV)
    x = df['noise_std'].to_numpy(dtype=np.float64)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.6), constrained_layout=True)

    ax1.plot(x, df['AUC'], marker='o', linewidth=1.6, color='#1d4ed8', label='AUC')
    ax1.plot(x, df['AP'], marker='s', linewidth=1.6, color='#0f766e', label='AP')
    ax1.set_title('(a) Threshold-free ranking metrics')
    ax1.set_xlabel('Perturbation $\\sigma$')
    ax1.set_ylabel('Metric value')
    ax1.set_ylim(0.93, 1.00)
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=False, fontsize=8, loc='lower left')

    ax2.plot(x, df['observed_test_benign_fpr'], marker='o', linewidth=1.6, color='#dc2626', label='Test FPR@$\\tau$')
    ax2.plot(x, df['precision'], marker='^', linewidth=1.6, color='#7c3aed', label='Precision@$\\tau$')
    ax2.plot(x, df['recall'], marker='v', linewidth=1.6, color='#2563eb', label='Recall@$\\tau$')
    ax2.plot(x, df['F1'], marker='D', linewidth=1.6, color='#111827', label='F1@$\\tau$')
    ax2.set_title('(b) Fixed-threshold alarm metrics')
    ax2.set_xlabel('Perturbation $\\sigma$')
    ax2.set_ylabel('Metric value')
    ax2.set_ylim(0.0, 1.02)
    ax2.grid(alpha=0.3)
    ax2.legend(frameon=False, fontsize=7, loc='center right')

    fig.savefig(OUT / 'fixed_threshold_sensitivity_mainpaper.pdf')
    fig.savefig(OUT / 'fixed_threshold_sensitivity_mainpaper.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    build_case_heatmap_figure()
    build_case_masking_figure()
    build_sensitivity_figure()
    print('Wrote figure files to', OUT)
