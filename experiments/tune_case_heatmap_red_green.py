#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/Users/lijie/Desktop/work/attack')
V1 = ROOT / 'results/cicids_strict_mps/xai_case_trace_e8'
V2 = ROOT / 'results/cicids_strict_mps/xai_case_trace_e8_v2'
OUT = ROOT / 'paper_acsac/figures'


def load_matrix():
    ig_long = pd.read_csv(V1 / 'case_ig_time_feature_matrix_e8.csv')
    top_nonred = pd.read_csv(V2 / 'case_top_features_nonredundant_e8_v2.csv').sort_values('nonredundant_rank')
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
    return transformed, ordered_features, top_nonred['mean_abs_ig_attribution'].to_numpy(dtype=np.float64)


def draw(out_pdf: Path, out_png: Path, data: np.ndarray, features: list[str], mean_ig: np.ndarray,
         cmap: str, p_low: float, p_high: float, gamma: float):
    vmax = float(np.percentile(data, p_high))
    vmin = float(np.percentile(data, p_low))
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = max(float(np.max(data)), 1e-8)
    if vmin >= vmax:
        vmin = 0.0

    clipped = np.clip(data, vmin, vmax)
    # contrast boost in normalized range
    norm = (clipped - vmin) / (vmax - vmin + 1e-12)
    boosted = np.power(norm, gamma)

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
        boosted,
        origin='upper',
        aspect='auto',
        interpolation='nearest',
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    ax_hm.set_xlabel('Time step')
    ax_hm.set_ylabel('Correlation-pruned feature')
    ax_hm.set_xticks([0, 32, 64, 96, 127])
    ax_hm.set_yticks(np.arange(len(features)))
    ax_hm.set_yticklabels(features)

    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.04, pad=0.02)
    cbar.set_label('|IG| contrast-scaled')

    ax_bar = fig.add_subplot(gs[0, 1])
    y = np.arange(len(features))
    ax_bar.barh(y, mean_ig, color='#2563eb')
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('Mean |IG|')
    ax_bar.grid(axis='x', alpha=0.3)

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, features, mean_ig = load_matrix()

    # Version A: vivid red-green, moderate boost
    draw(
        OUT / 'case_trace_attribution_heatmap_mainpaper_vivid_rg.pdf',
        OUT / 'case_trace_attribution_heatmap_mainpaper_vivid_rg.png',
        data, features, mean_ig,
        cmap='RdYlGn_r', p_low=5, p_high=99, gamma=0.65,
    )

    # Version B: vivid red-green, stronger boost (for small differences)
    draw(
        OUT / 'case_trace_attribution_heatmap_mainpaper_vivid_rg_strong.pdf',
        OUT / 'case_trace_attribution_heatmap_mainpaper_vivid_rg_strong.png',
        data, features, mean_ig,
        cmap='RdYlGn_r', p_low=12, p_high=99, gamma=0.50,
    )

    print('Wrote:')
    print(OUT / 'case_trace_attribution_heatmap_mainpaper_vivid_rg.pdf')
    print(OUT / 'case_trace_attribution_heatmap_mainpaper_vivid_rg.png')
    print(OUT / 'case_trace_attribution_heatmap_mainpaper_vivid_rg_strong.pdf')
    print(OUT / 'case_trace_attribution_heatmap_mainpaper_vivid_rg_strong.png')


if __name__ == '__main__':
    main()
