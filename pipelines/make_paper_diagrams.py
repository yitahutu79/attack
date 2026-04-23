#!/usr/bin/env python3
"""Generate paper diagrams for the Attentive TCN-WGAN-GP manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image


OUT_DIR = Path("attack/paper/figures")


COLORS = {
    "blue": "#D9EAF7",
    "blue_edge": "#2F6F9F",
    "green": "#DFF0D8",
    "green_edge": "#3F7F4F",
    "yellow": "#FFF3CD",
    "yellow_edge": "#A07900",
    "red": "#F8D7DA",
    "red_edge": "#9A3A40",
    "gray": "#F2F2F2",
    "gray_edge": "#666666",
    "purple": "#E8DFF5",
    "purple_edge": "#6B4FA0",
}


def _box(ax, xy, wh, title, body="", fc="blue", lw=1.5, fontsize=10):
    x, y = xy
    w, h = wh
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=lw,
        edgecolor=COLORS[f"{fc}_edge"],
        facecolor=COLORS[fc],
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", fontsize=fontsize, weight="bold")
    if body:
        ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center", fontsize=fontsize - 1, linespacing=1.15)
    return rect


def _arrow(ax, start, end, text="", rad=0.0, color="#333333", lw=1.25):
    arrow = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.025, text, ha="center", va="center", fontsize=8, color=color)


def _save(fig, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{stem}.svg")


def _mini_records(ax, x, y, w, h, color="#2F6F9F"):
    for i in range(5):
        yy = y + h * (0.16 + i * 0.14)
        ax.add_patch(patches.Rectangle((x + w * 0.12, yy), w * 0.68, h * 0.055, facecolor="white", edgecolor=color, lw=0.7))
        ax.add_patch(patches.Circle((x + w * 0.86, yy + h * 0.027), h * 0.018, facecolor=color, edgecolor="none"))


def _mini_windows(ax, x, y, w, h):
    colors = ["#BFD7EA", "#D6EAF8", "#EAF4FB"]
    for i, c in enumerate(colors):
        ax.add_patch(
            patches.Rectangle(
                (x + w * (0.16 + i * 0.08), y + h * (0.26 + i * 0.07)),
                w * 0.42,
                h * 0.38,
                facecolor=c,
                edgecolor=COLORS["blue_edge"],
                lw=0.9,
            )
        )


def _matrix(ax, x, y, w, h, rows=5, cols=9, cmap=("blue", "#F7FBFF")):
    edge = COLORS[f"{cmap[0]}_edge"] if cmap[0] in COLORS else "#777777"
    face = cmap[1]
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, lw=1.0))
    for r in range(1, rows):
        ax.plot([x, x + w], [y + h * r / rows, y + h * r / rows], color=edge, lw=0.35, alpha=0.55)
    for c in range(1, cols):
        ax.plot([x + w * c / cols, x + w * c / cols], [y, y + h], color=edge, lw=0.35, alpha=0.55)


def _conv_stack(ax, x, y, w, h):
    dilations = [1, 2, 4, 8]
    block_w = w / 4.9
    for i, d in enumerate(dilations):
        bx = x + i * block_w * 1.18
        ax.add_patch(
            patches.FancyBboxPatch(
                (bx, y),
                block_w,
                h,
                boxstyle="round,pad=0.01,rounding_size=0.01",
                facecolor=COLORS["purple"],
                edgecolor=COLORS["purple_edge"],
                lw=1.0,
            )
        )
        ax.text(bx + block_w / 2, y + h * 0.52, f"d={d}", ha="center", va="center", fontsize=7, weight="bold")


def _hidden_states(ax, x, y, w, h):
    n = 9
    for i in range(n):
        cx = x + (i + 0.5) * w / n
        alpha = 0.35 + 0.55 * ((i % 4) / 3)
        ax.add_patch(
            patches.Rectangle(
                (cx - w / n * 0.28, y),
                w / n * 0.55,
                h,
                facecolor="#D7C7EE",
                edgecolor=COLORS["purple_edge"],
                alpha=alpha,
                lw=0.7,
            )
        )
    ax.text(x + w / 2, y - 0.025, "$h_1, h_2, \\ldots, h_W$", ha="center", va="top", fontsize=8)


def _attention_bars(ax, x, y, w, h, label=True):
    vals = [0.18, 0.42, 0.25, 0.68, 0.34, 0.92, 0.51, 0.28, 0.72]
    n = len(vals)
    for i, v in enumerate(vals):
        bx = x + i * w / n + w / n * 0.2
        ax.add_patch(
            patches.Rectangle(
                (bx, y),
                w / n * 0.48,
                h * v,
                facecolor="#F5B26B",
                edgecolor="#A35A14",
                lw=0.45,
            )
        )
    if label:
        ax.text(x + w / 2, y + h + 0.018, "learned $a_t$", ha="center", va="bottom", fontsize=8)


def make_framework() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14.8, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.955, "Overall Framework of Attentive TCN-WGAN-GP", fontsize=20, weight="bold", ha="center")
    ax.text(
        0.5,
        0.915,
        "Window-level real-time network anomaly detection with calibrated scoring and post-hoc explanations",
        fontsize=10,
        ha="center",
        color="#555555",
    )

    _box(ax, (0.035, 0.66), (0.12, 0.17), "Network Flow\nRecords", "CICIDS2017\nordered features", "blue", fontsize=9)
    _mini_records(ax, 0.052, 0.69, 0.07, 0.07)
    _box(ax, (0.19, 0.66), (0.12, 0.17), "Preprocessing", "cleaning +\nnormalization", "blue", fontsize=9)
    _box(ax, (0.345, 0.66), (0.12, 0.17), "Sliding Window\nConstruction", "$X\\in\\mathbb{R}^{128\\times77}$\n$W=128, r=16$", "blue", fontsize=9)
    _mini_windows(ax, 0.363, 0.675, 0.08, 0.08)

    detector = patches.FancyBboxPatch(
        (0.50, 0.545),
        0.245,
        0.30,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.7,
        edgecolor=COLORS["purple_edge"],
        facecolor=COLORS["purple"],
    )
    ax.add_patch(detector)
    ax.text(0.622, 0.815, "Attentive TCN-WGAN-GP\nDetector", ha="center", va="center", fontsize=10, weight="bold")
    _box(ax, (0.525, 0.690), (0.085, 0.075), "Generator G", "$z\\rightarrow \\hat{X}$", "green", fontsize=7)
    _box(ax, (0.635, 0.690), (0.085, 0.075), "Critic D", "TCN critic", "red", fontsize=7)
    _box(ax, (0.635, 0.585), (0.085, 0.075), "Attention", "temporal weights", "yellow", fontsize=7)
    _box(ax, (0.525, 0.585), (0.085, 0.075), "WGAN-GP", "critic + penalty", "purple", fontsize=7)
    _arrow(ax, (0.610, 0.728), (0.635, 0.728), "", lw=1.1)
    _arrow(ax, (0.676, 0.690), (0.676, 0.660), "", lw=1.1)
    _arrow(ax, (0.635, 0.622), (0.610, 0.622), "", lw=1.1)
    _arrow(ax, (0.567, 0.660), (0.567, 0.690), "", lw=1.1)

    _box(ax, (0.795, 0.675), (0.13, 0.14), "Fused Anomaly\nScore", "$S=\\alpha S_D+(1-\\alpha)S_F$\n$\\alpha=0.24$", "green", fontsize=8)
    _box(ax, (0.795, 0.455), (0.13, 0.12), "Threshold\nCalibration", "target FPR\n0.05 / 0.15", "yellow", fontsize=8)
    _box(ax, (0.795, 0.265), (0.13, 0.12), "Online Alarm", "normal /\nanomalous window", "red", fontsize=8)

    _box(ax, (0.18, 0.18), (0.16, 0.14), "Evaluation\nProtocol", "baselines\nablation\nseed stability", "gray", fontsize=8)
    _box(ax, (0.48, 0.18), (0.16, 0.14), "XAI Analysis", "feature attribution\ntemporal attribution\nattention weights", "gray", fontsize=8)
    _box(ax, (0.75, 0.18), (0.16, 0.14), "Paper Claims", "accurate detection\nreal-time feasibility\ninterpretable alarms", "gray", fontsize=8)

    _arrow(ax, (0.155, 0.745), (0.19, 0.745))
    _arrow(ax, (0.31, 0.745), (0.345, 0.745))
    _arrow(ax, (0.465, 0.745), (0.50, 0.745))
    _arrow(ax, (0.745, 0.745), (0.795, 0.745))
    _arrow(ax, (0.86, 0.675), (0.86, 0.575))
    _arrow(ax, (0.86, 0.455), (0.86, 0.385))
    _arrow(ax, (0.405, 0.66), (0.265, 0.32), "same split", rad=0.20)
    _arrow(ax, (0.69, 0.545), (0.56, 0.32), "post-hoc", rad=0.13)
    _arrow(ax, (0.64, 0.25), (0.75, 0.25))
    _arrow(ax, (0.34, 0.25), (0.48, 0.25))

    ax.text(0.035, 0.075, "Frozen configuration: W=128, stride=16, score mode=fused, alpha=0.24", fontsize=9, color="#555555")
    fig.tight_layout(pad=0.8)
    _save(fig, "framework_overview")
    plt.close(fig)


def make_model_architecture() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.955, "Attentive TCN-WGAN-GP Model Architecture", fontsize=18, weight="bold", ha="center")

    ax.text(0.055, 0.875, "Real-window critic path", fontsize=10, weight="bold", color=COLORS["purple_edge"])
    _box(ax, (0.055, 0.700), (0.13, 0.13), "Input Window", "$X\\in\\mathbb{R}^{128\\times77}$", "blue", fontsize=8)
    _matrix(ax, 0.078, 0.720, 0.085, 0.055)
    _box(ax, (0.245, 0.690), (0.19, 0.15), "TCN Encoder", "", "purple", fontsize=9)
    _conv_stack(ax, 0.268, 0.705, 0.145, 0.035)
    _box(ax, (0.505, 0.690), (0.16, 0.15), "Hidden States", "$H=[h_1,\\ldots,h_W]$", "purple", fontsize=9)
    _hidden_states(ax, 0.525, 0.720, 0.12, 0.035)
    _box(ax, (0.730, 0.690), (0.18, 0.15), "Attention Pooling", "$h=\\sum_t a_t h_t$", "yellow", fontsize=9)
    _attention_bars(ax, 0.760, 0.710, 0.12, 0.038, label=False)
    _box(ax, (0.785, 0.505), (0.125, 0.10), "Critic D", "$S_D(X)$", "red", fontsize=9)

    _arrow(ax, (0.185, 0.765), (0.245, 0.765))
    _arrow(ax, (0.435, 0.765), (0.505, 0.765))
    _arrow(ax, (0.665, 0.765), (0.730, 0.765))
    _arrow(ax, (0.82, 0.690), (0.845, 0.605))

    ax.text(0.055, 0.575, "Generator path", fontsize=10, weight="bold", color=COLORS["green_edge"])
    _box(ax, (0.055, 0.410), (0.13, 0.12), "Latent Noise", "$z\\sim p(z)$", "blue", fontsize=8)
    ax.scatter([0.092, 0.116, 0.141, 0.110, 0.153], [0.448, 0.486, 0.445, 0.426, 0.482], s=12, color=COLORS["blue_edge"])
    _box(ax, (0.245, 0.395), (0.19, 0.15), "TCN Generator G", "", "green", fontsize=9)
    _conv_stack(ax, 0.268, 0.412, 0.145, 0.035)
    _box(ax, (0.505, 0.405), (0.16, 0.13), "Synthetic Window", "$\\hat{X}=G(z)$", "green", fontsize=9)
    _matrix(ax, 0.542, 0.425, 0.085, 0.050, cmap=("green", "#F4FBF1"))
    _arrow(ax, (0.185, 0.470), (0.245, 0.470))
    _arrow(ax, (0.435, 0.470), (0.505, 0.470))
    _arrow(ax, (0.665, 0.470), (0.785, 0.545), "fake", rad=-0.12)

    _box(ax, (0.280, 0.140), (0.25, 0.13), "WGAN-GP Training", "$L_D=E[D(\\hat{X})]-E[D(X)]+\\lambda GP$\n$L_G=-E[D(G(z))]$", "purple", fontsize=8)
    _box(ax, (0.635, 0.140), (0.25, 0.13), "Fused Inference Score", "$S=\\alpha S_D+(1-\\alpha)S_F$\n$\\alpha=0.24$, threshold $\\tau$", "gray", fontsize=8)
    _box(ax, (0.055, 0.160), (0.13, 0.09), "Feature\nDeviation", "$S_F(X)$", "gray", fontsize=8)
    _box(ax, (0.820, 0.315), (0.12, 0.08), "Decision", "normal /\nanomalous", "red", fontsize=8)

    _arrow(ax, (0.845, 0.505), (0.760, 0.270), "$S_D$")
    _arrow(ax, (0.185, 0.205), (0.635, 0.205), "$S_F$", rad=-0.05)
    _arrow(ax, (0.760, 0.270), (0.820, 0.355))
    _arrow(ax, (0.340, 0.395), (0.395, 0.270), "train G", rad=0.08)
    _arrow(ax, (0.785, 0.545), (0.530, 0.270), "train D", rad=0.08)

    fig.tight_layout(pad=0.8)
    _save(fig, "model_architecture")
    plt.close(fig)


def _open_trimmed(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGB")
    # Remove uniform white margins conservatively.
    import PIL.ImageChops as ImageChops

    bg = Image.new("RGB", im.size, "white")
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        left, top, right, bottom = bbox
        pad = 18
        left = max(0, left - pad)
        top = max(0, top - pad)
        right = min(im.width, right + pad)
        bottom = min(im.height, bottom + pad)
        im = im.crop((left, top, right, bottom))
    return im


def make_xai_panel() -> None:
    paths = [
        ("(a) Feature attribution", OUT_DIR / "xai_feature_importance.png"),
        ("(b) Temporal attribution", OUT_DIR / "xai_time_importance.png"),
        ("(c) Attention weights", OUT_DIR / "attn_weights.png"),
    ]
    images = [(label, _open_trimmed(path)) for label, path in paths]

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2))
    for ax, (label, im) in zip(axes, images):
        ax.imshow(im)
        ax.set_axis_off()
        ax.text(0.5, -0.06, label, ha="center", va="top", transform=ax.transAxes, fontsize=11)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.04, hspace=0.22)
    fig.savefig(OUT_DIR / "xai_panel.png", dpi=300)
    plt.close(fig)


def main() -> None:
    make_framework()
    make_model_architecture()
    make_xai_panel()
    print(f"Paper diagrams written to {OUT_DIR}")


if __name__ == "__main__":
    main()
