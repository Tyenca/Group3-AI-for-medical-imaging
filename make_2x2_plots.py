"""The 2x2: preprocessing on/off by postprocessing on/off, three seeds per cell.

Colour carries the preprocessing, fill carries the postprocessing, so the four
cells read as two factors rather than four unrelated conditions.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path("plots")
OUT.mkdir(exist_ok=True)
ORGANS = ["esophagus", "heart", "trachea", "aorta"]
FILTERED = set()  # one uniform rule now covers every class

C_BASE, C_PRE = "#2a78d6", "#eb6834"
INK, SECOND, MUTED, GRID, BASE_LINE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": BASE_LINE,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.family": "sans-serif", "font.size": 10.5, "axes.titlesize": 11.5,
    "figure.titlesize": 13,
})

CELLS = [
    ("baseline_minmax", "metrics",    C_BASE, False, "Baseline, no postprocessing"),
    ("baseline_minmax", "metrics_cc", C_BASE, True,  "Baseline, with postprocessing"),
    ("zscore_skip",     "metrics",    C_PRE,  False, "Preprocessed, no postprocessing"),
    ("zscore_skip",     "metrics_cc", C_PRE,  True,  "Preprocessed, with postprocessing"),
]
OFFSETS = np.linspace(-0.24, 0.24, len(CELLS))


def style(ax):
    ax.grid(True, axis="y", linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE_LINE)


def cell(variant: str, sub: str, metric: str) -> np.ndarray:
    runs = []
    for d in sorted(Path("results").glob(f"{variant}_seed*")):
        z = np.load(d / sub / f"{metric}.npz", allow_pickle=True)
        classes = [str(c) for c in z["classes"]]
        runs.append([np.nanmean(z["metric"][:, classes.index(o)]) for o in ORGANS])
    return np.array(runs)


def panel(ax, metric, ylabel, logy=False):
    x = np.arange(len(ORGANS))
    for (variant, sub, colour, filled, _), off in zip(CELLS, OFFSETS):
        vals = cell(variant, sub, metric)
        ax.errorbar(x + off, vals.mean(axis=0), yerr=vals.std(axis=0), fmt="o",
                    color=colour, markersize=8, capsize=4, linewidth=1.8,
                    markerfacecolor=colour if filled else SURF,
                    markeredgecolor=colour, markeredgewidth=1.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([o.capitalize() for o in ORGANS])
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    style(ax)


LEGEND = [
    Line2D([0], [0], color=C_BASE, marker="o", markersize=8, linewidth=1.8,
           markerfacecolor=SURF, markeredgecolor=C_BASE, markeredgewidth=1.8,
           label="Baseline, no postprocessing"),
    Line2D([0], [0], color=C_BASE, marker="o", markersize=8, linewidth=1.8,
           markerfacecolor=C_BASE, markeredgecolor=C_BASE, label="Baseline, with postprocessing"),
    Line2D([0], [0], color=C_PRE, marker="o", markersize=8, linewidth=1.8,
           markerfacecolor=SURF, markeredgecolor=C_PRE, markeredgewidth=1.8,
           label="Preprocessed, no postprocessing"),
    Line2D([0], [0], color=C_PRE, marker="o", markersize=8, linewidth=1.8,
           markerfacecolor=C_PRE, markeredgecolor=C_PRE, label="Preprocessed, with postprocessing"),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
panel(axes[0], "dice_3d", "Dice score (3D)")
axes[0].set_ylim(0, 1.0)
axes[0].set_title("3D Dice")
panel(axes[1], "hd_3d", "Hausdorff distance in mm (log scale)", logy=True)
axes[1].set_title("Hausdorff distance")

fig.suptitle("Postprocessing removes the distance penalty of preprocessing.\n"
             "Heart Hausdorff falls from 223mm to 22mm, and afterwards no distance "
             "comparison still favours the baseline")
fig.legend(handles=LEGEND, loc="lower center", ncol=2, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, -0.09))
fig.text(0.5, -0.005, "connected component filtering discards any component under 20 percent "
                      "of the largest, applied identically to all four classes and both "
                      "configurations", ha="center", fontsize=9, color=SECOND)
fig.tight_layout(rect=[0, 0.02, 1, 0.92])
fig.savefig(OUT / "11_postprocessing_2x2.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("2x2 summary (mean +/- std over 3 seeds)\n")
for metric, unit in [("dice_3d", ""), ("hd_3d", " mm")]:
    print(f"=== {metric} ===")
    print(f"  {'organ':<11}{'base':>17}{'base+CC':>17}{'pre':>17}{'pre+CC':>17}")
    for i, o in enumerate(ORGANS):
        row = f"  {o:<11}"
        for variant, sub, *_ in CELLS:
            v = cell(variant, sub, metric)[:, i]
            row += f"{v.mean():10.3f}+/-{v.std():5.3f}"
        print(row)
    print()
print(f"Figure written to {(OUT / '11_postprocessing_2x2.png').resolve()}")
