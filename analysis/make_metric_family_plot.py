"""All five metrics in one figure, grouped by what they measure.

Overlap metrics share a 0 to 1 axis, surface metrics share a log millimetre
axis, so within each family the variants are directly comparable. The surface
row is ordered by sensitivity to outliers, which is the axis along which the
two configurations diverge.

2D Dice is averaged within each patient first, so every panel carries the same
12 observations (4 patients x 3 seeds) and the panels are comparable.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

OUT = Path("plots")
ORGANS = ["esophagus", "heart", "trachea", "aorta"]
ORDER = [o.capitalize() for o in ORGANS]
CLASSES = ["background"] + ORGANS
CONFIGS = {"baseline_minmax": "Baseline", "zscore_skip": "Preprocessed"}
HUE = list(CONFIGS.values())

C_BASE, C_PRE = "#2a78d6", "#eb6834"
PALETTE = {"Baseline": C_BASE, "Preprocessed": C_PRE}
INK, SECOND, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

np.random.seed(0)   # stripplot jitter is random, seed it so figures reproduce

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "grid.linewidth": 0.8, "font.family": "sans-serif", "font.size": 10.5,
    "axes.titlesize": 12, "figure.titlesize": 13.5, "legend.frameon": False,
})

# Validation slices are written in sorted filename order, identical for every
# run, so one listing gives the patient boundaries for all of them.
slice_names = sorted(p.name for p in Path("results/zscore_skip/best_epoch/val").glob("*.png"))
slice_patient = np.array([re.match(r"(Patient_\d+)_", n).group(1) for n in slice_names])

rows = []
for variant, label in CONFIGS.items():
    for run in sorted(Path("results").glob(f"{variant}_seed*")):
        seed = run.name.split("seed")[-1]

        dv = np.load(run / "dice_val.npy")
        best = int(dv[:, :, 1:].mean(axis=(1, 2)).argmax())
        for organ in ORGANS:
            per_slice = dv[best, :, CLASSES.index(organ)]
            for patient in np.unique(slice_patient):
                rows.append({"config": label, "seed": seed, "patient": patient,
                             "organ": organ.capitalize(), "metric": "dice_2d",
                             "value": per_slice[slice_patient == patient].mean()})

        for metric in ["dice_3d", "assd_3d", "hd95_3d", "hd_3d"]:
            z = np.load(run / "metrics" / f"{metric}.npz", allow_pickle=True)
            classes = [str(c) for c in z["classes"]]
            for pi, patient in enumerate([str(p) for p in z["patients"]]):
                for organ in ORGANS:
                    rows.append({"config": label, "seed": seed, "patient": patient,
                                 "organ": organ.capitalize(), "metric": metric,
                                 "value": z["metric"][pi, classes.index(organ)]})

df = pd.DataFrame(rows)

OVERLAP = [("dice_2d", "2D Dice, averaged over slices"),
           ("dice_3d", "3D Dice, per patient")]
SURFACE = [("assd_3d", "ASSD\naverages the whole surface"),
           ("hd95_3d", "HD95\ndiscards the worst 5 percent"),
           ("hd_3d", "Hausdorff\nset by the single worst voxel")]
MM_TICKS = [10, 20, 50, 100, 200, 300]


def draw(ax, metric, log):
    sub = df[df.metric == metric].copy()
    if log:
        sub["value"] = np.log10(sub["value"])
    sns.violinplot(data=sub, x="organ", y="value", hue="config", order=ORDER,
                   hue_order=HUE, palette=PALETTE, split=True, inner=None,
                   density_norm="width", cut=0, linewidth=0, alpha=0.25, ax=ax,
                   legend=False)
    sns.boxplot(data=sub, x="organ", y="value", hue="config", order=ORDER,
                hue_order=HUE, palette=PALETTE, width=0.28, showfliers=False,
                boxprops={"alpha": 0.55, "zorder": 2},
                medianprops={"color": INK, "linewidth": 1.5}, ax=ax, legend=False)
    sns.stripplot(data=sub, x="organ", y="value", hue="config", order=ORDER,
                  hue_order=HUE, palette=PALETTE, dodge=True, jitter=0.12, size=4,
                  alpha=0.9, edgecolor=SURF, linewidth=0.4, ax=ax, legend=False,
                  zorder=3)
    ax.set_xlabel("")
    if log:
        lo, hi = sub["value"].min(), sub["value"].max()
        ticks = [t for t in MM_TICKS if lo - 0.2 <= np.log10(t) <= hi + 0.2]
        ax.set_yticks(np.log10(ticks))
        ax.set_yticklabels([str(t) for t in ticks])
    else:
        ax.set_ylim(0, 1.0)
    sns.despine(ax=ax, left=True)


handles = [Line2D([0], [0], marker="o", color=c, linestyle="none", markersize=10, label=n)
           for n, c in PALETTE.items()]

# One figure per slide rather than one crowded grid. The two Dice figures use
# identical geometry and an identical axis, so advancing from one slide to the
# next makes the ranking visibly flip.
DICE_SLIDES = [
    ("dice_2d", "17_slide_dice_2d.png",
     "2D Dice, averaged over slices",
     "The baseline is ahead on every organ, and nothing is significant.\n"
     "But 57 percent of validation slices contain no organ at all, and score a "
     "free 1.0 for predicting nothing"),
    ("dice_3d", "18_slide_dice_3d.png",
     "3D Dice, computed per patient",
     "The same predictions, scored without the free points.\nTrachea falls from "
     "0.81 to 0.48, and the ranking reverses: preprocessing now leads on heart, "
     "trachea and aorta"),
]

for metric, fname, title, subtitle in DICE_SLIDES:
    fig, ax = plt.subplots(figsize=(10, 6))
    draw(ax, metric, log=False)
    ax.set_ylabel("Dice")
    ax.set_title(f"{title}\n", fontsize=14)
    fig.text(0.5, 0.90, subtitle, ha="center", fontsize=10.5, color=SECOND)
    ax.legend(handles=handles, loc="lower right", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)

# Surface error: the two ends of the outlier sensitivity range, side by side.
fig, axes = plt.subplots(1, 2, figsize=(13.5, 6), sharey=True)
for ax, (metric, title) in zip(axes, [SURFACE[0], SURFACE[2]]):
    draw(ax, metric, log=True)
    ax.set_title(title.replace("\n", ": "), fontsize=12.5)
axes[0].set_ylabel("millimetres (log scale)")
axes[0].legend(handles=handles, loc="upper left", fontsize=10.5)
fig.suptitle("Surface error, at the two ends of outlier sensitivity\n"
             "Averaged over the whole boundary the two models are indistinguishable. "
             "Judged by their single worst voxel they are not.\nThe difference therefore "
             "lives in a few isolated points, not in how the organ boundary is traced",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.savefig(OUT / "19_slide_surface_assd_vs_hd.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"{len(df)} observations")
print(df.groupby("metric").value.count())
print(f"\nWritten to {(OUT / '17_metric_families.png').resolve()}")
