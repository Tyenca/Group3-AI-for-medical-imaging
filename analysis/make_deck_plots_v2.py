"""Slide figures 24 to 27, in the deck palette.

Revision of 20 to 23. Axes are fitted to each metric's own range, organs get
more horizontal room, the surface comparison goes back to the raincloud form,
and titles are set larger.
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

PAPER = "#FFFFFF"
INK = "#333333"
MUTED = "#8A8880"
GRID = "#DED9CD"
C_BASE = "#2E8C9E"
C_PRE = "#C4568F"
PALETTE = {"Baseline": C_BASE, "Preprocessed": C_PRE}

np.random.seed(0)   # stripplot jitter is random, seed it so figures reproduce

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "axes.edgecolor": "#C9C4B6",
    "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "grid.linewidth": 0.9, "font.family": "sans-serif", "font.size": 13,
    "legend.frameon": False,
})

slice_names = sorted(p.name for p in Path("results/zscore_skip/best_epoch/val").glob("*.png"))
slice_patient = np.array([re.match(r"(Patient_\d+)_", n).group(1) for n in slice_names])

rows = []
for variant, label in CONFIGS.items():
    for run in sorted(Path("results").glob(f"{variant}_seed*")):
        dv = np.load(run / "dice_val.npy")
        best = int(dv[:, :, 1:].mean(axis=(1, 2)).argmax())
        for organ in ORGANS:
            per_slice = dv[best, :, CLASSES.index(organ)]
            for patient in np.unique(slice_patient):
                rows.append({"config": label, "organ": organ.capitalize(),
                             "metric": "dice_2d",
                             "value": per_slice[slice_patient == patient].mean()})
        for sub, tag in [("metrics", ""), ("metrics_cc", "_cc")]:
            for metric in ["dice_3d", "assd_3d", "hd_3d"]:
                z = np.load(run / sub / f"{metric}.npz", allow_pickle=True)
                classes = [str(c) for c in z["classes"]]
                for pi in range(len(z["patients"])):
                    for organ in ORGANS:
                        rows.append({"config": label, "organ": organ.capitalize(),
                                     "metric": metric + tag,
                                     "value": z["metric"][pi, classes.index(organ)]})
df = pd.DataFrame(rows)
MM_TICKS = [3, 5, 10, 20, 50, 100, 200, 300]

KEY = [Line2D([0], [0], marker="o", color=c, linestyle="none", markersize=12, label=n)
       for n, c in PALETTE.items()]


def heading(ax, title, subtitle):
    ax.set_title(title, pad=42, loc="left", fontweight="bold", fontsize=19)
    ax.text(0, 1.045, subtitle, transform=ax.transAxes, fontsize=13,
            color=MUTED, va="bottom")


def raincloud(ax, metric, log=False):
    sub = df[df.metric == metric].copy()
    if log:
        sub["value"] = np.log10(sub["value"])
    sns.violinplot(data=sub, x="organ", y="value", hue="config", order=ORDER,
                   hue_order=HUE, palette=PALETTE, split=True, inner=None,
                   density_norm="width", cut=0, linewidth=0, alpha=0.22,
                   width=0.7, ax=ax, legend=False)
    sns.boxplot(data=sub, x="organ", y="value", hue="config", order=ORDER,
                hue_order=HUE, palette=PALETTE, width=0.24, showfliers=False,
                boxprops={"alpha": 0.5, "zorder": 2},
                medianprops={"color": INK, "linewidth": 1.8}, ax=ax, legend=False)
    sns.stripplot(data=sub, x="organ", y="value", hue="config", order=ORDER,
                  hue_order=HUE, palette=PALETTE, dodge=True, jitter=0.1, size=6,
                  alpha=0.95, edgecolor=PAPER, linewidth=0.6, ax=ax, legend=False,
                  zorder=3)
    ax.set_xlabel("")
    if log:
        lo, hi = sub["value"].min(), sub["value"].max()
        ticks = [t for t in MM_TICKS if lo - 0.15 <= np.log10(t) <= hi + 0.15]
        ax.set_yticks(np.log10(ticks))
        ax.set_yticklabels([str(t) for t in ticks])
    sns.despine(ax=ax, left=True)


# =====================================================================
# 24. 2D Dice
# =====================================================================
fig, ax = plt.subplots(figsize=(13, 6.6))
raincloud(ax, "dice_2d")
ax.set_ylim(0.25, 1.0)
ax.set_ylabel("2D Dice")
heading(ax, "2D Dice by organ",
        "Averaged over the validation slices. Each point is one patient from one seed.")
ax.legend(handles=KEY, loc="lower right", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "24_deck_dice_2d.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# =====================================================================
# 25. 3D Dice
# =====================================================================
fig, ax = plt.subplots(figsize=(13, 6.6))
raincloud(ax, "dice_3d")
ax.set_ylim(-0.07, 0.98)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax.set_ylabel("3D Dice")
heading(ax, "3D Dice by organ",
        "One score per patient volume. Each point is one patient from one seed.")
ax.legend(handles=KEY, loc="lower right", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "25_deck_dice_3d.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# =====================================================================
# 26. Surface distance, ASSD beside Hausdorff
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), sharey=True)
for ax, (metric, label) in zip(axes, [("assd_3d", "ASSD"), ("hd_3d", "Hausdorff")]):
    raincloud(ax, metric, log=True)
    ax.set_title(label, fontsize=16, pad=12)
axes[0].set_ylabel("Millimetres (log scale)")
axes[0].legend(handles=KEY, loc="upper left", fontsize=13)

# The panels share an axis, so ticks must span both metrics. Otherwise the
# second panel's range overwrites the first and ASSD is left unlabelled.
both = np.log10(df[df.metric.isin(["assd_3d", "hd_3d"])].value)
span = [t for t in MM_TICKS if both.min() - 0.15 <= np.log10(t) <= both.max() + 0.15]
axes[0].set_yticks(np.log10(span))
axes[0].set_yticklabels([str(t) for t in span])

fig.suptitle("Surface distance by organ", fontsize=25, fontweight="bold",
             x=0.055, ha="left", y=1.075)
fig.text(0.055, 0.985, "ASSD averages over the whole boundary. Hausdorff takes the "
                       "single largest error.",
         fontsize=15, color=MUTED, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(OUT / "26_deck_surface_distance.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# =====================================================================
# 27. Postprocessing, before and after
# =====================================================================
fig, axes = plt.subplots(2, 4, figsize=(17, 8.6), sharex=True, sharey=True)
for col, organ in enumerate(ORDER):
    for row, (suffix, state) in enumerate([("", "Before"), ("_cc", "After")]):
        ax = axes[row, col]
        for name, colour in PALETTE.items():
            d = df[(df.organ == organ) & (df.config == name)]
            dice = d[d.metric == f"dice_3d{suffix}"].value.values
            hd = d[d.metric == f"hd_3d{suffix}"].value.values
            ax.scatter(dice, hd, s=75, color=colour, alpha=0.85,
                       edgecolor=PAPER, linewidth=1, zorder=3)
            ax.axhline(np.median(hd), color=colour, alpha=0.4, linewidth=1.6,
                       linestyle=(0, (5, 3)), zorder=1)
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_yticks([20, 50, 100, 200, 300])
        ax.set_yticklabels(["20", "50", "100", "200", "300"])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        sns.despine(ax=ax)
        if row == 0:
            ax.set_title(organ, fontsize=16)
        if row == 1:
            ax.set_xlabel("Dice")
        if col == 0:
            ax.set_ylabel(f"{state}\nHausdorff, mm", fontsize=14)

fig.legend(handles=KEY, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.035),
           fontsize=14)
fig.suptitle("Hausdorff before and after removing small components",
             fontsize=26, fontweight="bold", x=0.055, ha="left", y=1.055)
fig.text(0.055, 0.975, "Components below 20% of the largest are deleted. The same rule "
                       "is applied to both configurations.",
         fontsize=15, color=MUTED, ha="left")
fig.tight_layout(rect=[0, 0.01, 1, 0.93])
fig.savefig(OUT / "27_deck_postprocessing.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("Written:")
for f in ["24_deck_dice_2d.png", "25_deck_dice_3d.png",
          "26_deck_surface_distance.png", "27_deck_postprocessing.png"]:
    print(f"  plots/{f}")
