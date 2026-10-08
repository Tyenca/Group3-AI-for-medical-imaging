"""Slide figures matched to the deck's palette.

Colours are darker versions of the deck's own accents so they stay legible as
small marks on the cream background: teal for the baseline, echoing the
"Baseline & Evaluation" tag, rose for preprocessed, echoing the preprocessing
boxes.

Writes figures 20 to 23. The earlier figures are left untouched.
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

PAPER = "#F2F0EA"          # deck background
INK = "#333333"            # deck text
MUTED = "#8A8880"
GRID = "#DED9CD"
C_BASE = "#2E8C9E"         # darkened deck teal
C_PRE = "#C4568F"          # darkened deck pink
ACCENT = "#7A9440"         # darkened deck sage
PALETTE = {"Baseline": C_BASE, "Preprocessed": C_PRE}

np.random.seed(0)   # stripplot jitter is random, seed it so figures reproduce

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "axes.edgecolor": "#C9C4B6",
    "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "grid.linewidth": 0.9, "font.family": "sans-serif", "font.size": 12,
    "axes.titlesize": 15, "legend.frameon": False,
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
                rows.append({"config": label, "patient": patient,
                             "organ": organ.capitalize(), "metric": "dice_2d",
                             "value": per_slice[slice_patient == patient].mean()})
        for sub, tag in [("metrics", ""), ("metrics_cc", "_cc")]:
            for metric in ["dice_3d", "assd_3d", "hd95_3d", "hd_3d"]:
                z = np.load(run / sub / f"{metric}.npz", allow_pickle=True)
                classes = [str(c) for c in z["classes"]]
                for pi, patient in enumerate([str(p) for p in z["patients"]]):
                    for organ in ORGANS:
                        rows.append({"config": label, "patient": patient,
                                     "organ": organ.capitalize(),
                                     "metric": metric + tag,
                                     "value": z["metric"][pi, classes.index(organ)]})
df = pd.DataFrame(rows)
MM_TICKS = [10, 20, 50, 100, 200, 300]


def raincloud(ax, metric, log=False):
    sub = df[df.metric == metric].copy()
    if log:
        sub["value"] = np.log10(sub["value"])
    sns.violinplot(data=sub, x="organ", y="value", hue="config", order=ORDER,
                   hue_order=HUE, palette=PALETTE, split=True, inner=None,
                   density_norm="width", cut=0, linewidth=0, alpha=0.22, ax=ax,
                   legend=False)
    sns.boxplot(data=sub, x="organ", y="value", hue="config", order=ORDER,
                hue_order=HUE, palette=PALETTE, width=0.26, showfliers=False,
                boxprops={"alpha": 0.5, "zorder": 2},
                medianprops={"color": INK, "linewidth": 1.6}, ax=ax, legend=False)
    sns.stripplot(data=sub, x="organ", y="value", hue="config", order=ORDER,
                  hue_order=HUE, palette=PALETTE, dodge=True, jitter=0.12, size=5,
                  alpha=0.95, edgecolor=PAPER, linewidth=0.5, ax=ax, legend=False,
                  zorder=3)
    ax.set_xlabel("")
    if log:
        lo, hi = sub["value"].min(), sub["value"].max()
        ticks = [t for t in MM_TICKS if lo - 0.2 <= np.log10(t) <= hi + 0.2]
        ax.set_yticks(np.log10(ticks))
        ax.set_yticklabels([str(t) for t in ticks])
    sns.despine(ax=ax, left=True)


KEY = [Line2D([0], [0], marker="o", color=c, linestyle="none", markersize=11, label=n)
       for n, c in PALETTE.items()]

# =====================================================================
# 20 and 21: the two Dice views, identical geometry so the ranking flips
# =====================================================================
for metric, fname, title, sub in [
    ("dice_2d", "20_deck_dice_2d.png", "2D Dice, per slice",
     "Baseline looks better everywhere. But 57% of slices contain no organ, "
     "and score 1.0 for correctly predicting nothing."),
    ("dice_3d", "21_deck_dice_3d.png", "3D Dice, per patient",
     "Same predictions, counted once per patient. Trachea drops from 0.81 to "
     "0.48, and the order reverses."),
]:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    raincloud(ax, metric)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Dice")
    ax.set_title(title, pad=34, loc="left", fontweight="bold")
    ax.text(0, 1.045, sub, transform=ax.transAxes, fontsize=11.5, color=MUTED, va="bottom")
    ax.legend(handles=KEY, loc="lower right", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

# =====================================================================
# 22: how the gap grows as outliers start to count
# =====================================================================
SURF_METRICS = [("assd_3d", "ASSD\nwhole surface"),
                ("hd95_3d", "HD95\nworst 5% dropped"),
                ("hd_3d", "Hausdorff\nsingle worst voxel")]

fig, ax = plt.subplots(figsize=(10, 6.4))
x = np.arange(3)
ax.axhline(1.0, color=MUTED, linewidth=1.3, linestyle=(0, (5, 4)), zorder=1)
ax.text(-0.2, 1.03, "no difference", fontsize=11, color=MUTED, va="bottom")

for organ in ORDER:
    ratios = []
    for metric, _ in SURF_METRICS:
        s = df[(df.metric == metric) & (df.organ == organ)]
        b = s[s.config == "Baseline"].value.mean()
        p = s[s.config == "Preprocessed"].value.mean()
        ratios.append(p / b)
    highlight = organ in ("Heart", "Trachea")
    colour = C_PRE if highlight else MUTED
    ax.plot(x, ratios, color=colour, linewidth=3.0 if highlight else 1.8,
            marker="o", markersize=10 if highlight else 7,
            markeredgecolor=PAPER, markeredgewidth=1.4,
            alpha=1.0 if highlight else 0.65, zorder=3 if highlight else 2)
    ax.annotate(organ, (2, ratios[-1]), xytext=(12, 0), textcoords="offset points",
                fontsize=12, color=colour, va="center",
                fontweight="bold" if highlight else "normal")

ax.set_xticks(x)
ax.set_xticklabels([t for _, t in SURF_METRICS])
ax.set_xlim(-0.25, 2.75)
ax.set_yscale("log")
ax.set_yticks([0.5, 1, 2, 3, 4])
ax.set_yticklabels(["0.5x", "1x", "2x", "3x", "4x"])
ax.yaxis.set_minor_locator(plt.NullLocator())   # log minor ticks fight the custom labels
ax.set_ylabel("Preprocessed surface error, relative to baseline")
ax.set_title("Preprocessing only looks worse once outliers count", pad=48,
             loc="left", fontweight="bold")
ax.text(0, 1.10, "Averaged over the boundary the models are close. By single worst "
                 "voxel, heart is 3.9x worse.",
        transform=ax.transAxes, fontsize=11.5, color=MUTED, va="bottom")
sns.despine(ax=ax, left=True)
fig.tight_layout()
fig.savefig(OUT / "22_deck_outlier_sensitivity.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# =====================================================================
# 23: postprocessing, before and after, all four organs
# =====================================================================
fig, axes = plt.subplots(2, 4, figsize=(16, 8.2), sharex=True, sharey=True)
for col, organ in enumerate(ORDER):
    for row, (suffix, state) in enumerate([("", "Before"), ("_cc", "After")]):
        ax = axes[row, col]
        for name, colour in PALETTE.items():
            d = df[(df.organ == organ) & (df.config == name)]
            dice = d[d.metric == f"dice_3d{suffix}"].value.values
            hd = d[d.metric == f"hd_3d{suffix}"].value.values
            ax.scatter(dice, hd, s=70, color=colour, alpha=0.85,
                       edgecolor=PAPER, linewidth=1, zorder=3)
            ax.axhline(np.median(hd), color=colour, alpha=0.4, linewidth=1.5,
                       linestyle=(0, (5, 3)), zorder=1)
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_yticks(MM_TICKS)
        ax.set_yticklabels([str(t) for t in MM_TICKS])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        sns.despine(ax=ax)
        if row == 0:
            ax.set_title(organ, fontsize=14)
        if row == 1:
            ax.set_xlabel("Dice")
        if col == 0:
            ax.set_ylabel(f"{state} postprocessing\nHausdorff, mm")

fig.legend(handles=KEY, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03),
           fontsize=12.5)
fig.suptitle("Dropping stray fragments cuts the distance, not the overlap",
             fontsize=16, fontweight="bold", x=0.09, ha="left", y=1.005)
fig.text(0.09, 0.955, "Any component under 20% of the largest is removed, same rule for "
                      "both models. Heart falls from 223mm to 22mm.",
         fontsize=12, color=MUTED, ha="left")
fig.tight_layout(rect=[0, 0.01, 1, 0.93])
fig.savefig(OUT / "23_deck_postprocessing.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("Written:")
for f in ["20_deck_dice_2d.png", "21_deck_dice_3d.png",
          "22_deck_outlier_sensitivity.png", "23_deck_postprocessing.png"]:
    print(f"  plots/{f}")
