"""Presentation figures for the baseline versus preprocessed comparison.

Three figures, each chosen for what the data can actually support:

12  Raincloud per metric. Half violin plus box plus every raw observation.
    With 12 points per cell (4 patients x 3 seeds) a bare violin would be
    showing its own smoothing bandwidth, so the raw points stay visible.

13  Dice against Hausdorff, one point per patient seed. The contradiction that
    drives the whole analysis, in a single view: excellent overlap sitting
    alongside a surface error the width of a torso.

14  Paired slopes. Both configurations are scored on the identical four
    patients, so the observations are paired. A distribution plot discards
    that; this keeps it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

OUT = Path("plots")
OUT.mkdir(exist_ok=True)
ORGANS = ["esophagus", "heart", "trachea", "aorta"]
CONFIGS = {"baseline_minmax": "Baseline", "zscore_skip": "Preprocessed"}
METRICS = {"dice_3d": "Dice (3D)", "hd_3d": "Hausdorff (mm)",
           "hd95_3d": "HD95 (mm)", "assd_3d": "ASSD (mm)"}

C_BASE, C_PRE = "#2a78d6", "#eb6834"
PALETTE = {"Baseline": C_BASE, "Preprocessed": C_PRE}
INK, SECOND, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

np.random.seed(0)   # stripplot jitter is random, seed it so figures reproduce

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "grid.linewidth": 0.8, "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 12, "figure.titlesize": 14, "legend.frameon": False,
})


def load_long() -> pd.DataFrame:
    """One row per config, seed, patient, organ, metric."""
    rows = []
    for variant, label in CONFIGS.items():
        for run in sorted(Path("results").glob(f"{variant}_seed*")):
            seed = run.name.split("seed")[-1]
            for metric in METRICS:
                z = np.load(run / "metrics" / f"{metric}.npz", allow_pickle=True)
                classes = [str(c) for c in z["classes"]]
                patients = [str(p) for p in z["patients"]]
                for pi, patient in enumerate(patients):
                    for organ in ORGANS:
                        rows.append({
                            "config": label, "seed": seed, "patient": patient,
                            "organ": organ.capitalize(), "metric": metric,
                            "value": z["metric"][pi, classes.index(organ)],
                        })
    return pd.DataFrame(rows)


df = load_long()
ORDER = [o.capitalize() for o in ORGANS]
HUE = list(CONFIGS.values())


# =====================================================================
# 12. Raincloud per metric
# =====================================================================
MM_TICKS = [10, 20, 50, 100, 200, 300]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (metric, label) in zip(axes.flat, METRICS.items()):
    sub = df[df.metric == metric].copy()

    # Distances span two orders of magnitude and are strongly skewed. Taking
    # log10 first means the density is estimated in the space it is drawn in;
    # a KDE fitted in linear space and displayed on a log axis is distorted.
    log_axis = metric != "dice_3d"
    if log_axis:
        sub["value"] = np.log10(sub["value"])

    sns.violinplot(data=sub, x="organ", y="value", hue="config", order=ORDER,
                   hue_order=HUE, palette=PALETTE, split=True, inner=None,
                   density_norm="width", cut=0, linewidth=0, alpha=0.25, ax=ax,
                   legend=False)
    sns.boxplot(data=sub, x="organ", y="value", hue="config", order=ORDER,
                hue_order=HUE, palette=PALETTE, width=0.28, showfliers=False,
                boxprops={"alpha": 0.55, "zorder": 2},
                medianprops={"color": INK, "linewidth": 1.6}, ax=ax, legend=False)
    sns.stripplot(data=sub, x="organ", y="value", hue="config", order=ORDER,
                  hue_order=HUE, palette=PALETTE, dodge=True, jitter=0.12,
                  size=4.5, alpha=0.9, edgecolor=SURF, linewidth=0.5, ax=ax,
                  legend=False, zorder=3)

    ax.set_title(label)
    ax.set_xlabel("")
    ax.set_ylabel(label)
    if log_axis:
        lo, hi = sub["value"].min(), sub["value"].max()
        ticks = [t for t in MM_TICKS if lo - 0.15 <= np.log10(t) <= hi + 0.15]
        ax.set_yticks(np.log10(ticks))
        ax.set_yticklabels([str(t) for t in ticks])
    else:
        ax.set_ylim(0, 1.0)
    sns.despine(ax=ax, left=True)

handles = [Line2D([0], [0], marker="o", color=c, linestyle="none", markersize=9, label=n)
           for n, c in PALETTE.items()]
fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Every 3D metric, all 12 observations per cell (4 patients x 3 seeds).\n"
             "Raw points are drawn over the density because at this sample size the "
             "curve reflects smoothing, not data")
fig.tight_layout(rect=[0, 0.03, 1, 0.94])
fig.savefig(OUT / "12_raincloud_all_metrics.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# =====================================================================
# 13. The paradox: Dice against Hausdorff
# =====================================================================
wide = df.pivot_table(index=["config", "seed", "patient", "organ"],
                      columns="metric", values="value").reset_index()

fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), sharey=True)
for ax, organ in zip(axes, ORDER):
    sub = wide[wide.organ == organ]
    for name, colour in PALETTE.items():
        s = sub[sub.config == name]
        ax.scatter(s.dice_3d, s.hd_3d, s=70, color=colour, alpha=0.85,
                   edgecolor=SURF, linewidth=1, label=name, zorder=3)
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_title(organ)
    ax.set_xlabel("Dice (3D)")
    sns.despine(ax=ax)
axes[0].set_ylabel("Hausdorff distance in mm (log scale)")

axes[1].annotate("excellent overlap,\nsurface error the\nwidth of a torso",
                 xy=(0.90, 250), xytext=(0.30, 250), fontsize=9.5, color=SECOND,
                 va="center", arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))

fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("One point per patient and seed. Dice and Hausdorff disagree completely for "
             "heart and trachea,\nwhich is only possible if small predicted fragments sit "
             "far from the organ")
fig.tight_layout(rect=[0, 0.02, 1, 0.88])
fig.savefig(OUT / "13_dice_vs_hausdorff_paradox.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# =====================================================================
# 14. Paired slopes, seed averaged per patient
# =====================================================================
paired = (df.groupby(["config", "patient", "organ", "metric"], as_index=False)
            .value.mean())

fig, axes = plt.subplots(2, 4, figsize=(15, 8))
for row, metric in enumerate(["dice_3d", "hd_3d"]):
    for col, organ in enumerate(ORDER):
        ax = axes[row, col]
        sub = paired[(paired.metric == metric) & (paired.organ == organ)]
        ends = []
        for patient in sorted(sub.patient.unique()):
            p = sub[sub.patient == patient].set_index("config").value
            y0, y1 = p["Baseline"], p["Preprocessed"]
            ax.plot([0, 1], [y0, y1], color=MUTED, linewidth=1.3, zorder=1)
            ax.scatter([0], [y0], color=C_BASE, s=65, zorder=3,
                       edgecolor=SURF, linewidth=1)
            ax.scatter([1], [y1], color=C_PRE, s=65, zorder=3,
                       edgecolor=SURF, linewidth=1)
            ends.append([y1, patient.replace("Patient_", "P")])

        # Nudge labels apart when endpoints nearly coincide, otherwise they
        # overprint. Done in the axis's own space, so log for the distances.
        log_axis = metric == "hd_3d"
        ends.sort(key=lambda e: e[0])
        pos = [np.log10(y) if log_axis else y for y, _ in ends]
        # Space is a fraction of the whole panel, not of the endpoint spread,
        # otherwise tightly clustered endpoints get a nudge too small to read.
        span_vals = np.log10(sub.value) if log_axis else sub.value
        space = (span_vals.max() - span_vals.min()) * 0.085 or 0.02
        for i in range(1, len(pos)):
            pos[i] = max(pos[i], pos[i - 1] + space)
        for (_, name), y_label in zip(ends, pos):
            ax.annotate(name, (1, 10 ** y_label if log_axis else y_label),
                        xytext=(8, 0), textcoords="offset points", fontsize=8,
                        color=SECOND, va="center")
        ax.set_xlim(-0.35, 1.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Baseline", "Preproc."], fontsize=9)
        if metric == "hd_3d":
            ax.set_yscale("log")
        if row == 0:
            ax.set_title(organ)
        if col == 0:
            ax.set_ylabel(METRICS[metric])
        sns.despine(ax=ax)

fig.suptitle("The same four patients are scored under both configurations, so the "
             "observations are paired.\nEach line is one patient, averaged over its three "
             "seeds. Dice on top, Hausdorff below")
fig.tight_layout(rect=[0, 0, 1, 0.91])
fig.savefig(OUT / "14_paired_slopes.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# =====================================================================
# 15. The same view after postprocessing, showing where the points move
# =====================================================================
def load_metrics_sub(sub_dir: str) -> pd.DataFrame:
    rows = []
    for variant, label in CONFIGS.items():
        for run in sorted(Path("results").glob(f"{variant}_seed*")):
            for metric in ["dice_3d", "hd_3d"]:
                path = run / sub_dir / f"{metric}.npz"
                if not path.exists():
                    continue
                z = np.load(path, allow_pickle=True)
                classes = [str(c) for c in z["classes"]]
                for pi, patient in enumerate([str(p) for p in z["patients"]]):
                    for organ in ORGANS:
                        rows.append({
                            "config": label, "seed": run.name.split("seed")[-1],
                            "patient": patient, "organ": organ.capitalize(),
                            "metric": metric,
                            "value": z["metric"][pi, classes.index(organ)]})
    return pd.DataFrame(rows).pivot_table(
        index=["config", "seed", "patient", "organ"], columns="metric",
        values="value").reset_index()


before = load_metrics_sub("metrics")
after = load_metrics_sub("metrics_cc")

if after.empty:
    print("no metrics_cc found, skipping figure 15")
else:
    merged = before.merge(after, on=["config", "seed", "patient", "organ"],
                          suffixes=("_pre", "_post"))
    # One uniform rule now covers every class. These two simply move the most,
    # so they get the focused figure.
    filtered_organs = ["Heart", "Trachea"]

    # Same scatter as figure 13, repeated before and after, on shared axes.
    # Juxtaposition carries the change; arrows between the two states turned
    # into unreadable spaghetti.
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9), sharex=True, sharey=True)
    states = [("before", "Before postprocessing"), ("after", "After postprocessing")]

    for row, organ in enumerate(filtered_organs):
        sub = merged[merged.organ == organ]
        for col, (state, state_label) in enumerate(states):
            ax = axes[row, col]
            dice_col = f"dice_3d_{'pre' if state == 'before' else 'post'}"
            hd_col = f"hd_3d_{'pre' if state == 'before' else 'post'}"
            for name, colour in PALETTE.items():
                s = sub[sub.config == name]
                ax.scatter(s[dice_col], s[hd_col], s=70, color=colour, alpha=0.85,
                           edgecolor=SURF, linewidth=1, zorder=3)
            # median of each config, so the shift is readable without counting dots
            for name, colour in PALETTE.items():
                s = sub[sub.config == name]
                ax.axhline(s[hd_col].median(), color=colour, alpha=0.35,
                           linewidth=1.4, linestyle=(0, (5, 3)), zorder=1)
            ax.set_yscale("log")
            ax.set_xlim(0.3, 1.0)
            sns.despine(ax=ax)
            if row == 0:
                ax.set_title(state_label)
            if row == 1:
                ax.set_xlabel("Dice (3D)")
            if col == 0:
                ax.set_ylabel(f"{organ}\nHausdorff in mm (log)")

    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("The same view as before, repeated after removing floating fragments.\n"
                 "Hausdorff collapses by an order of magnitude, Dice does not move, and "
                 "the two configurations end up overlapping.\nDashed lines mark each "
                 "configuration's median")
    fig.tight_layout(rect=[0, 0.01, 1, 0.90])
    fig.savefig(OUT / "15_postprocessing_movement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =================================================================
    # 16. All four organs, so the untouched classes act as a control
    # =================================================================
    fig, axes = plt.subplots(2, 4, figsize=(16.5, 8.4), sharex=True, sharey=True)
    for col, organ in enumerate(ORDER):
        sub = merged[merged.organ == organ]
        for row, (state, state_label) in enumerate(states):
            ax = axes[row, col]
            suffix = "pre" if state == "before" else "post"
            for name, colour in PALETTE.items():
                s = sub[sub.config == name]
                ax.scatter(s[f"dice_3d_{suffix}"], s[f"hd_3d_{suffix}"], s=60,
                           color=colour, alpha=0.85, edgecolor=SURF, linewidth=1,
                           zorder=3)
                ax.axhline(s[f"hd_3d_{suffix}"].median(), color=colour, alpha=0.35,
                           linewidth=1.4, linestyle=(0, (5, 3)), zorder=1)
            ax.set_yscale("log")
            ax.set_xlim(0, 1)
            sns.despine(ax=ax)
            if row == 0:
                ax.set_title(organ)
            if row == 1:
                ax.set_xlabel("Dice (3D)")
            if col == 0:
                ax.set_ylabel(f"{state_label}\nHausdorff in mm (log)")

    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("All four organs, before and after postprocessing. One uniform rule, "
                 "no per class exceptions:\ndiscard any component under 20 percent of the "
                 "largest. The preprocessed model gains most, because it is\nthe one "
                 "producing the debris")
    fig.tight_layout(rect=[0, 0.01, 1, 0.88])
    fig.savefig(OUT / "16_postprocessing_all_organs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nMovement caused by postprocessing (mean over 12 observations):")
    for organ in filtered_organs:
        for name in PALETTE:
            s = merged[(merged.organ == organ) & (merged.config == name)]
            print(f"  {organ:<8} {name:<13} HD {s.hd_3d_pre.mean():7.1f} -> "
                  f"{s.hd_3d_post.mean():6.1f} mm    Dice {s.dice_3d_pre.mean():.3f} -> "
                  f"{s.dice_3d_post.mean():.3f}")

print(f"{len(df)} observations loaded")
print(df.groupby(['config', 'metric']).value.count().head(4))
print(f"\nWritten to {OUT.resolve()}:")
for f in ["12_raincloud_all_metrics.png", "13_dice_vs_hausdorff_paradox.png",
          "14_paired_slopes.png"]:
    print(f"  {f}")
