"""Compare preprocessing variants, using whatever seeds are present on disk.

Run directories are discovered by glob, so this works with one run per config or
with three seeds each, without editing anything. A config with a single run is
drawn without error bars, since it has no spread to report.

    python analysis/make_comparison_plots.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path("plots")
OUT.mkdir(exist_ok=True)
RESULTS = Path("results")
ORGANS = ["esophagus", "heart", "trachea", "aorta"]
CLASS_NAMES = ["background"] + ORGANS

C_BASE, C_ZS = "#2a78d6", "#eb6834"
INK, SECOND, MUTED, GRID, BASE_LINE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": BASE_LINE,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.family": "sans-serif", "font.size": 10.5, "axes.titlesize": 11.5,
    "figure.titlesize": 13,
})


def discover(variant: str) -> list[Path]:
    """Seeded run dirs for a variant, falling back to an unseeded one."""
    dirs = sorted(RESULTS.glob(f"{variant}_seed*"))
    if not dirs and (RESULTS / variant).exists():
        dirs = [RESULTS / variant]
    return [d for d in dirs if (d / "metrics" / "dice_3d.npz").exists()]


CONFIGS = [
    {"variant": "baseline_minmax", "color": C_BASE, "marker": "o",
     "name": "Baseline, minmax", "dirs": discover("baseline_minmax")},
    {"variant": "zscore_skip", "color": C_ZS, "marker": "D",
     "name": "Preprocessed, zscore + skip empty", "dirs": discover("zscore_skip")},
]
for c in CONFIGS:
    n = len(c["dirs"])
    if n == 0:
        raise SystemExit(f"No finished runs found for {c['variant']}")
    c["label"] = f"{c['name']} ({n} seeds)" if n > 1 else f"{c['name']} (1 run, no spread)"

LEGEND = [Line2D([0], [0], color=c["color"], marker=c["marker"], markersize=9,
                 linewidth=2 if len(c["dirs"]) > 1 else 0, markeredgecolor=SURF,
                 label=c["label"]) for c in CONFIGS]
OFFSETS = np.linspace(-0.12, 0.12, len(CONFIGS))


def style(ax):
    ax.grid(True, axis="y", linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE_LINE)


def organ_means(run_dir: Path, metric: str) -> np.ndarray:
    """Mean over the validation patients, per organ, for one run."""
    d = np.load(run_dir / "metrics" / f"{metric}.npz", allow_pickle=True)
    classes = [str(c) for c in d["classes"]]
    return np.array([np.nanmean(d["metric"][:, classes.index(o)]) for o in ORGANS])


def stack(cfg, metric: str) -> np.ndarray:
    """(runs, organs) for one config."""
    return np.stack([organ_means(d, metric) for d in cfg["dirs"]])


def draw_config(ax, x, values, cfg, size=90):
    """Error bars with visible individual runs, or a bare marker for a single run."""
    if values.shape[0] > 1:
        ax.errorbar(x, values.mean(axis=0), yerr=values.std(axis=0), fmt=cfg["marker"],
                    color=cfg["color"], markersize=9, capsize=5, linewidth=2,
                    markeredgecolor=SURF, markeredgewidth=1, zorder=3)
        for row in values:
            ax.scatter(x, row, color=cfg["color"], s=22, alpha=0.5, zorder=2)
    else:
        ax.scatter(x, values[0], color=cfg["color"], s=size, marker=cfg["marker"],
                   zorder=3, edgecolor=SURF, linewidth=1)


def compare_panel(ax, metric, ylabel, logy=False):
    x = np.arange(len(ORGANS))
    for cfg, off in zip(CONFIGS, OFFSETS):
        draw_config(ax, x + off, stack(cfg, metric), cfg)
    ax.set_xticks(x)
    ax.set_xticklabels([o.capitalize() for o in ORGANS])
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    style(ax)


# =====================================================================
# 6. Headline: 3D Dice per organ
# =====================================================================
fig, ax = plt.subplots(figsize=(8.5, 5.5))
compare_panel(ax, "dice_3d", "Dice score (3D)")
ax.set_ylim(0, 1.0)
ax.set_title("3D Dice per organ: preprocessing helps trachea and aorta, is borderline on heart,\n"
             "and fails on esophagus where one of three seeds never learned the class")
ax.legend(handles=LEGEND, frameon=False, loc="lower right", fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT / "06_dice_3d_baseline_vs_preprocessed.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# =====================================================================
# 7. Why the metric choice matters
# =====================================================================
def two_d_scores(run_dir: Path):
    dv = np.load(run_dir / "dice_val.npy")
    fg = dv[:, :, 1:].mean(axis=(1, 2))
    best = int(fg.argmax())
    at_best = dv[best, :, 1:]
    free = at_best > 0.9999          # both ground truth and prediction empty
    return fg[best], at_best[~free].mean(), float(free.mean())


LABELS = ["2D per slice\n(all slices)", "2D per slice\n(informative only)", "3D per patient"]
fig, ax = plt.subplots(figsize=(8.5, 5.5))
x = np.arange(3)
summary = {}
for cfg, off in zip(CONFIGS, OFFSETS):
    two_d = np.array([two_d_scores(d) for d in cfg["dirs"]])        # (runs, 3)
    three_d = stack(cfg, "dice_3d").mean(axis=1)                    # (runs,)
    vals = np.column_stack([two_d[:, 0], two_d[:, 1], three_d])     # (runs, 3)
    summary[cfg["variant"]] = (vals, two_d[:, 2].mean())
    draw_config(ax, x + off, vals, cfg, size=100)

base_v, zs_v = summary["baseline_minmax"][0], summary["zscore_skip"][0]
for i in range(3):
    gap = zs_v[:, i].mean() - base_v[:, i].mean()
    top = max(zs_v[:, i].mean(), base_v[:, i].mean())
    ax.annotate(f"gap {gap:+.3f}", (x[i], top + 0.04), ha="center", fontsize=10, color=SECOND)

ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Mean foreground Dice")
ax.set_title("The 2D and 3D metrics disagree about which model is better.\n"
             "Slices where both prediction and ground truth are empty score a free 1.0")
ax.legend(handles=LEGEND, frameon=False, loc="lower left", fontsize=9.5)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "07_metric_choice_changes_the_conclusion.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# =====================================================================
# 8. Distance metrics
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, (metric, title) in zip(axes, [
    ("hd_3d", "Hausdorff distance"),
    ("hd95_3d", "95th percentile Hausdorff"),
    ("assd_3d", "Average symmetric surface distance"),
]):
    compare_panel(ax, metric, "Distance in mm (log scale)", logy=True)
    ax.set_title(title)
fig.suptitle("With both configs seeded the baseline has lower surface distance on almost "
             "every organ, decisively so for heart and trachea")
fig.legend(handles=LEGEND, loc="lower center", ncol=2, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=[0, 0.03, 1, 0.93])
fig.savefig(OUT / "08_distance_metrics_reversal.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# =====================================================================
# 9. Heart Hausdorff per patient
# =====================================================================
def per_patient(run_dir: Path, metric: str, organ: str):
    d = np.load(run_dir / "metrics" / f"{metric}.npz", allow_pickle=True)
    classes = [str(c) for c in d["classes"]]
    return [str(p) for p in d["patients"]], d["metric"][:, classes.index(organ)]


patients, _ = per_patient(CONFIGS[0]["dirs"][0], "hd_3d", "heart")
x = np.arange(len(patients))
width = 0.8 / len(CONFIGS)
fig, ax = plt.subplots(figsize=(8.5, 5))
for i, cfg in enumerate(CONFIGS):
    vals = np.stack([per_patient(d, "hd_3d", "heart")[1] for d in cfg["dirs"]])
    pos = x + (i - (len(CONFIGS) - 1) / 2) * width
    err = vals.std(axis=0) if vals.shape[0] > 1 else None
    ax.bar(pos, vals.mean(axis=0), yerr=err, width=width * 0.9, color=cfg["color"],
           capsize=5, label=cfg["label"], edgecolor=SURF, linewidth=1)
    tops = vals.mean(axis=0) + (err if err is not None else 0)
    for xi, v, t in zip(pos, vals.mean(axis=0), tops):
        ax.annotate(f"{v:.0f}", (xi, t), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9, color=SECOND)

ax.set_xticks(x)
ax.set_xticklabels(patients)
ax.set_ylabel("Hausdorff distance in mm")
ax.set_title("Heart Hausdorff distance per validation patient.\n"
             "The preprocessed model is worse in all four patients, consistently across seeds")
ax.legend(frameon=False, loc="upper left", fontsize=9.5)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "09_heart_hausdorff_per_patient.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# =====================================================================
# 10. Seed consistency, both configs overlaid
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
fig.suptitle("Validation Dice per organ, every seed of both configs overlaid.\n"
             "One preprocessed seed never learns the esophagus, which an average alone would hide")
for ax, organ in zip(axes.flat, ORGANS):
    k = CLASS_NAMES.index(organ)
    for cfg in CONFIGS:
        for d in cfg["dirs"]:
            dv = np.load(d / "dice_val.npy").mean(axis=1)
            ax.plot(np.arange(dv.shape[0]), dv[:, k], color=cfg["color"],
                    linewidth=1.5, alpha=0.85, solid_capstyle="round")
    ax.set_title(organ.capitalize())
    ax.set_ylim(0, 1.02)
    style(ax)
for ax in axes[-1, :]:
    ax.set_xlabel("Epoch")
for ax in axes[:, 0]:
    ax.set_ylabel("Validation Dice (2D)")
fig.legend(handles=LEGEND, loc="lower center", ncol=2, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.02, 1, 0.92])
fig.savefig(OUT / "10_baseline_seed_consistency.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# =====================================================================
# Console summary
# =====================================================================
for cfg in CONFIGS:
    print(f"\n{cfg['label']}  ({len(cfg['dirs'])} run(s))")
    for d in cfg["dirs"]:
        print(f"    {d}")

def significance(metric: str):
    """Effect size, Welch p and whether the two groups' ranges overlap.

    With three runs per group the pooled standard deviation is itself a loose
    estimate, so the effect size alone overstates confidence. The p value and
    the overlap flag are reported alongside it for that reason.
    """
    from scipy import stats
    b, z = stack(CONFIGS[0], metric), stack(CONFIGS[1], metric)
    rows = []
    for i, o in enumerate(ORGANS):
        gap = z[:, i].mean() - b[:, i].mean()
        pooled = np.sqrt(b[:, i].std() ** 2 + z[:, i].std() ** 2)
        eff = gap / pooled if pooled > 0 else np.nan
        if b.shape[0] > 1 and z.shape[0] > 1:
            p = float(stats.ttest_ind(z[:, i], b[:, i], equal_var=False).pvalue)
            overlap = not (z[:, i].min() > b[:, i].max() or b[:, i].min() > z[:, i].max())
        else:
            p, overlap = np.nan, True
        rows.append((o, b[:, i].mean(), b[:, i].std(), z[:, i].mean(), z[:, i].std(), gap, eff, p, overlap))
    return rows


for metric in ["dice_3d", "hd_3d", "hd95_3d", "assd_3d"]:
    print(f"\n{metric}: baseline vs preprocessed")
    print(f"  {'organ':<11}{'baseline':>18}{'preprocessed':>20}{'gap':>10}{'effect':>8}{'p':>8}  verdict")
    for o, bm, bsd, zm, zsd, gap, eff, p, overlap in significance(metric):
        if np.isnan(p):
            verdict = "single run, no test"
        elif overlap or p >= 0.05:
            verdict = "not distinguishable"
        elif p < 0.01:
            verdict = "clear"
        else:
            verdict = "borderline"
        print(f"  {o:<11}{bm:>10.3f} +/-{bsd:>6.3f}{zm:>12.3f} +/-{zsd:>6.3f}"
              f"{gap:>+10.3f}{eff:>8.1f}{p:>8.3f}  {verdict}")

print("\nSeed to seed variability (coefficient of variation, %):")
for cfg in CONFIGS:
    if len(cfg["dirs"]) < 2:
        continue
    print(f"  {cfg['variant']}")
    for metric in ["dice_3d", "hd_3d", "hd95_3d", "assd_3d"]:
        s = stack(cfg, metric)
        cv = s.std(axis=0) / np.abs(s.mean(axis=0)) * 100
        print(f"    {metric:<9} " + "  ".join(f"{o}={c:.1f}" for o, c in zip(ORGANS, cv)))

print(f"\nPlots written to {OUT.resolve()}")
