"""Generate baseline result plots from the downloaded training/metric artifacts."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

# --- validated categorical palette (dataviz skill reference palette, light mode) ---
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"

ORGAN_COLOR = {
    "esophagus": BLUE,
    "heart": ORANGE,
    "trachea": AQUA,
    "aorta": YELLOW,
}
CLASS_NAMES = ["background", "esophagus", "heart", "trachea", "aorta"]

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "font.family": "sans-serif",
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "figure.titlesize": 13,
})


def style_axis(ax):
    ax.grid(True, axis="y", linewidth=0.8, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)


# =====================================================================
# 1. Dice score per organ across training epochs, train vs validation
# =====================================================================
dice_tra = np.load("results/zscore_skip/dice_tra.npy").mean(axis=1)  # (epochs, 5)
dice_val = np.load("results/zscore_skip/dice_val.npy").mean(axis=1)  # (epochs, 5)
epochs = np.arange(dice_tra.shape[0])

fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
fig.suptitle("Dice score per organ across 50 training epochs, training versus validation")

for ax, organ in zip(axes.flat, ["esophagus", "heart", "trachea", "aorta"]):
    k = CLASS_NAMES.index(organ)
    color = ORGAN_COLOR[organ]
    ax.plot(epochs, dice_tra[:, k], color=color, linewidth=2, solid_capstyle="round",
            label="Training")
    ax.plot(epochs, dice_val[:, k], color=color, linewidth=2, linestyle="--",
            dashes=(4, 2), solid_capstyle="round", label="Validation")
    ax.set_title(organ.capitalize())
    ax.set_ylim(0, 1.02)
    style_axis(ax)

for ax in axes[-1, :]:
    ax.set_xlabel("Epoch")
for ax in axes[:, 0]:
    ax.set_ylabel("Dice score")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig.savefig(OUT_DIR / "01_dice_curves_per_organ.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# =====================================================================
# 2. Cross entropy loss during training and validation across epochs
# =====================================================================
loss_tra = np.load("results/zscore_skip/loss_tra.npy").mean(axis=1)  # (epochs,)
loss_val = np.load("results/zscore_skip/loss_val.npy").mean(axis=1)  # (epochs,)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(epochs, loss_tra, color=BLUE, linewidth=2, solid_capstyle="round", label="Training")
ax.plot(epochs, loss_val, color=ORANGE, linewidth=2, solid_capstyle="round", label="Validation")
ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross entropy loss (log scale)")
ax.set_title("Cross entropy loss during training and validation across 50 epochs")
style_axis(ax)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(OUT_DIR / "02_loss_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# =====================================================================
# 3. Final 3D Dice score per organ on the 4 validation patients
# =====================================================================
d3 = np.load("results/zscore_skip/metrics/dice_3d.npz", allow_pickle=True)
metric = d3["metric"]  # (patients, classes)
patients = [str(p) for p in d3["patients"]]
classes = [str(c) for c in d3["classes"]]
organs = ["esophagus", "heart", "trachea", "aorta"]

fig, ax = plt.subplots(figsize=(7.5, 5))
positions = np.arange(len(organs))
box_data = [metric[:, classes.index(o)] for o in organs]

bp = ax.boxplot(box_data, positions=positions, widths=0.45, patch_artist=True,
                 showfliers=False, medianprops={"color": INK, "linewidth": 1.5},
                 whiskerprops={"color": BASELINE}, capprops={"color": BASELINE})
for patch, organ in zip(bp["boxes"], organs):
    patch.set_facecolor(ORGAN_COLOR[organ])
    patch.set_alpha(0.35)
    patch.set_edgecolor(ORGAN_COLOR[organ])

rng = np.random.default_rng(0)
for i, organ in enumerate(organs):
    ys = box_data[i]
    xs = i + rng.uniform(-0.08, 0.08, size=len(ys))
    ax.scatter(xs, ys, color=ORGAN_COLOR[organ], s=64, zorder=3,
               edgecolor=SURFACE, linewidth=1)

ax.set_xticks(positions)
ax.set_xticklabels([o.capitalize() for o in organs])
ax.set_ylim(0, 1.02)
ax.set_ylabel("Dice score (3D)")
ax.set_title("Final 3D Dice score per organ, measured on the 4 held out validation patients")
style_axis(ax)
fig.tight_layout()
fig.savefig(OUT_DIR / "03_dice_3d_per_organ.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# =====================================================================
# 4. 3D surface distance error per organ (HD95 and ASSD), two panels
# =====================================================================
def load_metric(name):
    d = np.load(f"results/zscore_skip/metrics/{name}.npz", allow_pickle=True)
    return d["metric"], [str(c) for c in d["classes"]]

hd95, hd95_classes = load_metric("hd95_3d")
assd, assd_classes = load_metric("assd_3d")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("3D surface distance error per organ on the validation patients")

for ax, (data, cls, title) in zip(
    axes,
    [
        (hd95, hd95_classes, "95th percentile Hausdorff distance (mm)"),
        (assd, assd_classes, "Average symmetric surface distance (mm)"),
    ],
):
    box_data = [data[:, cls.index(o)] for o in organs]
    bp = ax.boxplot(box_data, positions=positions, widths=0.45, patch_artist=True,
                     showfliers=False, medianprops={"color": INK, "linewidth": 1.5},
                     whiskerprops={"color": BASELINE}, capprops={"color": BASELINE})
    for patch, organ in zip(bp["boxes"], organs):
        patch.set_facecolor(ORGAN_COLOR[organ])
        patch.set_alpha(0.35)
        patch.set_edgecolor(ORGAN_COLOR[organ])
    for i, organ in enumerate(organs):
        ys = box_data[i]
        xs = i + rng.uniform(-0.08, 0.08, size=len(ys))
        ax.scatter(xs, ys, color=ORGAN_COLOR[organ], s=64, zorder=3,
                   edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([o.capitalize() for o in organs])
    ax.set_ylabel("Distance (mm)")
    ax.set_title(title)
    style_axis(ax)

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_DIR / "04_distance_metrics_per_organ.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# =====================================================================
# 5. Diagnostic: raw Hausdorff distance for the heart, per patient
# =====================================================================
hd_raw, hd_raw_classes = load_metric("hd_3d")
heart_vals = hd_raw[:, hd_raw_classes.index("heart")]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(patients, heart_vals, color=ORGAN_COLOR["heart"], width=0.55,
              edgecolor=ORGAN_COLOR["heart"])
for rect, val in zip(bars, heart_vals):
    ax.annotate(f"{val:.0f}", (rect.get_x() + rect.get_width() / 2, val),
                textcoords="offset points", xytext=(0, 4), ha="center",
                fontsize=9.5, color=SECONDARY_INK)
ax.set_ylabel("Hausdorff distance (mm)")
ax.set_title("Raw Hausdorff distance for the heart in each validation patient, showing a false positive outlier")
style_axis(ax)
fig.tight_layout()
fig.savefig(OUT_DIR / "05_heart_hd_outlier_by_patient.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Saved plots to", OUT_DIR.resolve())
