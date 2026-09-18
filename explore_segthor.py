"""
explore_segthor.py
===================

Basic data exploration for the (corrected) SegTHOR dataset. Produces:
  - a per-patient CSV with spacing, shape, HU intensity stats, and organ
    voxel counts
  - a printed summary (means/ranges across the dataset)
  - two PNG charts you can drop straight into slides:
        class_imbalance.png  -> log-scale bar chart of voxel share per class
        spacing_hist.png     -> histogram of slice thickness (Z spacing)

Usage
-----
    # Just explore (spacing, class imbalance, intensity stats, anisotropy):
    python explore_segthor.py --data-root data/segthor_part1_corrected --out-dir exploration_out

    # Also resample every patient to isotropic spacing (e.g. 1mm) and save it:
    python explore_segthor.py --data-root data/segthor_part1_corrected --out-dir exploration_out \\
        --resample-to 1.0 --resampled-out data/segthor_part1_isotropic

Requires: numpy, nibabel, matplotlib, pandas, scipy, seaborn
    pip install numpy nibabel matplotlib pandas scipy seaborn
"""

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom as ndi_zoom
import seaborn as sns

LABEL_NAMES = {0: "background", 1: "esophagus", 2: "heart", 3: "trachea", 4: "aorta"}


def compute_slice_profile(patient_dir: Path, n_bins: int = 50) -> list[dict]:
    """For one patient, compute what % of each axial slice's area is occupied
    by each organ, then bin along the normalized scan position (0=top,
    1=bottom) so profiles from patients with different slice counts can be
    aggregated together. Returns long-form rows: patient, label, bin, pct."""
    gt_path = patient_dir / "GT.nii.gz"
    if not gt_path.exists():
        return []

    gt_img = nib.load(gt_path)
    gt_data = np.asanyarray(gt_img.dataobj).astype(np.int16)
    depth = gt_data.shape[2]
    slice_area = gt_data.shape[0] * gt_data.shape[1]

    z_frac = np.linspace(0.0, 1.0, depth)
    bin_idx = np.clip((z_frac * n_bins).astype(int), 0, n_bins - 1)

    rows = []
    for label, name in LABEL_NAMES.items():
        if label == 0:
            continue  # background isn't interesting for this chart
        per_slice_pct = (gt_data == label).sum(axis=(0, 1)) / slice_area * 100.0
        for b in range(n_bins):
            mask = bin_idx == b
            if not mask.any():
                continue
            rows.append({
                "patient": patient_dir.name,
                "label": name,
                "bin": b,
                "pct": float(per_slice_pct[mask].mean()),
            })
    return rows


def make_organ_profile_plot(data_root: Path, out_dir: Path, n_bins: int = 50):
    """Seaborn line plot: for each organ, % of slice area occupied as a
    function of normalized position along the scan (mean ± 95% CI across
    all patients). This shows how the anatomy evolves through the volume,
    rather than a single static average like a bar chart would."""
    all_rows = []
    for p_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        all_rows.extend(compute_slice_profile(p_dir, n_bins))

    if not all_rows:
        print("No data found for organ profile plot — skipping.")
        return

    df = pd.DataFrame(all_rows)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x="bin", y="pct", hue="label", errorbar=("ci", 95), ax=ax)
    ax.set_xlabel("Normalized position along scan (0 = top / superior, 1 = bottom / inferior)")
    ax.set_ylabel("% of slice area occupied by organ")
    ax.set_title("Organ presence along the scan (mean \u00b1 95% CI across patients)")
    ax.legend(title="Organ")
    fig.tight_layout()
    fig.savefig(out_dir / "organ_profile_along_scan.png", dpi=150)
    plt.close(fig)
    print(f"Saved organ profile plot -> {out_dir / 'organ_profile_along_scan.png'}")


def find_volume_file(patient_dir: Path) -> Path | None:
    """Guess which file in the patient folder is the CT volume (i.e. not GT/GT2)."""
    candidates = [f for f in patient_dir.glob("*.nii.gz") if f.name not in ("GT.nii.gz", "GT2.nii.gz")]
    return candidates[0] if candidates else None


def resample_volume(data: np.ndarray, old_spacing: tuple, new_spacing: tuple, order: int) -> np.ndarray:
    """Resample a 3D array from old_spacing to new_spacing (mm per voxel).
    order=1 (trilinear) for intensity volumes, order=0 (nearest) for label masks."""
    zoom_factors = np.array(old_spacing) / np.array(new_spacing)
    return ndi_zoom(data, zoom=zoom_factors, order=order)


def rescale_affine(affine: np.ndarray, old_spacing: tuple, new_spacing: tuple) -> np.ndarray:
    """Adjust a NIfTI affine's voxel-size columns for the new spacing
    (assumes an axis-aligned / non-sheared affine, true for SegTHOR)."""
    new_affine = affine.copy()
    scale = np.array(new_spacing) / np.array(old_spacing)
    new_affine[:3, :3] = affine[:3, :3] * scale[np.newaxis, :]
    return new_affine


def explore_patient(patient_dir: Path, resample_to: float | None, resampled_out: Path | None) -> dict | None:
    gt_path = patient_dir / "GT.nii.gz"
    if not gt_path.exists():
        return None

    gt_img = nib.load(gt_path)
    gt_data = np.asanyarray(gt_img.dataobj).astype(np.int16)
    spacing = gt_img.header.get_zooms()[:3]

    row = {
        "patient": patient_dir.name,
        "shape_x": gt_data.shape[0],
        "shape_y": gt_data.shape[1],
        "shape_z": gt_data.shape[2],
        "spacing_x_mm": spacing[0],
        "spacing_y_mm": spacing[1],
        "spacing_z_mm": spacing[2],
        "anisotropy_ratio": float(max(spacing) / min(spacing)),
    }

    total_voxels = gt_data.size
    for label, name in LABEL_NAMES.items():
        count = int((gt_data == label).sum())
        row[f"voxels_{name}"] = count
        row[f"pct_{name}"] = 100.0 * count / total_voxels

    vol_path = find_volume_file(patient_dir)
    vol_data = None
    if vol_path is not None:
        vol_img = nib.load(vol_path)
        vol_data = np.asanyarray(vol_img.dataobj).astype(np.float32)
        row["hu_min"] = float(vol_data.min())
        row["hu_max"] = float(vol_data.max())
        row["hu_mean"] = float(vol_data.mean())
        row["hu_std"] = float(vol_data.std())

        # Sanity check: does the CT volume have the same shape/spacing as its GT?
        vol_spacing_check = vol_img.header.get_zooms()[:3]
        row["shape_matches_gt"] = bool(vol_data.shape == gt_data.shape)
        row["spacing_matches_gt"] = bool(
            np.allclose(vol_spacing_check, spacing, atol=1e-3)
        )
        if not row["shape_matches_gt"]:
            print(f"  WARNING: {patient_dir.name} shape mismatch! "
                  f"CT={vol_data.shape} vs GT={gt_data.shape}")
        if not row["spacing_matches_gt"]:
            print(f"  WARNING: {patient_dir.name} spacing mismatch! "
                  f"CT={vol_spacing_check} vs GT={spacing}")
    else:
        row["hu_min"] = row["hu_max"] = row["hu_mean"] = row["hu_std"] = np.nan
        row["shape_matches_gt"] = row["spacing_matches_gt"] = None

    # Optional: resample this patient to isotropic spacing and save it.
    # NOTE: this changes voxel counts (shape), so output will NOT satisfy
    # the baseline pipeline's sanity_ct/sanity_gt hard-coded shape checks
    # ((x,y)==(512,512), 135<=z<=284) without modifying those assertions
    # in your own copy of slice_segthor.py.
    if resample_to is not None and resampled_out is not None:
        new_spacing = (resample_to, resample_to, resample_to)
        out_dir = resampled_out / patient_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # GT: nearest-neighbor only, cast to uint8 to match sanity_gt's dtype requirement.
        gt_resampled = resample_volume(gt_data, spacing, new_spacing, order=0).astype(np.uint8)
        new_affine_gt = rescale_affine(gt_img.affine, spacing, new_spacing)
        gt_out = nib.Nifti1Image(gt_resampled, new_affine_gt)
        gt_out.header.set_data_dtype(np.uint8)
        nib.save(gt_out, out_dir / "GT.nii.gz")

        if vol_data is not None:
            vol_spacing = vol_img.header.get_zooms()[:3]
            vol_resampled = resample_volume(vol_data, vol_spacing, new_spacing, order=1)
            # Round (don't just truncate) before casting back to int16, to
            # match sanity_ct's dtype requirement (int16/int32) rather than
            # leaving it as float32.
            vol_resampled_int = np.round(vol_resampled).astype(np.int16)
            new_affine_vol = rescale_affine(vol_img.affine, vol_spacing, new_spacing)
            vol_out = nib.Nifti1Image(vol_resampled_int, new_affine_vol)
            vol_out.header.set_data_dtype(np.int16)
            nib.save(vol_out, out_dir / vol_path.name)

        row["resampled_shape"] = str(gt_resampled.shape)
        print(f"  {patient_dir.name}: resampled {gt_data.shape} @ {spacing} -> {gt_resampled.shape} @ {new_spacing}")

    return row


def make_class_imbalance_chart(df: pd.DataFrame, out_dir: Path):
    pct_cols = [f"pct_{name}" for name in LABEL_NAMES.values()]
    means = df[pct_cols].mean()
    means.index = list(LABEL_NAMES.values())

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(means.index, means.values, color=["#888888", "#e74c3c", "#f1c40f", "#2ecc71", "#3498db"])
    ax.set_yscale("log")
    ax.set_ylabel("% of total voxels (log scale)")
    ax.set_title("Class imbalance across dataset (average per patient)")
    for bar, val in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "class_imbalance.png", dpi=150)
    plt.close(fig)


def make_spacing_hist(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["spacing_z_mm"], bins=15, color="#3498db", edgecolor="black")
    ax.set_xlabel("Slice thickness / Z spacing (mm)")
    ax.set_ylabel("Number of patients")
    ax.set_title("Distribution of Z voxel spacing across patients")
    fig.tight_layout()
    fig.savefig(out_dir / "spacing_hist.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("exploration_out"))
    parser.add_argument("--resample-to", type=float, default=None,
                         help="If set (e.g. 1.0), also resample every patient to this isotropic spacing in mm.")
    parser.add_argument("--resampled-out", type=Path, default=None,
                         help="Where to save resampled volumes/GT. Required if --resample-to is set.")
    args = parser.parse_args()

    if args.resample_to is not None and args.resampled_out is None:
        parser.error("--resampled-out is required when --resample-to is set.")

    if args.resample_to is not None:
        print("NOTE: isotropic resampling changes each volume's shape. "
              "The baseline's slice_segthor.py hard-codes sanity checks for "
              "(512,512) in-plane shape and 135<=z<=284, so resampled output "
              "will fail those assertions unless you also update them in "
              "your own copy of slice_segthor.py.\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.resampled_out is not None:
        args.resampled_out.mkdir(parents=True, exist_ok=True)

    rows = []
    patients = sorted(p for p in args.data_root.iterdir() if p.is_dir())
    for p_dir in patients:
        row = explore_patient(p_dir, args.resample_to, args.resampled_out)
        if row is not None:
            rows.append(row)
        else:
            print(f"Skipping {p_dir.name}: no GT.nii.gz found.")

    if not rows:
        print("No patients found — check --data-root.")
        return

    df = pd.DataFrame(rows)
    csv_path = args.out_dir / "per_patient_stats.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nProcessed {len(df)} patients.\n")
    print("=== Volume shape (voxels) ===")
    print(df[["shape_x", "shape_y", "shape_z"]].describe().loc[["min", "mean", "max"]])

    print("\n=== Voxel spacing (mm) ===")
    print(df[["spacing_x_mm", "spacing_y_mm", "spacing_z_mm"]].describe().loc[["min", "mean", "max"]])

    print("\n=== Anisotropy (max spacing / min spacing per patient) ===")
    print(f"  mean: {df['anisotropy_ratio'].mean():.2f}x   min: {df['anisotropy_ratio'].min():.2f}x   "
          f"max: {df['anisotropy_ratio'].max():.2f}x")
    if df["anisotropy_ratio"].mean() > 1.5:
        print("  -> Noticeably anisotropic voxels (z-spacing is coarser than in-plane spacing).")
        print("     This is a real motivation for resampling to isotropic spacing before training.")

    print("\n=== HU intensity range ===")
    print(df[["hu_min", "hu_max", "hu_mean", "hu_std"]].describe().loc[["min", "mean", "max"]])

    print("\n=== CT/GT shape & spacing consistency check ===")
    n_shape_ok = int(df["shape_matches_gt"].fillna(False).sum())
    n_spacing_ok = int(df["spacing_matches_gt"].fillna(False).sum())
    n_checked = int(df["shape_matches_gt"].notna().sum())
    print(f"  Shape matches:   {n_shape_ok}/{n_checked} patients OK")
    print(f"  Spacing matches: {n_spacing_ok}/{n_checked} patients OK")
    if n_shape_ok < n_checked or n_spacing_ok < n_checked:
        print("  -> See WARNING lines above for which patients mismatch.")
    else:
        print("  -> All good: every patient's CT volume aligns with its GT in shape and spacing.")

    print("\n=== Class imbalance (% of voxels per label, averaged across patients) ===")
    for name in LABEL_NAMES.values():
        col = f"pct_{name}"
        print(f"  {name:12s}: mean {df[col].mean():.4f}%  (min {df[col].min():.4f}%, max {df[col].max():.4f}%)")

    make_class_imbalance_chart(df, args.out_dir)
    make_spacing_hist(df, args.out_dir)
    make_organ_profile_plot(args.data_root, args.out_dir)

    print(f"\nSaved per-patient CSV -> {csv_path}")
    print(f"Saved charts -> {args.out_dir / 'class_imbalance.png'}, {args.out_dir / 'spacing_hist.png'}")


if __name__ == "__main__":
    main()
