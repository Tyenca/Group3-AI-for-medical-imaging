#!/usr/bin/env python3
"""
metrics_3d.py
=============

Computes **3D** segmentation metrics per patient, from the 2D prediction
slices that main.py saves.

Why this exists
---------------
main.py logs `dice_coef` ("bk...->bk"), which is Dice **per 2D slice**,
averaged over slices. Two problems with that:

1. It is not 3D Dice. The project rubric asks for metrics "3D & correctly
   computed". utils.py even ships `dice_batch` ("bk...->k") labelled
   "used for 3d dice", but main.py never calls it.

2. Slices where a class is absent from BOTH ground truth and prediction
   score (0 + smooth) / (0 + smooth) = 1.0 -- a free perfect score for
   predicting nothing. For a rare class like the esophagus (~0.04% of
   voxels, absent from many slices) this dominates the reported average
   and does not measure segmentation quality at all.

This script instead groups slices back into per-patient 3D volumes and
computes, per patient and per class:

  - Dice (3D)
  - Hausdorff distance (HD), in mm
  - 95th-percentile Hausdorff distance (HD95), in mm
  - Average symmetric surface distance (ASSD), in mm

Distances use real voxel spacing, so they are in millimetres and
comparable to the values reported in the SegTHOR paper.

Empty-mask handling is explicit rather than silently scoring 1.0:
  - GT empty AND prediction empty -> NaN (excluded from averages;
    "nothing to find, nothing found" is not a measurement of quality)
  - exactly one of them empty     -> Dice 0.0, distances NaN (undefined,
                                     there is no surface to measure to)
Both cases are counted and reported so you can see how often they happen.

Usage
-----
    # Compare predictions against the sliced GT PNGs (quick, 256x256 grid)
    python metrics_3d.py --pred-dir results/segthor/best_epoch/val \\
        --gt-dir data/SEGTHOR/val/gt \\
        --spacing-pkl data/SEGTHOR/spacing.pkl \\
        --out-dir results/segthor/metrics

    # More rigorous: resize predictions back to native resolution and
    # compare against the original GT NIfTI volumes
    python metrics_3d.py --pred-dir results/segthor/best_epoch/val \\
        --gt-nifti-root data/segthor_part1_corrected \\
        --out-dir results/segthor/metrics

Requires: numpy, scipy, pillow, nibabel (only for --gt-nifti-root)
    pip install numpy scipy pillow nibabel
"""

import argparse
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

CLASS_NAMES = {0: "background", 1: "esophagus", 2: "heart", 3: "trachea", 4: "aorta"}
K = 5
LABEL_MULT = 63  # slice_segthor.py encodes classes as {0, 63, 126, 189, 252}

# Filenames look like Patient_01_0123.png
STEM_RE = re.compile(r"^(?P<patient>.+)_(?P<idz>\d+)$")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_slices_as_volume(png_dir: Path, patient: str) -> np.ndarray:
    """Stack all 2D PNG slices of one patient into a (H, W, Z) label volume."""
    paths = sorted(png_dir.glob(f"{patient}_*.png"))
    if not paths:
        raise FileNotFoundError(f"No slices for {patient} in {png_dir}")

    slices = []
    for p in paths:
        arr = np.array(Image.open(p))
        # Undo the *63 visualization encoding -> back to {0,1,2,3,4}
        labels = np.rint(arr.astype(np.float32) / LABEL_MULT).astype(np.uint8)
        slices.append(labels)

    return np.stack(slices, axis=-1)


def group_patients(png_dir: Path) -> list[str]:
    patients: set[str] = set()
    for p in png_dir.glob("*.png"):
        m = STEM_RE.match(p.stem)
        if m:
            patients.add(m.group("patient"))
    return sorted(patients)


def resize_nearest(vol: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a label volume (never interpolate labels)."""
    if vol.shape == target_shape:
        return vol
    zoom = np.array(target_shape, dtype=float) / np.array(vol.shape, dtype=float)
    out = ndi.zoom(vol, zoom=zoom, order=0)
    # zoom can be off by a voxel from rounding; pad/crop to be exact.
    fixed = np.zeros(target_shape, dtype=vol.dtype)
    sl = tuple(slice(0, min(a, b)) for a, b in zip(out.shape, target_shape))
    fixed[sl] = out[sl]
    return fixed


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def dice_3d(gt: np.ndarray, pred: np.ndarray) -> float:
    """Plain 3D Dice for one binary mask pair. No smoothing term: empty
    cases are handled explicitly by the caller instead of being papered
    over with a 1e-8 that turns 0/0 into 1.0."""
    inter = np.logical_and(gt, pred).sum()
    denom = gt.sum() + pred.sum()
    return float(2.0 * inter / denom)


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Voxels of the mask that touch its boundary (6-connectivity erosion)."""
    if not mask.any():
        return mask
    eroded = ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(3, 1))
    return mask & ~eroded


def surface_distances(gt: np.ndarray, pred: np.ndarray,
                      spacing: tuple[float, float, float]) -> np.ndarray | None:
    """Symmetric set of surface-to-surface distances in mm, or None if
    either mask is empty (no surface to measure between)."""
    gt_surf = surface_voxels(gt)
    pred_surf = surface_voxels(pred)
    if not gt_surf.any() or not pred_surf.any():
        return None

    # Distance from every voxel to the nearest GT surface voxel, and vice versa.
    dt_to_gt = ndi.distance_transform_edt(~gt_surf, sampling=spacing)
    dt_to_pred = ndi.distance_transform_edt(~pred_surf, sampling=spacing)

    d_pred_to_gt = dt_to_gt[pred_surf]
    d_gt_to_pred = dt_to_pred[gt_surf]

    return np.concatenate([d_pred_to_gt, d_gt_to_pred])


def evaluate_pair(gt_vol: np.ndarray, pred_vol: np.ndarray,
                  spacing: tuple[float, float, float]) -> dict:
    """All metrics for one patient, for every class."""
    res: dict[str, dict] = {}
    for k in range(K):
        gt_k = gt_vol == k
        pred_k = pred_vol == k

        entry: dict[str, float | str] = {}

        if not gt_k.any() and not pred_k.any():
            entry["dice"] = np.nan
            entry["hd"] = entry["hd95"] = entry["assd"] = np.nan
            entry["case"] = "both_empty"
        elif not gt_k.any() or not pred_k.any():
            entry["dice"] = 0.0
            entry["hd"] = entry["hd95"] = entry["assd"] = np.nan
            entry["case"] = "one_empty"
        else:
            entry["dice"] = dice_3d(gt_k, pred_k)
            dists = surface_distances(gt_k, pred_k, spacing)
            if dists is None or dists.size == 0:
                entry["hd"] = entry["hd95"] = entry["assd"] = np.nan
            else:
                entry["hd"] = float(dists.max())
                entry["hd95"] = float(np.percentile(dists, 95))
                entry["assd"] = float(dists.mean())
            entry["case"] = "ok"

        res[str(k)] = entry
    return res


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def print_table(all_results: dict, metric: str, title: str, fmt: str = "{:.4f}"):
    print(f"\n=== {title} ===")
    header = f"{'patient':<14}" + "".join(f"{CLASS_NAMES[k]:>12}" for k in range(1, K))
    print(header)
    print("-" * len(header))

    for patient in sorted(all_results):
        row = f"{patient:<14}"
        for k in range(1, K):
            v = all_results[patient][str(k)][metric]
            row += f"{'--':>12}" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{fmt.format(v):>12}"
        print(row)

    print("-" * len(header))
    mean_row = f"{'MEAN':<14}"
    for k in range(1, K):
        vals = [all_results[p][str(k)][metric] for p in all_results]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        mean_row += f"{'--':>12}" if not vals else f"{fmt.format(float(np.mean(vals))):>12}"
    print(mean_row)

    std_row = f"{'STD':<14}"
    for k in range(1, K):
        vals = [all_results[p][str(k)][metric] for p in all_results]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        std_row += f"{'--':>12}" if len(vals) < 2 else f"{fmt.format(float(np.std(vals))):>12}"
    print(std_row)


def report_empty_cases(all_results: dict):
    counts: dict[int, dict[str, int]] = {k: defaultdict(int) for k in range(1, K)}
    for patient in all_results:
        for k in range(1, K):
            counts[k][all_results[patient][str(k)]["case"]] += 1

    print("\n=== Empty-mask cases (why some cells show '--') ===")
    print(f"{'class':<14}{'ok':>8}{'one_empty':>12}{'both_empty':>12}")
    for k in range(1, K):
        c = counts[k]
        print(f"{CLASS_NAMES[k]:<14}{c['ok']:>8}{c['one_empty']:>12}{c['both_empty']:>12}")
    print("\nNote: 'both_empty' volumes are excluded from the means above.")
    print("The baseline's per-slice dice_coef would have scored each such")
    print("slice 1.0 instead, which is what inflates 2D slice-averaged Dice.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", type=Path, required=True,
                        help="Directory of predicted PNG slices (e.g. results/.../best_epoch/val)")
    parser.add_argument("--gt-dir", type=Path, default=None,
                        help="Directory of GT PNG slices (e.g. data/SEGTHOR/val/gt)")
    parser.add_argument("--gt-nifti-root", type=Path, default=None,
                        help="Root with Patient_XX/GT.nii.gz; predictions are resized back "
                             "to native resolution and compared there (more rigorous)")
    parser.add_argument("--spacing-pkl", type=Path, default=None,
                        help="spacing.pkl written by slice_segthor.py (needed with --gt-dir)")
    parser.add_argument("--in-plane-scale", type=float, default=None,
                        help="With --gt-dir: factor the in-plane resolution was reduced by "
                             "during slicing (512->256 means 2.0). Auto-detected if omitted.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if (args.gt_dir is None) == (args.gt_nifti_root is None):
        parser.error("Pass exactly one of --gt-dir or --gt-nifti-root.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    spacings: dict[str, tuple[float, float, float]] = {}
    if args.spacing_pkl is not None and args.spacing_pkl.exists():
        with open(args.spacing_pkl, "rb") as f:
            spacings = pickle.load(f)

    patients = group_patients(args.pred_dir)
    if not patients:
        print(f"No prediction slices found in {args.pred_dir}")
        return
    print(f"Found {len(patients)} patients: {', '.join(patients)}")

    all_results: dict[str, dict] = {}

    for patient in patients:
        pred_vol = load_slices_as_volume(args.pred_dir, patient)

        if args.gt_nifti_root is not None:
            import nibabel as nib
            gt_path = args.gt_nifti_root / patient / "GT.nii.gz"
            gt_img = nib.load(gt_path)
            gt_vol = np.asanyarray(gt_img.dataobj).astype(np.uint8)
            spacing = tuple(float(s) for s in gt_img.header.get_zooms()[:3])
            # Resize the prediction back up to the GT's native grid.
            pred_vol = resize_nearest(pred_vol, gt_vol.shape)
        else:
            gt_vol = load_slices_as_volume(args.gt_dir, patient)
            if gt_vol.shape != pred_vol.shape:
                raise ValueError(f"{patient}: GT {gt_vol.shape} != pred {pred_vol.shape}")

            if patient in spacings:
                dx, dy, dz = spacings[patient]
                # slice_segthor.py resized each slice from 512x512 down to
                # (typically) 256x256, so each in-plane voxel now covers
                # proportionally more millimetres.
                scale = args.in_plane_scale if args.in_plane_scale else 512.0 / pred_vol.shape[0]
                spacing = (float(dx) * scale, float(dy) * scale, float(dz))
            else:
                print(f"  WARNING: no spacing for {patient}; distances will be in voxels, not mm.")
                spacing = (1.0, 1.0, 1.0)

        all_results[patient] = evaluate_pair(gt_vol, pred_vol, spacing)
        d = all_results[patient]
        summary = ", ".join(
            f"{CLASS_NAMES[k][:4]}={d[str(k)]['dice']:.3f}" if not np.isnan(d[str(k)]["dice"]) else f"{CLASS_NAMES[k][:4]}=--"
            for k in range(1, K))
        print(f"  {patient}: {summary}")

    print_table(all_results, "dice", "3D Dice score (higher is better)")
    print_table(all_results, "hd95", "95th-percentile Hausdorff distance, mm (lower is better)", "{:.2f}")
    print_table(all_results, "assd", "Average symmetric surface distance, mm (lower is better)", "{:.2f}")
    print_table(all_results, "hd", "Hausdorff distance, mm (lower is better)", "{:.2f}")
    report_empty_cases(all_results)

    # Save one .npz per metric, shape (n_patients, K), matching the
    # metric01.npz / metric02.npz layout the submission archive expects.
    patient_list = sorted(all_results)
    for metric in ["dice", "hd", "hd95", "assd"]:
        arr = np.full((len(patient_list), K), np.nan, dtype=np.float32)
        for i, p in enumerate(patient_list):
            for k in range(K):
                v = all_results[p][str(k)][metric]
                arr[i, k] = np.nan if v is None else v
        out_path = args.out_dir / f"{metric}_3d.npz"
        np.savez(out_path, metric=arr, patients=np.array(patient_list),
                 classes=np.array([CLASS_NAMES[k] for k in range(K)]))
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
