"""
fix_segthor_gt.py
==================

Fixes the merged esophagus/aorta ground-truth label found in the SegTHOR
dataset used by the ai4mi_project assignment.

Background
----------
Labels are supposed to be:
    0 = background
    1 = esophagus
    2 = heart
    3 = trachea
    4 = aorta

In most patients, the esophagus and aorta were accidentally merged into a
single label (1). Patient_07 is the exception: it ships both the corrupted
GT.nii.gz *and* a corrected GT2.nii.gz, which we use as ground truth to
calibrate and validate the correction algorithm before applying it to
every other patient.

Method
------
1. Extract the merged mask (voxels == 1).
2. Compute a distance transform (in mm, using real voxel spacing) that
   measures how deep each voxel is inside the merged blob.
3. The aorta is a much thicker tube than the esophagus, so:
     - "aorta seed"     = deepest voxels (far from the blob's surface)
     - "esophagus seed" = voxels far away from the aorta seed
4. Run a seeded watershed on the inverted distance map, restricted to the
   merged mask, so the boundary naturally falls at the narrow "neck"
   between the two organs.
5. Clean up tiny stray fragments per axial slice.
6. Relabel: esophagus part -> 1 (unchanged), aorta part -> 4.
7. Save a corrected copy; leave the original data untouched.

For Patient_07, GT2.nii.gz is used directly as the corrected file (no need
to run the algorithm on it, since we already have the true answer).

Usage
-----
    # Validate the algorithm against Patient_07's known-correct GT2
    python fix_segthor_gt.py validate --data-root data/segthor_train/train

    # Run the fix on the whole dataset
    python fix_segthor_gt.py run --data-root data/segthor_train/train --out-root corrected

Dependencies: numpy, scipy, scikit-image, nibabel
    pip install numpy scipy scikit-image nibabel
"""

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.measure import label as cc_label

ESOPHAGUS = 1
HEART = 2
TRACHEA = 3
AORTA = 4
MERGED = 1  # the corrupted label containing both esophagus + aorta

AORTA_SEED_DEPTH_MM = 6.0       # voxels deeper than this -> "definitely aorta"
ESOPHAGUS_MIN_DIST_MM = 10.0    # voxels this far from aorta seed -> "definitely esophagus"
MIN_FRAGMENT_VOXELS = 15        # per-slice islands smaller than this get reassigned


def split_merged_label(gt: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """Given a GT volume with merged label==1, return a corrected copy
    where that region is split into esophagus (1) and aorta (4).
    Always returns uint8, matching the baseline pipeline's sanity_gt check."""
    merged_mask = gt == MERGED
    if not merged_mask.any():
        return gt.copy().astype(np.uint8)

    # Distance from the *outside* of the merged blob, in mm.
    depth = ndi.distance_transform_edt(merged_mask, sampling=spacing)

    # Seed 1: aorta = deepest core of the blob (thick tube).
    aorta_seed_mask = merged_mask & (depth > AORTA_SEED_DEPTH_MM)
    aorta_seed_mask = _largest_component(aorta_seed_mask)

    if not aorta_seed_mask.any():
        # Fallback: no voxel deep enough (unusually thin aorta on this scan) —
        # relax the threshold once.
        aorta_seed_mask = _largest_component(merged_mask & (depth > AORTA_SEED_DEPTH_MM / 2))

    # Seed 2: esophagus = merged voxels far from the aorta seed.
    dist_from_aorta_seed = ndi.distance_transform_edt(~aorta_seed_mask, sampling=spacing)
    esoph_seed_mask = merged_mask & (dist_from_aorta_seed > ESOPHAGUS_MIN_DIST_MM)

    # Build marker volume for watershed: 1 = esophagus seed, 2 = aorta seed, 0 = unknown.
    markers = np.zeros(gt.shape, dtype=np.int32)
    markers[esoph_seed_mask] = 1
    markers[aorta_seed_mask] = 2

    if markers.max() < 2:
        # Could not find two distinct seeds — leave this volume's merged
        # region as esophagus and flag it for manual review.
        print("  WARNING: could not find two separate seeds; leaving as label 1.")
        return gt.copy().astype(np.uint8)

    # Watershed on the inverted depth map (so it floods from the seeds
    # outward and cuts at the shallowest connecting "neck").
    elevation = -depth
    labels_ws = watershed(elevation, markers=markers, mask=merged_mask)

    corrected = gt.copy()
    corrected[labels_ws == 1] = ESOPHAGUS
    corrected[labels_ws == 2] = AORTA

    corrected = _clean_small_fragments(corrected)
    return corrected.astype(np.uint8)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    lbl = cc_label(mask)
    if lbl.max() == 0:
        return mask
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    biggest = sizes.argmax()
    return lbl == biggest


def _clean_small_fragments(vol: np.ndarray) -> np.ndarray:
    """Per axial slice, reassign tiny esophagus/aorta islands to whichever
    of the two neighbours them, to remove watershed speckle noise."""
    out = vol.copy()
    for z in range(vol.shape[2]):
        sl = out[:, :, z]
        for target_label, other_label in ((ESOPHAGUS, AORTA), (AORTA, ESOPHAGUS)):
            mask = sl == target_label
            if not mask.any():
                continue
            lbl = cc_label(mask)
            for i in range(1, lbl.max() + 1):
                comp = lbl == i
                if comp.sum() < MIN_FRAGMENT_VOXELS:
                    dilated = ndi.binary_dilation(comp, iterations=2)
                    border = dilated & ~comp
                    if (sl[border] == other_label).sum() > (border.sum() / 2):
                        sl[comp] = other_label
        out[:, :, z] = sl
    return out


def dice(a: np.ndarray, b: np.ndarray, label: int) -> float:
    a_mask, b_mask = a == label, b == label
    inter = np.logical_and(a_mask, b_mask).sum()
    denom = a_mask.sum() + b_mask.sum()
    return 1.0 if denom == 0 else 2.0 * inter / denom


def validate(data_root: Path, patient: str = "Patient_07"):
    p_dir = data_root / patient
    bad = nib.load(p_dir / "GT.nii.gz")
    good = nib.load(p_dir / "GT2.nii.gz")

    bad_data = np.asanyarray(bad.dataobj).astype(np.int16)
    good_data = np.asanyarray(good.dataobj).astype(np.int16)
    spacing = bad.header.get_zooms()[:3]

    fixed = split_merged_label(bad_data, spacing)

    n_wrong = int((fixed != good_data).sum())
    n_total = fixed.size
    print(f"Validation against {patient}/GT2.nii.gz")
    print(f"  Voxels differing: {n_wrong} / {n_total} ({100 * n_wrong / n_total:.4f}%)")
    print(f"  Dice esophagus (1): {dice(fixed, good_data, ESOPHAGUS):.4f}")
    print(f"  Dice aorta (4):     {dice(fixed, good_data, AORTA):.4f}")
    print(f"  Dice heart (2):     {dice(fixed, good_data, HEART):.4f}")
    print(f"  Dice trachea (3):   {dice(fixed, good_data, TRACHEA):.4f}")


def run(data_root: Path, out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    patients = sorted(p for p in data_root.iterdir() if p.is_dir())

    for p_dir in patients:
        gt_path = p_dir / "GT.nii.gz"
        gt2_path = p_dir / "GT2.nii.gz"
        if not gt_path.exists():
            continue

        out_dir = out_root / p_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "GT.nii.gz"

        print(f"Processing {p_dir.name} ...")

        if gt2_path.exists():
            # We already have the correct answer for this patient — reload and
            # re-save as uint8 to guarantee dtype compatibility with the
            # baseline pipeline's sanity_gt check, rather than assuming the
            # raw file already matches.
            gt2_img = nib.load(gt2_path)
            gt2_data = np.asanyarray(gt2_img.dataobj).astype(np.uint8)
            gt2_out = nib.Nifti1Image(gt2_data, affine=gt2_img.affine, header=gt2_img.header)
            gt2_out.header.set_data_dtype(np.uint8)
            nib.save(gt2_out, out_path)
            print("  Has GT2.nii.gz -> saved as the corrected GT (cast to uint8).")
            continue

        img = nib.load(gt_path)
        data = np.asanyarray(img.dataobj).astype(np.int16)
        spacing = img.header.get_zooms()[:3]

        fixed = split_merged_label(data, spacing)

        fixed_img = nib.Nifti1Image(fixed, affine=img.affine, header=img.header)
        fixed_img.header.set_data_dtype(np.uint8)
        nib.save(fixed_img, out_path)
        print(f"  Saved corrected GT -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="Validate the algorithm against Patient_07's GT2.")
    p_val.add_argument("--data-root", type=Path, required=True)
    p_val.add_argument("--patient", type=str, default="Patient_07")

    p_run = sub.add_parser("run", help="Run the correction on the whole dataset.")
    p_run.add_argument("--data-root", type=Path, required=True)
    p_run.add_argument("--out-root", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "validate":
        validate(args.data_root, args.patient)
    elif args.command == "run":
        run(args.data_root, args.out_root)


if __name__ == "__main__":
    main()
