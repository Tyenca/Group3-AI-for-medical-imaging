"""Check every preprocessing step actually happened, and report PASS or FAIL.

Steps 1 and 2 run anywhere the NIfTI data is present. Step 3 runs only where a
sliced directory exists, so on Snellius pass the variant you want to check:

    python verify_preprocessing.py
    python verify_preprocessing.py --slice-dir data/SEGTHOR_zscore_skip --variant zscore_skip
    python verify_preprocessing.py --slice-dir data/SEGTHOR_baseline_minmax --variant baseline_minmax
"""

import argparse
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np

ESOPHAGUS, HEART, TRACHEA, AORTA = 1, 2, 3, 4
EXPECTED_PNG_VALUES = {0, 63, 126, 189, 252}
EXPECTED_SLICES = {"zscore_skip": 1923, "baseline_minmax": 3046, "zscore_noskip": 3046}
EXPECTED_VAL_SLICES = 686
EXPECTED_VAL_PATIENTS = {"Patient_11", "Patient_15", "Patient_17", "Patient_19"}

results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str):
    results.append((ok, msg))
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")


def step1_gt_correction(raw_root: Path, cor_root: Path):
    """The merged label 1 must be split into esophagus and aorta, conserving voxels."""
    print("\nStep 1: ground truth correction (fix_segthor_gt.py)")
    import nibabel as nib

    if not cor_root.exists():
        check(False, f"corrected ground truth not found at {cor_root}")
        return

    # The raw data is only needed to prove the split conserved voxels. It
    # normally lives on the machine where the correction was run, so on a
    # cluster holding only the corrected copy those two checks are skipped
    # rather than failed.
    have_raw = raw_root.exists()
    if not have_raw:
        print(f"  note: raw data absent at {raw_root}, so the conservation check is skipped here."
              f" Run this where the raw data lives to check it.")

    patients = sorted(p.name for p in cor_root.iterdir() if p.is_dir())
    check(len(patients) == 20, f"20 corrected patients found (got {len(patients)})")

    bad_dtype, missing_aorta, not_conserved = [], [], []
    for name in patients:
        cor = np.asanyarray(nib.load(cor_root / name / "GT.nii.gz").dataobj)
        if cor.dtype != np.uint8:
            bad_dtype.append(name)
        if not (cor == AORTA).any():
            missing_aorta.append(name)

        raw_path = raw_root / name / "GT.nii.gz"
        if have_raw and raw_path.exists():
            raw = np.asanyarray(nib.load(raw_path).dataobj)
            # Patient_07 ships a separately drawn GT2, so exact conservation
            # does not apply to it.
            if not (raw_root / name / "GT2.nii.gz").exists():
                if int((raw == ESOPHAGUS).sum()) != int((cor == ESOPHAGUS).sum()) + int((cor == AORTA).sum()):
                    not_conserved.append(name)

    check(not bad_dtype, f"all corrected GT are uint8 (bad: {bad_dtype or 'none'})")
    check(not missing_aorta, f"all corrected GT contain an aorta label (missing: {missing_aorta or 'none'})")
    if have_raw:
        check(not not_conserved,
              f"merged voxels conserved, raw label 1 == corrected label 1 + label 4 "
              f"(violations: {not_conserved or 'none'})")
        sample = np.asanyarray(nib.load(raw_root / patients[0] / "GT.nii.gz").dataobj)
        check(not (sample == AORTA).any(),
              "raw GT has no aorta label, confirming the correction was needed")


def step2_volumes_present(cor_root: Path):
    """Each corrected patient folder must be self contained: CT plus GT."""
    print("\nStep 2: CT volumes copied alongside the corrected GT (copy_volumes_to_corrected.py)")
    if not cor_root.exists():
        check(False, f"{cor_root} not found")
        return
    missing = []
    for p in sorted(x for x in cor_root.iterdir() if x.is_dir()):
        if not (p / "GT.nii.gz").exists() or not (p / f"{p.name}.nii.gz").exists():
            missing.append(p.name)
    check(not missing, f"every patient folder has both CT and GT (incomplete: {missing or 'none'})")


def step3_slicing(slice_dir: Path, variant: str | None):
    """Slice counts, label encoding, image format, spacing, and split integrity."""
    print(f"\nStep 3: slicing ({slice_dir})")
    if not slice_dir.exists():
        print("  skipped, slice directory not present on this machine")
        return

    from PIL import Image

    train_img = sorted((slice_dir / "train" / "img").glob("*.png"))
    val_img = sorted((slice_dir / "val" / "img").glob("*.png"))

    if variant in EXPECTED_SLICES:
        want = EXPECTED_SLICES[variant]
        check(len(train_img) == want,
              f"{variant} produced {len(train_img)} training slices (expected {want})")
    else:
        print(f"  note: {len(train_img)} training slices, no expected count for variant {variant}")
    check(len(val_img) == EXPECTED_VAL_SLICES,
          f"{len(val_img)} validation slices (expected {EXPECTED_VAL_SLICES}, never filtered)")

    # Patient level split integrity: no volume may appear on both sides.
    def patients_of(paths):
        return {re.match(r"(Patient_\d+)_\d+\.png", p.name).group(1) for p in paths}

    train_p, val_p = patients_of(train_img), patients_of(val_img)
    check(not (train_p & val_p),
          f"no patient leaks between train and val (overlap: {sorted(train_p & val_p) or 'none'})")
    check(val_p == EXPECTED_VAL_PATIENTS,
          f"validation patients are the expected four (got {sorted(val_p)})")
    check(len(train_p) == 16, f"16 training patients (got {len(train_p)})")

    # Label encoding and image format, sampled across the split.
    stride = max(1, len(train_img) // 40)
    seen = Counter()
    bad_shape = []
    for p in train_img[::stride]:
        gt = np.array(Image.open(slice_dir / "train" / "gt" / p.name))
        img = np.array(Image.open(p))
        seen.update(np.unique(gt).tolist())
        if img.shape != (256, 256) or img.dtype != np.uint8:
            bad_shape.append(p.name)
    check(set(seen) <= EXPECTED_PNG_VALUES,
          f"GT PNG values within {sorted(EXPECTED_PNG_VALUES)} (found {sorted(seen)})")
    check(not bad_shape, f"images are 256x256 uint8 (bad: {bad_shape[:3] or 'none'})")

    # Normalization fingerprint. Every other check in this file passes whichever
    # normalization ran, because they test structure rather than pixel values.
    # minmax subtracts the volume minimum and divides by its maximum, so a volume
    # spans exactly 0 to 255. The zscore path maps a fixed HU window instead and
    # reaches neither extreme, landing near 100 to 235.
    patient = sorted(train_p)[0]
    lo, hi = 255, 0
    for f in sorted((slice_dir / "train" / "img").glob(f"{patient}_*.png")):
        arr = np.array(Image.open(f))
        lo, hi = min(lo, int(arr.min())), max(hi, int(arr.max()))

    if variant == "baseline_minmax":
        check(lo <= 1 and hi >= 254,
              f"minmax spans the full range for {patient}: observed {lo} to {hi}, expected 0 to 255")
    elif variant in ("zscore_skip", "zscore_noskip"):
        check(lo >= 50 and hi <= 250,
              f"zscore stays inside the range for {patient}: observed {lo} to {hi}, "
              f"expected roughly 100 to 235 (minmax would give 0 and 255)")
    else:
        print(f"  note: {patient} spans {lo} to {hi}, no expected pattern for variant {variant}")

    spacing_path = slice_dir / "spacing.pkl"
    if spacing_path.exists():
        with open(spacing_path, "rb") as f:
            spacing = pickle.load(f)
        check(set(spacing) >= train_p | val_p,
              f"spacing.pkl covers all {len(spacing)} patients, needed for mm distances")
    else:
        check(False, "spacing.pkl exists (metrics_3d.py needs it for mm units)")


def step4_load_time(slice_dir: Path):
    """What the network actually receives after the dataset transforms."""
    print("\nStep 4: load time transforms (main.py img_transform / gt_transform)")
    if not slice_dir.exists():
        print("  skipped, slice directory not present on this machine")
        return
    try:
        import torch
        from PIL import Image
        from main import img_transform, gt_transform
    except Exception as exc:
        check(False, f"could not import transforms: {exc}")
        return

    sample = sorted((slice_dir / "val" / "img").glob("*.png"))[0]
    img = img_transform(Image.open(sample))
    gt = gt_transform(5, Image.open(slice_dir / "val" / "gt" / sample.name))

    check(img.shape == (1, 256, 256), f"image tensor shape {tuple(img.shape)} is (1, 256, 256)")
    check(float(img.min()) >= 0.0 and float(img.max()) <= 1.0,
          f"image scaled into [0, 1] (range {float(img.min()):.3f} to {float(img.max()):.3f})")
    check(gt.shape == (5, 256, 256), f"one hot GT shape {tuple(gt.shape)} is (5, 256, 256)")
    check(bool((gt.sum(dim=0) == 1).all()),
          "one hot GT assigns exactly one class per pixel")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", type=Path, default=Path("data/segthor_part1/train"))
    ap.add_argument("--corrected-root", type=Path, default=Path("data/segthor_part1_corrected/train"))
    ap.add_argument("--slice-dir", type=Path, default=None)
    ap.add_argument("--variant", type=str, default=None,
                    choices=list(EXPECTED_SLICES) + [None])
    args = ap.parse_args()

    print("Preprocessing verification")
    step1_gt_correction(args.raw_root, args.corrected_root)
    step2_volumes_present(args.corrected_root)
    if args.slice_dir:
        step3_slicing(args.slice_dir, args.variant)
        step4_load_time(args.slice_dir)
    else:
        print("\nSteps 3 and 4 skipped, pass --slice-dir to check sliced data.")

    failed = [m for ok, m in results if not ok]
    print(f"\n{len(results) - len(failed)} of {len(results)} checks passed")
    if failed:
        print("Failed:")
        for m in failed:
            print(f"  - {m}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
