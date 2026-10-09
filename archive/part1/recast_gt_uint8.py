"""
recast_gt_uint8.py
===================

fix_segthor_gt.py now saves corrected GT.nii.gz as uint8 (matching the
baseline's sanity_gt check), but if you already ran the full `run` command
before that fix, your existing corrected GT files may be saved as a
different dtype (e.g. int16). This script just reloads and re-saves each
one as uint8 -- no recomputation of the label-splitting algorithm, so it's
fast (seconds, not minutes).

Usage
-----
    python recast_gt_uint8.py --data-root data/segthor_part1_corrected
"""

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    patients = sorted(p for p in args.data_root.iterdir() if p.is_dir())
    n_fixed = 0
    for p_dir in patients:
        gt_path = p_dir / "GT.nii.gz"
        if not gt_path.exists():
            continue

        img = nib.load(gt_path)
        if img.get_data_dtype() == np.uint8:
            print(f"{p_dir.name}: already uint8, skipping.")
            continue

        data = np.asanyarray(img.dataobj).astype(np.uint8)
        out_img = nib.Nifti1Image(data, affine=img.affine, header=img.header)
        out_img.header.set_data_dtype(np.uint8)
        nib.save(out_img, gt_path)
        print(f"{p_dir.name}: recast to uint8.")
        n_fixed += 1

    print(f"\nDone. {n_fixed}/{len(patients)} patients recast.")


if __name__ == "__main__":
    main()