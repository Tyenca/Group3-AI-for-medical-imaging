"""
copy_volumes_to_corrected.py
=============================

fix_segthor_gt.py only ever writes GT.nii.gz into the corrected output
folder. This script copies everything else (the CT volume, and any other
files) from the original per-patient folder into the corrected one, so
each corrected patient folder ends up self-contained: volume + fixed GT,
ready to load together in 3D Slicer.

It does NOT touch or recompute GT.nii.gz — it only copies files that are
missing, and skips GT.nii.gz / GT2.nii.gz (those are handled by
fix_segthor_gt.py already).

Usage
-----
    python copy_volumes_to_corrected.py --data-root data/segthor_part1/train --out-root data/segthor_part1_corrected
"""

import argparse
import shutil
from pathlib import Path

SKIP_NAMES = {"GT.nii.gz", "GT2.nii.gz"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True, help="Original patient folders (with the volume files).")
    parser.add_argument("--out-root", type=Path, required=True, help="Corrected folders produced by fix_segthor_gt.py run.")
    args = parser.parse_args()

    patients = sorted(p for p in args.data_root.iterdir() if p.is_dir())

    for p_dir in patients:
        out_dir = args.out_root / p_dir.name
        if not out_dir.exists():
            print(f"Skipping {p_dir.name}: no corrected folder found at {out_dir}")
            continue

        copied_any = False
        for src_file in p_dir.iterdir():
            if not src_file.is_file():
                continue
            if src_file.name in SKIP_NAMES:
                continue

            dst_file = out_dir / src_file.name
            if dst_file.exists():
                continue  # already there, don't overwrite

            shutil.copyfile(src_file, dst_file)
            print(f"  {p_dir.name}: copied {src_file.name} -> {dst_file}")
            copied_any = True

        if not copied_any:
            print(f"  {p_dir.name}: nothing new to copy (already present or no extra files).")


if __name__ == "__main__":
    main()