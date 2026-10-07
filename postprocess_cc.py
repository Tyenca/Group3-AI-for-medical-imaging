"""Connected component filtering of predicted slices, applied per patient in 3D.

Stray isolated blobs barely move Dice but set the Hausdorff distance by
themselves, so removing them targets exactly the failure this project measured.

The ground truth for heart, trachea and aorta is a single connected component in
every validation patient, so keeping the largest component is anatomically
justified for those classes. It is not free, though: when the model fragments a
true structure, dropping the smaller piece loses real prediction. `--min-frac`
keeps any component at least that fraction of the largest, which is the more
forgiving rule.

    python postprocess_cc.py --pred-dir results/X/best_epoch/val --out-dir results/X/best_epoch/val_cc
    python postprocess_cc.py --pred-dir ... --out-dir ... --min-frac 0.25
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

CLASS_VALUES = {"esophagus": 63, "heart": 126, "trachea": 189, "aorta": 252}

# One rule for every class. Keeping only the largest component wrecks esophagus
# and aorta, whose predictions fragment genuinely and whose ground truth is in
# the esophagus case split into up to 16 components. A size threshold instead of
# a single survivor improves Hausdorff on all four organs while leaving Dice
# essentially untouched, and needs no per class exceptions.
DEFAULT_CLASSES = ["esophagus", "heart", "trachea", "aorta"]
DEFAULT_MIN_FRAC = 0.20


def filter_volume(vol: np.ndarray, min_frac: float, values: list[int]) -> tuple[np.ndarray, int, int]:
    """Drop components smaller than min_frac of the largest, per class."""
    out = vol.copy()
    removed_components = removed_voxels = 0
    for value in values:
        mask = vol == value
        if not mask.any():
            continue
        labels, n = ndi.label(mask)
        if n <= 1:
            continue
        sizes = ndi.sum(mask, labels, range(1, n + 1))
        keep = sizes >= max(sizes) * min_frac
        for i, keep_it in enumerate(keep, start=1):
            if not keep_it:
                comp = labels == i
                out[comp] = 0
                removed_components += 1
                removed_voxels += int(comp.sum())
    return out, removed_components, removed_voxels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-frac", type=float, default=DEFAULT_MIN_FRAC,
                    help=f"Keep components at least this fraction of the largest. "
                         f"1.0 keeps only the largest. Default {DEFAULT_MIN_FRAC}, which "
                         f"measured best across all four classes.")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                    choices=list(CLASS_VALUES),
                    help=f"Classes to filter. Default {DEFAULT_CLASSES}, which are the "
                         f"two that measurably benefit.")
    args = ap.parse_args()
    values = [CLASS_VALUES[c] for c in args.classes]
    print(f"Filtering {args.classes} with min-frac {args.min_frac}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(args.pred_dir.glob("*.png")):
        groups[re.match(r"(Patient_\d+)_\d+\.png", p.name).group(1)].append(p)

    total_c = total_v = 0
    for patient, files in groups.items():
        vol = np.stack([np.array(Image.open(f)) for f in files], axis=-1)
        filtered, n_comp, n_vox = filter_volume(vol, args.min_frac, values)
        for i, f in enumerate(files):
            Image.fromarray(filtered[:, :, i]).save(args.out_dir / f.name)
        pct = 100 * n_vox / max(1, int((vol > 0).sum()))
        print(f"  {patient}: removed {n_comp} components, {n_vox} voxels ({pct:.2f}% of foreground)")
        total_c += n_comp
        total_v += n_vox

    print(f"Total: {total_c} components, {total_v} voxels removed -> {args.out_dir}")


if __name__ == "__main__":
    main()
