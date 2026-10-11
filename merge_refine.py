#!/usr/bin/env python3
"""Put the stage 2 esophagus back into the stage 1 prediction, at native size.

    python merge_refine.py --stage1 results/zscore_unet_k16_aug_seed1/best_epoch/val \\
        --stage2 results/refine_seed1/stage2 \\
        --boxes data/SEGTHOR_crop_s1seed1/val/boxes.json --dest results/refine_seed1

Writes two folders of 512x512 PNG slices, upsampled the same way, so that the
only difference between them is the refinement:
    <dest>/stage1   stage 1 alone
    <dest>/refined  inside each window the esophagus comes from stage 2: pixels
                    stage 1 called esophagus get stage 2's label, and every
                    pixel stage 2 calls esophagus becomes esophagus. The other
                    organs and everything outside the windows stay stage 1's.
Score both with metrics_3d.py --gt-nifti-root, against the native labels.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.io import imsave

ESOPHAGUS = 63  # class 1 in the *63 PNG encoding


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1", type=Path, required=True, help="stage 1 PNG predictions (256)")
    parser.add_argument("--stage2", type=Path, required=True, help="stage 2 PNG predictions of the windows")
    parser.add_argument("--boxes", type=Path, required=True, help="boxes.json of the windows")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--native", type=int, default=512)
    args = parser.parse_args()

    meta = json.loads(args.boxes.read_text())
    size, boxes = meta["size"], meta["boxes"]
    for sub in ["stage1", "refined"]:
        (args.dest / sub).mkdir(parents=True, exist_ok=True)

    changed = 0
    paths = sorted(args.stage1.glob("*.png"))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        for p in paths:
            s1 = np.array(Image.open(p))
            f = args.native // s1.shape[0]
            s1 = s1.repeat(f, axis=0).repeat(f, axis=1)
            imsave(args.dest / "stage1" / p.name, s1)

            patient, z = p.stem.rsplit("_", 1)
            corner = boxes.get(patient, {}).get(str(int(z)))
            if corner is not None:
                x0, y0 = corner
                s2 = np.array(Image.open(args.stage2 / p.name))
                assert s2.shape == (size, size), (p.name, s2.shape)
                win = s1[x0:x0 + size, y0:y0 + size].copy()
                was = win == ESOPHAGUS
                win[was] = s2[was]
                win[s2 == ESOPHAGUS] = ESOPHAGUS
                changed += (win != s1[x0:x0 + size, y0:y0 + size]).sum()
                s1[x0:x0 + size, y0:y0 + size] = win
            imsave(args.dest / "refined" / p.name, s1)

    print(f"Merged {len(paths)} slices, {sum(map(len, boxes.values()))} windows, "
          f"{changed} pixels changed -> {args.dest}")


if __name__ == "__main__":
    main()
