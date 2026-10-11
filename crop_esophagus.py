#!/usr/bin/env python3
"""Native resolution crops around the esophagus, for a two-stage refinement.

Stage 1 is the normal 256x256 model. It loses detail on the thinnest organ:
even a perfect 256 label scaled back to 512 only reaches 0.93 esophagus Dice
(heart 0.99). Stage 2 sees the esophagus at its native 512 resolution, in a
small window around it, and its esophagus replaces the one of stage 1.

Where the window sits on each slice:
    --centers gt     the ground truth esophagus (to train stage 2). In the train
                     split each window is shifted by a random jitter, so stage 2
                     does not learn that the organ is always perfectly centred.
    --centers <dir>  the esophagus predicted by stage 1 (its PNG slices), which
                     is all there is at test time.

Slices inside the esophagus range without a centre of their own take one
interpolated from their neighbours, and --margin slices are added beyond both
ends, so stage 2 can continue an esophagus that stage 1 stopped too early.

    python crop_esophagus.py --source_dir data/segthor_train_full \\
        --dest_dir data/SEGTHOR_crop --splits train val --centers gt
    python crop_esophagus.py --source_dir data/segthor_train_full \\
        --dest_dir data/SEGTHOR_crop_s1seed1 --splits val \\
        --centers results/zscore_unet_k16_aug_seed1/best_epoch/val

Writes <dest>/<split>/img, gt (not for test) and boxes.json, which records the
corner of every window so merge_refine.py can put the crops back.
"""

import argparse
import json
import os
import random
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.io import imsave

from slice_segthor import get_splits, norm_arr_clip_zscore

ESOPHAGUS = 1


def esophagus_mask(patient: str, source: Path, centers: str, shape: tuple) -> np.ndarray:
    """The esophagus that places the windows, boolean (X, Y, Z) at native size."""
    if centers == "gt":
        return np.asarray(nib.load(source / "train" / patient / "GT.nii.gz").dataobj) == ESOPHAGUS

    paths = sorted(Path(centers).glob(f"{patient}_*.png"))
    assert len(paths) == shape[2], (patient, len(paths), shape)
    pred = np.stack([np.array(Image.open(p)) for p in paths], axis=-1) == 63 * ESOPHAGUS
    f = shape[0] // pred.shape[0]
    pred = pred.repeat(f, axis=0).repeat(f, axis=1)

    # Stray false positives would stretch the slice range and pull the centres
    # away. Keep the components of at least 5% of the largest one, after
    # bridging short gaps along z, so a tube that stage 1 broke stays whole.
    bridged = ndi.binary_dilation(pred, structure=np.ones((1, 1, 7), bool))
    labels, n = ndi.label(bridged)
    if n > 1:
        sizes = ndi.sum(pred, labels, range(1, n + 1))
        keep = 1 + np.flatnonzero(sizes >= 0.05 * sizes.max())
        pred &= np.isin(labels, keep)
    return pred


def window_corners(mask: np.ndarray, size: int, margin: int, jitter: int,
                   rng: np.random.Generator) -> dict[int, tuple[int, int]]:
    """Top left corner of the window for every slice that gets one."""
    X, Y, Z = mask.shape
    zs = np.flatnonzero(mask.any(axis=(0, 1)))
    if len(zs) == 0:
        return {}
    cx = np.array([np.argwhere(mask[:, :, z])[:, 0].mean() for z in zs])
    cy = np.array([np.argwhere(mask[:, :, z])[:, 1].mean() for z in zs])

    span = np.arange(max(zs[0] - margin, 0), min(zs[-1] + margin, Z - 1) + 1)
    # np.interp holds the end values, so the margin slices reuse the last centre
    cx = ndi.median_filter(np.interp(span, zs, cx), size=5, mode="nearest")
    cy = ndi.median_filter(np.interp(span, zs, cy), size=5, mode="nearest")

    corners = {}
    for z, x, y in zip(span, cx, cy):
        dx, dy = rng.integers(-jitter, jitter + 1, size=2) if jitter else (0, 0)
        x0 = int(np.clip(round(x) - size // 2 + dx, 0, X - size))
        y0 = int(np.clip(round(y) - size // 2 + dy, 0, Y - size))
        corners[int(z)] = (x0, y0)
    return corners


def crop_patient(patient: str, source: Path, dest: Path, centers: str, test_mode: bool,
                 size: int, margin: int, jitter: int, seed: int) -> dict:
    ct_path = source / "test" / f"{patient}.nii.gz" if test_mode else source / "train" / patient / f"{patient}.nii.gz"
    ct = np.asarray(nib.load(ct_path).dataobj)
    norm = norm_arr_clip_zscore(ct)  # same normalisation as the zscore variant of stage 1
    gt = None if test_mode else np.asarray(nib.load(source / "train" / patient / "GT.nii.gz").dataobj)

    rng = np.random.default_rng([seed, int(patient.split("_")[-1])])
    corners = window_corners(esophagus_mask(patient, source, centers, ct.shape),
                             size, margin, jitter, rng)

    for sub in ["img"] + ([] if test_mode else ["gt"]):
        (dest / sub).mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        for z, (x0, y0) in corners.items():
            box = np.s_[x0:x0 + size, y0:y0 + size, z]
            imsave(dest / "img" / f"{patient}_{z:04d}.png", norm[box])
            if gt is not None:
                imsave(dest / "gt" / f"{patient}_{z:04d}.png", (gt[box] * 63).astype(np.uint8))
    return {str(z): c for z, c in corners.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--dest_dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    parser.add_argument("--centers", default="gt", help="gt, or a folder with the stage 1 PNG predictions")
    parser.add_argument("--size", type=int, default=128, help="window side in native pixels (128 = about 12.5 cm)")
    parser.add_argument("--margin", type=int, default=5, help="extra slices beyond both ends of the esophagus")
    parser.add_argument("--jitter", type=int, default=16, help="max random shift of the train windows, pixels")
    parser.add_argument("--retains", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42, help="the split seed, as in slice.job")
    args = parser.parse_args()
    assert not args.dest_dir.exists(), args.dest_dir
    assert args.centers == "gt" or "test" not in args.splits or Path(args.centers).is_dir()

    random.seed(args.seed)  # get_splits shuffles with the global generator, as in slice_segthor.py
    train_ids, val_ids, test_ids = get_splits(args.source_dir, args.retains, 0)
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}

    for split in args.splits:
        assert not (split == "test" and args.centers == "gt"), "the test set has no labels to centre on"
        dest = args.dest_dir / split
        fn = partial(crop_patient, source=args.source_dir, dest=dest, centers=args.centers,
                     test_mode=split == "test", size=args.size, margin=args.margin,
                     jitter=args.jitter if split == "train" else 0, seed=args.seed)
        cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        with Pool(min(cores, 8)) as pool:  # whole CT volumes in memory, so not too many at once
            boxes = dict(zip(splits[split], pool.map(fn, splits[split])))
        with open(dest / "boxes.json", "w") as f:
            json.dump({"size": args.size, "boxes": boxes}, f)
        print(f"{split}: {sum(map(len, boxes.values()))} windows from {len(boxes)} patients -> {dest}")


if __name__ == "__main__":
    main()
