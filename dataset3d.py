#!/usr/bin/env python3

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset import make_dataset

LABEL_MULT = 63  


def load_volumes(root, subset: str, debug: bool = False) -> list[dict]:
    """Group the slices of each patient and stack them along z.

    """
    per_patient: dict[str, list] = defaultdict(list)
    for img_path, gt_path in make_dataset(root, subset):
        patient, idz = img_path.stem.rsplit('_', 1)
        per_patient[patient].append((int(idz), img_path, gt_path))

    patients = sorted(per_patient)
    if debug:
        patients = patients[:2] if subset == 'train' else patients[:1]

    volumes = []
    for patient in patients:
        rows = sorted(per_patient[patient])  # z order
        img = np.stack([np.array(Image.open(p).convert('L')) for _, p, _ in rows])
        gt = None
        if rows[0][2] is not None:
            gt = np.stack([np.rint(np.array(Image.open(g)) / LABEL_MULT) for _, _, g in rows])
            gt = gt.astype(np.uint8)
        volumes.append({"patient": patient,
                        "img": img,
                        "gt": gt,
                        "stems": [p.stem for _, p, _ in rows]})
    print(f">> Loaded {len(volumes)} {subset} volumes, "
          f"{sum(len(v['stems']) for v in volumes)} slices")
    return volumes


def pad_to(arr: np.ndarray, shape) -> np.ndarray:
    pad = [(0, max(0, s - a)) for a, s in zip(arr.shape, shape)]
    return np.pad(arr, pad) if any(p[1] for p in pad) else arr


class PatchDataset(Dataset):

    def __init__(self, volumes: list[dict], patch_size, samples_per_epoch: int,
                 fg_prob: float = 0.66):
        self.volumes = volumes
        self.patch = tuple(patch_size)
        self.samples_per_epoch = samples_per_epoch
        self.fg_prob = fg_prob

        # organ voxel coordinates per volume, thinned out to keep memory low
        self.fg_coords = []
        for v in volumes:
            coords = np.argwhere(v["gt"] > 0)
            self.fg_coords.append(coords[::20] if len(coords) else coords)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        i = np.random.randint(len(self.volumes))
        img = pad_to(self.volumes[i]["img"], self.patch)
        gt = pad_to(self.volumes[i]["gt"], self.patch)
        fg = self.fg_coords[i]

        if len(fg) and np.random.rand() < self.fg_prob:
            centre = fg[np.random.randint(len(fg))]
            # random offset so the organ is not always dead centre
            start = [c - p // 2 + np.random.randint(-p // 4, p // 4 + 1)
                     for c, p in zip(centre, self.patch)]
        else:
            start = [np.random.randint(0, s - p + 1) for s, p in zip(img.shape, self.patch)]
        start = [int(np.clip(st, 0, s - p)) for st, s, p in zip(start, img.shape, self.patch)]

        sl = tuple(slice(st, st + p) for st, p in zip(start, self.patch))
        img_p = torch.from_numpy(img[sl].astype(np.float32) / 255)[None]  # (1, D, H, W)
        gt_p = torch.from_numpy(gt[sl].astype(np.int64))                  # (D, H, W)
        return {"images": img_p, "gts": gt_p}


class VolumeDataset(Dataset):

    def __init__(self, volumes: list[dict]):
        self.volumes = volumes

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, index):
        v = self.volumes[index]
        out = {"images": torch.from_numpy(v["img"].astype(np.float32) / 255)[None],
               "patient": v["patient"],
               "stems": v["stems"]}
        if v["gt"] is not None:
            out["gts"] = torch.from_numpy(v["gt"].astype(np.int64))
        return out
