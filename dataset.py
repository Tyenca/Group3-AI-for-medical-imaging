#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
from typing import Callable, Union

from torch import Tensor
from PIL import Image
import torch
from torch.utils.data import Dataset


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != 'test':
        full_labels = sorted(full_path.glob("*.png"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False, context_25d=False):  ## Added context_25d argument
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize
        self.test_mode: bool = subset == 'test'
        # Optional 2.5D input using adjacent slices
        self.context_25d: bool = context_25d

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    # Added method to get adjacent slices for 2.5D context. Returns the path of adjacent slices from the same patient. If slice does not exist, the current slice is reused.
    def _get_adjacent_slices(self, img_path: Path, offset: int) -> Path:
        patient_id, slice_str = img_path.stem.split('_', 1)
        slice_index = int(slice_str)
        adjacent_index = slice_index + offset

        if adjacent_index < 0:
            return img_path  

        adjacent_path = img_path.parent / f"{patient_id}_{adjacent_index:04d}.png"

        if not adjacent_path.exists():
            return img_path

        return adjacent_path

    

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        if self.context_25d:
            prev_slice_path = self._get_adjacent_slices(img_path, -1)
            next_slice_path = self._get_adjacent_slices(img_path, 1)

            img_prev = self.img_transform(Image.open(prev_slice_path))
            img_current = self.img_transform(Image.open(img_path))
            img_next = self.img_transform(Image.open(next_slice_path))

            # Stack the three slices as three input channels
            img = torch.cat([img_prev, img_current, img_next], dim=0)

        else:
            img: Tensor = self.img_transform(Image.open(img_path))

        data_dict = {"images": img,
                     "stems": img_path.stem}

        if not self.test_mode:
            gt: Tensor = self.gt_transform(Image.open(gt_path))

            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            data_dict["gts"] = gt

        return data_dict
