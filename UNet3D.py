#!/usr/bin/env python3
"""

A small 3D U-Net for patch based segmentation of the CT volumes.

Same interface as ENet.py and UNet.py

Every 2D operation becomes its 3D counterpart, so each filter also looks
at the slices above and below. After slicing to 256x256 the voxels are
roughly 1.8 x 1.8 x 2-2.5 mm, close enough to isotropic that plain 2x2x2
pooling is used at every level.
"""

import torch
import torch.nn as nn
from torch import Tensor


class ConvBlock3D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_dim, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_dim, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Down3D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ConvBlock3D(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.pool(x))


class Up3D(nn.Module):
    def __init__(self, in_dim: int, skip_dim: int, out_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_dim + skip_dim, out_dim)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        # pad if an odd size made the upsampled map one voxel too small
        diff = [s - x_ for s, x_ in zip(skip.shape[2:], x.shape[2:])]
        if any(diff):
            x = nn.functional.pad(x, [diff[2] // 2, diff[2] - diff[2] // 2,
                                      diff[1] // 2, diff[1] - diff[1] // 2,
                                      diff[0] // 2, diff[0] - diff[0] // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet3D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        K: int = kwargs.get("kernels", 16)  # base channel width

        # four downsamplings: patch sizes must be divisible by 16
        self.inc = ConvBlock3D(in_dim, K)
        self.down1 = Down3D(K, K * 2)
        self.down2 = Down3D(K * 2, K * 4)
        self.down3 = Down3D(K * 4, K * 8)
        self.down4 = Down3D(K * 8, K * 16)

        self.up1 = Up3D(K * 16, K * 8, K * 8)
        self.up2 = Up3D(K * 8, K * 4, K * 4)
        self.up3 = Up3D(K * 4, K * 2, K * 2)
        self.up4 = Up3D(K * 2, K, K)

        self.outc = nn.Conv3d(K, out_dim, kernel_size=1)

        print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) with {kwargs}")

    def forward(self, input: Tensor) -> Tensor:
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

    def init_weights(self, *args, **kwargs):
        def init(m):
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm3d) and m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(init)
