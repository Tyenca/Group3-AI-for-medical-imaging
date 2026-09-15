# Enhancement of the ENet architecture, follows a 3D residual dilated UNet architecture

# Design motivated by Vesal et al. (SegTHOR 2019), who showed that a
# 2D UNet with dilated convolutions in the bottleneck and residual
# encoder blocks improves esophagus/heart/trachea/aorta segmentation
# on this exact dataset 
# https://ceur-ws.org/Vol-2349/SegTHOR2019_paper_13.pdf

import torch
import torch.nn as nn
from torch import Tensor

from ENet import random_weights_init  # shared generic init helper from Enet.py

class ResidualDoubleConv(nn.Module):

        # Creating two 3x3 convulutional layers with batch normalization and ReLU activation, with a residual connection. 
        # Meaning the model largely just learns the residual of the input, making the network slightly easier to train and more robust to vanishing gradients. 
        # This is an improvement over the standard double convolution used in a standard UNet, as it helps with feature propagation through the encoder.
        def __init__(self, in_dim: int, out_dim: int):
                super().__init__()
                self.block = nn.Sequential(
                        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_dim),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_dim),
                )
                # 1x1 projection for the residual path when channel counts differ
                self.residual_proj = (nn.Conv2d(in_dim, out_dim, kernel_size=1)
                                      if in_dim != out_dim else nn.Identity())
                self.relu_out = nn.ReLU(inplace=True)

        # forward pass for the residual double convolution block. Takes an input tensor, applies the convolutional block, and adds the residual connection before applying the final ReLU activation.
        def forward(self, x: Tensor) -> Tensor:
                residual = self.residual_proj(x)
                out = self.block(x)
                return self.relu_out(out + residual)


class DilatedBottleneck(nn.Module):

        # Class that implements a stack of dilated convolutional layers. The purpose is to deal with the bottleneck of the UNet architecture, 
        # where the spatial resolution of the feature maps is most downsampled. The block also includes batch normalization and ReLU activation after each convolutional layer.
        # It mirrors the dilation pattern ENet itself uses in its BottleNeck blocks, but applied here inside a UNet's bottleneck.
        def __init__(self, in_dim: int, out_dim: int, dilations=(1, 2, 4, 8)):
                super().__init__()
                layers = []
                dim = in_dim
                for d in dilations:
                        layers.append(nn.Conv2d(dim, out_dim, kernel_size=3,
                                                padding=d, dilation=d))
                        layers.append(nn.BatchNorm2d(out_dim))
                        layers.append(nn.ReLU(inplace=True))
                        dim = out_dim
                self.block = nn.Sequential(*layers)

                self.residual_proj = (nn.Conv2d(in_dim, out_dim, kernel_size=1)
                                      if in_dim != out_dim else nn.Identity())
                self.relu_out = nn.ReLU(inplace=True)

        # forward pass for the dilated bottleneck block. Takes an input tensor, applies the dilated convolutional block, and adds the residual connection before applying the final ReLU activation.
        def forward(self, x: Tensor) -> Tensor:
                residual = self.residual_proj(x)
                out = self.block(x)
                return self.relu_out(out + residual)


class Down(nn.Module):
      
        # This class implements the downsampling operation in the UNet architecture. 
        # It first applies a max pooling operation to reduce the spatial dimensions of the input feature map, 
        # followed by a residual double convolution block to extract features from the downsampled representation.
        def __init__(self, in_dim: int, out_dim: int):
                super().__init__()
                self.pool = nn.MaxPool2d(2)
                self.conv = ResidualDoubleConv(in_dim, out_dim)

        def forward(self, x: Tensor) -> Tensor:
                return self.conv(self.pool(x))


class Up(nn.Module):

        # This class implements the upsampling operation in the UNet architecture.
        def __init__(self, in_dim: int, skip_dim: int, out_dim: int):
                super().__init__()
                self.up = nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2)
                self.conv = ResidualDoubleConv(in_dim // 2 + skip_dim, out_dim)

        def forward(self, x: Tensor, skip: Tensor) -> Tensor:
                x = self.up(x)

                # Handlels potential size mismatches from odd input dimensions (pool or upsample rounding) by padding rather than crashing.
                diffY = skip.size(2) - x.size(2)
                diffX = skip.size(3) - x.size(3)
                if diffY != 0 or diffX != 0:
                        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                                  diffY // 2, diffY - diffY // 2])

                x = torch.cat([skip, x], dim=1)
                return self.conv(x)


class UNet(nn.Module):

        # Replacement for ENet, it uses the same constructor signature,  init_weights() and forward() contract to output [B, out_dim, H, W]).
        def __init__(self, in_dim: int, out_dim: int, **kwargs):
                super().__init__()
                K: int = kwargs.get("kernels", 16)  # base channel width

                # Encoder for residual blocks (small stack of layers)
                self.inc = ResidualDoubleConv(in_dim, K)
                self.down1 = Down(K, K * 2)
                self.down2 = Down(K * 2, K * 4)
                self.down3 = Down(K * 4, K * 8)

                # Pools the output of the last downsampling layer to reduce its spatial dimensions by a factor of 2, preparing it for the bottleneck.
                self.pool4 = nn.MaxPool2d(2)
                self.bottleneck = DilatedBottleneck(K * 8, K * 16, dilations=(1, 2, 4, 8))

                # Decoder for residual blocks, with skip connections from the encoder
                self.up1 = Up(K * 16, K * 8, K * 8)
                self.up2 = Up(K * 8, K * 4, K * 4)
                self.up3 = Up(K * 4, K * 2, K * 2)
                self.up4 = Up(K * 2, K, K)

                # Final 1x1 conv to map to class logits
                self.outc = nn.Conv2d(K, out_dim, kernel_size=1)

                print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) with {kwargs}")

        def forward(self, input: Tensor) -> Tensor:
                x1 = self.inc(input)
                x2 = self.down1(x1)
                x3 = self.down2(x2)
                x4 = self.down3(x3)

                x5 = self.bottleneck(self.pool4(x4))

                x = self.up1(x5, x4)
                x = self.up2(x, x3)
                x = self.up3(x, x2)
                x = self.up4(x, x1)

                return self.outc(x)

        def init_weights(self, *args, **kwargs):
                self.apply(random_weights_init)
