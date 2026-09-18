>**NOT THE FINAL README** ⚠️
> This file only documents what's been added so far for the UNet architecture to support internal collaboration.
> It is not the project's final README and will need to be merged/rewritten before submission, 
> once all teams' work is integrated.

# UNet addition status notes

## What was added

- **`UNet.py`**: a new network module, written as a drop in replacement
  for `ENet.py`. It is not derived from ENet's code (different
  architecture family), but matches the same interface so it plugs into
  the existing training loop with no changes to `main.py`'s logic:
  - `__init__(self, in_dim, out_dim, **kwargs)`
  - `init_weights(self)`
  - `forward(self, input)` -> returns logits of shape `[B, out_dim, H, W]`,
    same spatial size as the input.

- **`main.py`**: a small change was made, I added a `--network` CLI flag (`enet` / `unet`) so the
  network can be switched without editing code:
  ```bash
  python main.py --dataset SEGTHOR --network unet --epoch 25 --dest results/segthor/unet --gpu
  ```
  If `--network` is omitted, the dataset's default network (ENet) is
  used, so existing commands/scripts are unaffected.

  **'2.5D input extension'**
  A 2.5D input option was added on op of the existing Unet. We can now use three adjacent slices:
  - previous slice
  - current slice
  - next slice

  These are joined along the channel dimension, resulting in an input tensor of shape '[3, H, W]' instead of '[1, H, W]'. This gives the model its 2.5D context without replacing the 2D training with a full 3D network.

  ## Implementation

  dataset.py
  - added an optional 'context_25d' argument to 'SliceDataset'
  - Added function to retreive adjacent slices from the same patient
  - Current slice is reused if a previous or next slice does not exist
  - Three slices are joined in a 3-channel tensor

  main.py
  - Added a '--context_25d' cl flag
  - When enabled 'SliceDataset' uses 2.5D input
  - The network input dimension is automatically changed from 1 to 3 channels

## Why 2.5D experiment

The original SegTHOR volumes are 3D CT scans but the baseline preprocessing converts them to 2D slices. Using adjacent slices gives the network additional context while keeping the existing 2D UNet almost unchanged. This allows direct comparison between 2D Unet and 2.5D Unet while keeping the training procedure the same.


## Architecture chosen: dilated residual UNet

A UNet (encoder/decoder with skip connections) rather than plain
ENet is being used, with two specific modifications on top of a vanilla standard UNet:

1. **Residual encoder/decoder blocks.** Each conv block adds its input
   back onto its output (`output = conv_block(x) + x`, with a 1x1 conv
   to fix channel counts when needed) instead of a plain feed forward
   block. This helps gradients flow through the network during training
   and makes deep stacks of layers easier to train.

2. **Dilated bottleneck.** At the deepest, most compressed point of the
   network (the "bottleneck"), we use a stack of dilated convolutions
   (dilation rates 1, 2, 4, 8) instead of plain convolutions. Dilation
   lets each layer "see" a wider area of the image without shrinking the
   feature map further and without adding more parameters. This
   enlarges the network's receptive field (how much of the input
   influences a given output location), which is useful for reasoning
   about context around small/thin structures.

### Why this specific design, not just "a UNet"

This approach was selected after reviewing relevant research and comparing the advantages and disadvantages of several alternative architectures. The combination of residual connections and dilated convolutions within a U-Net architecture was considered the most suitable for this segmentation task. Previous findings also indicate that the esophagus is typically the most challenging organ to segment, with the lowest Dice score. Its performance should therefore be monitored particularly closely when comparing our model with the baseline ENet.

### Why we're not using "attention"
Some UNet variants (e.g. "Attention UNet") add a mechanism that lets
the network learn to focus more on certain parts of the image and
ignore others, the idea being it could help the network zero in on
small, hard to spot organs like the esophagus.

It sounds like it should help, but after checking papers that actually
tested this on the SegTHOR dataset specifically, adding attention did
**not** give a reliable improvement. Sometimes a little better,
sometimes not. The dilated + residual design being using instead has
much stronger evidence behind it on this exact dataset (0.887 Dice
reported), so the option that's proven to work here rather
than the one that just sounds promising was chosen.

We could still try adding attention later as a stretch goal if there's
time left after the core UNet is working and evaluated.

## Baseline reference

Baseline **ENet** has already been trained fully on real SEGTHOR:

- Best validation Dice: **0.810** at epoch 23 (up from 0.789)
- Loss: cross-entropy, `--mode full`

**Note on where the actual files are:** `results/` is not pushed to
git (see `.gitignore` training outputs are regenerable artifacts, not
source code, and would bloat the repo). So `results/SEGTHOR/ce` is a
path on the machine that trained it, not something you can just `git
pull` and find. If you need the actual prediction files or trained
weights (not just the Dice number above), check our shared OneDrive.

This Dice number is the team's shared reference point, in case training the ENet from scratch takes too much time to get it on your own pc. Any teammate working on their own improvement (architecture, augmentation, loss, optimizer, post-processing) can compare against this baseline number directly.
This is not part of the UNet task below; it's included here for
reference since UNet's numbers will be compared against it.

##  Tasks completed so far

- [x] Write `UNet.py` with the same interface as `ENet.py`
- [x] Wire it into `main.py` via a `--network` flag
- [x] Confirm correct output shape with a plain torch tensor test
      (`net(torch.randn(2, 1, 256, 256))` → correct `[B, K, H, W]` shape)
- [x] Quick `--debug` run and Full UNet training run on real SEGTHOR
- [x] Add 2.5D adjacent slide input
- [x] Add '--context_25d' flag
- [x] Confirmed 2.5D input shape [3, 256, 256]
- [x] Run full 2.5D Unet training

## How to run it

```bash
# Quick sanity check (few samples, few epochs)
python main.py --dataset SEGTHOR --network unet --epoch 2 --debug --dest results/segthor/unet_test --gpu

# Full run
python main.py --dataset SEGTHOR --network unet --epoch 25 --dest results/segthor/unet --gpu
```

Compare `results/segthor/unet/best_epoch.txt` against the baseline
ENet's `results/segthor/ce/best_epoch.txt` for a quick 2D sanity check
(not the final 3D comparison number).

## How to run 2.5D
python main.py --dataset SEGTHOR --network unet --context_25d --epoch 25 --dest results/segthor/unet_25d --gpu

## References

1. **Vesal S, Ravikumar N, Maier A. A 2D DILATED RESIDUAL U-NET FOR MULTI-ORGAN SEGMENTATION IN THORACIC CT [Internet]. [cited 2026 Sept 15]. Available from: https://eprints.whiterose.ac.uk/id/eprint/149281/1/SegTHOR2019_paper_13.pdf** 
2. **Z. Lambert, C. Petitjean, B. Dubray, S. Ruan. SegTHOR: Segmentation of Thoracic Organs at Risk in CT images [Internet]. 2019 [cited 2026 Sept 15]. Available from: https://arxiv.org/abs/1912.05950v1** 
3. **Manko M, Popov A, Gorriz JM, Ramirez J. Improved organs at risk segmentation based on modified U‐Net with self‐attention and consistency regularisation. CAAI Transactions on Intelligence Technology [Internet]. 2024 Mar 25 [cited 2026 Sept 15];9(4):850–65. Available from: https://digibug.ugr.es/bitstream/handle/10481/91757/CAAI%20Trans%20on%20Intel%20Tech%20-%202024.pdf?sequence=1**