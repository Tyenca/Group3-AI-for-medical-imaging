#!/usr/bin/env python3
import argparse
import pickle
import random
from pathlib import Path
from pprint import pprint
from shutil import rmtree

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

from dataset3d import load_volumes, PatchDataset, VolumeDataset, pad_to
from UNet3D import UNet3D
from losses import CrossEntropy, DiceCE
from utils import class2one_hot, tqdm_

K = 5            
LABEL_MULT = 63  


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def first_item(batch):
    return batch[0]


def as_2d(t: Tensor) -> Tensor:
    b, k, d, h, w = t.shape
    return t.reshape(b, k, d * h, w)


def augment_batch(img: Tensor, gt: Tensor, p: float = 0.5) -> tuple[Tensor, Tensor]:
    
    b = img.shape[0]
    theta = torch.zeros(b, 3, 4, device=img.device)
    theta[:, 0, 0] = theta[:, 1, 1] = theta[:, 2, 2] = 1.0  # identity

    apply = torch.rand(b, device=img.device) < p
    if apply.any():
        n = int(apply.sum())
        angle = torch.empty(n, device=img.device).uniform_(-10, 10).deg2rad()
        scale = torch.empty(n, device=img.device).uniform_(0.9, 1.1)
        shift = torch.empty(n, 2, device=img.device).uniform_(-0.1, 0.1)  # 5% of [-1, 1]
        cos, sin = angle.cos() / scale, angle.sin() / scale
        # affine_grid order is (x, y, z) = (W, H, D); depth stays untouched
        theta[apply, 0, 0], theta[apply, 0, 1] = cos, -sin
        theta[apply, 1, 0], theta[apply, 1, 1] = sin, cos
        theta[apply, 0, 3], theta[apply, 1, 3] = shift[:, 0], shift[:, 1]

    grid = F.affine_grid(theta, list(img.shape), align_corners=False)
    img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    gt = F.grid_sample(gt[:, None].float(), grid, mode='nearest',
                       padding_mode='zeros', align_corners=False)[:, 0].long()
    return img.clamp(0, 1), gt


def window_starts(size: int, patch: int, stride: int) -> list[int]:
    starts = list(range(0, max(size - patch, 0) + 1, stride))
    if starts[-1] != max(size - patch, 0):
        starts.append(max(size - patch, 0))  # always cover the end
    return starts


@torch.no_grad()
def predict_volume(net, img: Tensor, patch, overlap: float, device, amp: bool) -> Tensor:
    
    orig = img.shape[1:]
    img = torch.from_numpy(pad_to(img.numpy(), (1, *patch)))
    shape = img.shape[1:]
    strides = [max(1, int(p * (1 - overlap))) for p in patch]

    probs = torch.zeros((K, *shape), device=device)
    count = torch.zeros(shape, device=device)
    for z in window_starts(shape[0], patch[0], strides[0]):
        for y in window_starts(shape[1], patch[1], strides[1]):
            for x in window_starts(shape[2], patch[2], strides[2]):
                sl = (slice(z, z + patch[0]), slice(y, y + patch[1]), slice(x, x + patch[2]))
                window = img[(slice(None), *sl)][None].to(device)  # (1, 1, d, h, w)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    logits = net(window)
                probs[(slice(None), *sl)] += F.softmax(logits.float(), dim=1)[0]
                count[sl] += 1
    probs /= count
    return probs[:, :orig[0], :orig[1], :orig[2]]


def save_volume_png(pred: Tensor, stems: list[str], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    pred = (pred.cpu().numpy() * LABEL_MULT).astype(np.uint8)
    for z, stem in enumerate(stems):
        Image.fromarray(pred[z]).save(folder / f"{stem}.png")


def setup(args):
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    net = UNet3D(1, K, kernels=args.kernels)
    net.init_weights()
    net.to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f">> {n_params / 1e6:.2f}M parameters")

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    print(f">> Using {args.optimizer} optimizer with learning rate {args.lr}")

    root_dir = args.data_dir if args.data_dir else Path("data") / args.dataset
    train_vols = load_volumes(root_dir, 'train', args.debug)
    val_vols = load_volumes(root_dir, 'val', args.debug)

    for v in train_vols + val_vols:
        assert all(s >= p for s, p in zip(v["img"].shape[1:], args.patch[1:])), \
            f"{v['patient']}: in plane size {v['img'].shape[1:]} smaller than patch {args.patch[1:]}"

    iters = 4 if args.debug else args.iters
    train_set = PatchDataset(train_vols, args.patch, iters * args.batch_size, args.fg_prob)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers,
                              worker_init_fn=worker_init_fn, persistent_workers=False)
    val_loader = DataLoader(VolumeDataset(val_vols), batch_size=1, num_workers=0,
                            collate_fn=first_item)

    args.dest.mkdir(parents=True, exist_ok=True)
    with open(args.dest / "train_val_split.pkl", 'wb') as f:
        pickle.dump({'train': [v["patient"] for v in train_vols],
                     'val': [v["patient"] for v in val_vols],
                     'seed': args.seed}, f, pickle.HIGHEST_PROTOCOL)

    return net, optimizer, device, train_loader, val_loader


def runTraining(args):
    print(f">>> Setting up 3D training on {args.data_dir or args.dataset}")
    net, optimizer, device, train_loader, val_loader = setup(args)
    amp: bool = args.amp and device.type == 'cuda'

    idk = list(range(K))
    match args.loss:
        case 'ce':
            loss_fn = CrossEntropy(idk=idk)
        case 'dicece':
            loss_fn = DiceCE(idk=idk)

    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_loss_tra = torch.zeros((args.epochs, len(train_loader)))
    log_loss_val = torch.zeros((args.epochs, len(val_loader)))
    log_dice3d_val = torch.zeros((args.epochs, len(val_loader), K))

    best_dice: float = 0
    for e in range(args.epochs):
        # ---- training on random patches
        net.train()
        tq_iter = tqdm_(enumerate(train_loader), total=len(train_loader), desc=f">> Training   ({e: 4d})")
        for i, data in tq_iter:
            img = data['images'].to(device)
            gt = data['gts'].to(device)
            if args.augment:
                img, gt = augment_batch(img, gt)
            gt = class2one_hot(gt, K)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits = net(img)
            probs = F.softmax(logits.float(), dim=1)
            loss = loss_fn(as_2d(probs), as_2d(gt))
            loss.backward()
            optimizer.step()

            log_loss_tra[e, i] = loss.item()
            tq_iter.set_postfix({"Loss": f"{log_loss_tra[e, :i + 1].mean():5.2e}"})

        # ---- validation on whole volumes
        net.eval()
        preds: dict[str, tuple[Tensor, list[str]]] = {}
        tq_iter = tqdm_(enumerate(val_loader), total=len(val_loader), desc=f">> Validation ({e: 4d})")
        for i, data in tq_iter:
            probs = predict_volume(net, data['images'], args.patch, args.overlap, device, amp)
            gt = class2one_hot(data['gts'][None].to(device), K)
            with torch.no_grad():
                log_loss_val[e, i] = loss_fn(as_2d(probs[None]), as_2d(gt)).item()

            pred = probs.argmax(dim=0)
            pred_oh = class2one_hot(pred[None], K)
            inter = (pred_oh * gt).sum(dim=(2, 3, 4))[0].float()
            sizes = (pred_oh.sum(dim=(2, 3, 4)) + gt.sum(dim=(2, 3, 4)))[0].float()
            # same convention as main.py: a class absent from both counts as 1
            log_dice3d_val[e, i] = torch.where(sizes > 0, 2 * inter / sizes, 1.0).cpu()
            preds[data['patient']] = (pred.cpu(), data['stems'])
            tq_iter.set_postfix({"Dice3D": f"{log_dice3d_val[e, :i + 1, 1:].mean():05.3f}"})

        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice3d_val.npy", log_dice3d_val)

        if scheduler:
            scheduler.step()

        per_class = log_dice3d_val[e].mean(dim=0)
        print(f"   epoch {e}: 3D Dice " + ", ".join(f"{c}={d:.3f}" for c, d in
              zip(["eso", "heart", "trach", "aorta"], per_class[1:].tolist())))

        current_dice: float = per_class[1:].mean().item()
        if current_dice > best_dice:
            message = f">>> Improved 3d dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(message)

            best_folder = args.dest / "best_epoch" / "val"
            if best_folder.exists():
                rmtree(best_folder)
            for pred, stems in preds.values():
                save_volume_png(pred, stems, best_folder)

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=180, type=int)
    parser.add_argument('--dataset', default='SEGTHOR')
    parser.add_argument('--data_dir', type=Path, default=None,
                        help="Directory holding the sliced data, e.g. data/SEGTHOR_zscore.")
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--loss', default='ce', choices=['ce', 'dicece'])
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="2 train volumes, 1 val volume, 4 iterations per epoch.")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--augment', action='store_true',
                        help="In plane random affine, the same for every slice of a patch.")
    parser.add_argument('--kernels', default=16, type=int, help="Base channel width of the 3D U-Net.")
    parser.add_argument('--patch', default=[48, 128, 128], type=int, nargs=3,
                        help="Patch size D H W. Each must be divisible by 16.")
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--iters', default=250, type=int, help="Training batches per epoch.")
    parser.add_argument('--fg_prob', default=0.66, type=float,
                        help="Fraction of patches centred on an organ voxel.")
    parser.add_argument('--overlap', default=0.5, type=float, help="Sliding window overlap.")
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--scheduler', default='none', choices=['none', 'cosine'])
    parser.add_argument('--workers', default=5, type=int)
    parser.add_argument('--no_amp', dest='amp', action='store_false',
                        help="Disable bfloat16 mixed precision (on by default on GPU).")
    parser.add_argument('--track_co2', action='store_true')
    args = parser.parse_args()

    assert all(p % 16 == 0 for p in args.patch), "patch sizes must be divisible by 16"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    pprint(args)

    if args.track_co2:
        from codecarbon import OfflineEmissionsTracker, OutputMethod
        args.dest.mkdir(parents=True, exist_ok=True)
        tracker = OfflineEmissionsTracker(project_name="Medical Imaging Segmentation", country_iso_code="NLD",
                                          output_dir=str(args.dest), output_file="emissions.csv",
                                          output_methods=[OutputMethod.CSV], measure_power_secs=10)
        tracker.start()
        try:
            runTraining(args)
        finally:
            emissions = tracker.stop()
        if emissions is not None:
            print(f"Total CO2 emissions: {emissions:.6f} kg CO2eq")
    else:
        runTraining(args)


if __name__ == '__main__':
    main()
