import argparse
import warnings
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import probs2class, save_images, tqdm_


def predict_2d(net, args, device):
    from dataset import SliceDataset
    from main import img_transform, gt_transform

    K = 5
    data = SliceDataset(args.subset, args.data_dir,
                        img_transform=img_transform,
                        gt_transform=partial(gt_transform, K),
                        context_slices=args.context_slices,
                        position=args.position)
    loader = DataLoader(data, batch_size=8, num_workers=4, shuffle=False)

    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        for batch in tqdm_(loader, desc=f">> Predicting {args.subset}"):
            probs = F.softmax(net(batch['images'].to(device)), dim=1)
            save_images(probs2class(probs) * 63, batch['stems'], args.dest)

    print(f"Saved {len(data)} predictions to {args.dest}")


def predict_3d(net, args, device):
    from dataset3d import load_volumes
    from main3d import predict_volume, save_volume_png

    amp = device.type == 'cuda'
    volumes = load_volumes(args.data_dir, args.subset)
    for v in tqdm_(volumes, desc=f">> Predicting {args.subset} (3d)"):
        img = torch.from_numpy(v["img"].astype("float32") / 255)[None]
        probs = predict_volume(net, img, args.patch, args.overlap, device, amp)
        save_volume_png(probs.argmax(dim=0), v["stems"], args.dest)

    print(f"Saved {sum(len(v['stems']) for v in volumes)} predictions to {args.dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True, help="bestmodel.pkl of a run")
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--subset', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--context_slices', default=1, type=int, choices=[1, 3, 5])
    parser.add_argument('--position', action='store_true',
                        help="The model was trained with --position")
    parser.add_argument('--patch', default=[48, 128, 128], type=int, nargs=3,
                        help="3d models only: sliding window size, as in training")
    parser.add_argument('--overlap', default=0.5, type=float,
                        help="3d models only: sliding window overlap, as in training")
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    net = torch.load(args.model, map_location=device, weights_only=False)
    net.eval()

    if type(net).__name__ == "UNet3D":
        predict_3d(net, args, device)
    else:
        predict_2d(net, args, device)


if __name__ == "__main__":
    main()