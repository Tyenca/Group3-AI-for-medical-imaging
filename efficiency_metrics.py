
import argparse
import time
import torch

from torch.profiler import profile, ProfilerActivity
from ENet import ENet


def count_parameters(model):
    """
    Counts the total number of parameters in a given model.

    Returns: the total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model, x):
    """
    Counts the total number of floating point operations (FLOPs) for a given model and input.

    Args:
        model: the model to analyze.
        x: the input tensor to the model.

    Returns: the total number of FLOPs for the model and input.
    """

    model = model.cpu()
    x = x.cpu()

    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
            model(x)

    total_flops = sum(event.flops for event in prof.key_averages() if event.flops is not None)
    return total_flops

def measure_inference_time(model, x, device, warmup = 20, runs=100):
    """
    Measures the average inference time of a given model on a specified device.

    Args:
        model: the model to analyze.
        x: the input tensor to the model.
        device: the device on which to perform the inference.
        warmup: the number of warm-up iterations.
        runs: the number of runs to average over.

    Returns: the average inference time in milliseconds.
    """
    model = model.to(device)
    x = x.to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()   

    average_seconds = (end - start) / runs
    return average_seconds * 1000

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_channels", type=int, default=1, choices=[1,3,5], 
                        help="1 for 2D, 3 or 5 for 2.5D input")

    parser.add_argument("--image_size", type=int, default=256, 
                        help="Size of the input image (assumed square)")
    parser.add_argument("--runs", type=int, default=100, 
                        help="Number of runs to average over for inference time measurement")
    parser.add_argument("--warmup", type=int, default=20, 
                        help="Number of warm-up iterations before measuring inference time")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    model = ENet(args.input_channels, 5, kernels=8, factor=2)
    model.init_weights()
    model.eval()

    x = torch.randn(1, args.input_channels, args.image_size, args.image_size)

    parameters = count_parameters(model)
    flops = count_flops(model, x)
    inference_time = measure_inference_time(model, x, device, warmup=args.warmup, runs=args.runs)

    print(f"Device: {device}")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Number of parameters: {parameters}")
    print(f"Total FLOPs per slice: {flops/1e9:.2f} GFLOPs")
    print(f"Average inference time: {inference_time:.3f} ms/slice")

if __name__ == "__main__":
    main()

