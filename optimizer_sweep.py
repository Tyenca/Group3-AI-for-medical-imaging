import sys
import subprocess
from pathlib import Path


SWEEP_CONFIGS = [
    {'optimizer': 'adam', 'lr': 0.0005},
    {'optimizer': 'adam', 'lr': 0.001},
    {'optimizer': 'adamw', 'lr': 0.0005},
    {'optimizer': 'adamw', 'lr': 0.001},
    {'optimizer': 'sgd', 'lr': 0.01}
]

for config in SWEEP_CONFIGS:
    optimizer = config['optimizer']
    lr = config['lr']

    run_name = f"optimizer_{optimizer}_lr_{lr}"
    log_path = Path("results") / "optimizer_sweep" / run_name

    command = [
        sys.executable, "main.py",
        "--log_path", str(log_path),
        "--optimizer", optimizer,
        "--lr", str(lr),

        # baseline
        "--dataset", "SEGTHOR",
        "--network", "enet",
        "--context_slices", "1",
        "--loss", "ce",
        "--sampling", "uniform",    
        "--seed", "0",
        "--mode", "full",

        "--epochs", "10",
    ]


    print(f"\nStart sweep run: {run_name}")
    print(" ".join(command))
    subprocess.run(command, check=True)