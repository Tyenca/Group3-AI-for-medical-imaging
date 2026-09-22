"""Summarize 3D metrics across one or more runs.

Works with a single run (preliminary result) or several seeds (mean and spread).

    python summarize_runs.py results/baseline_minmax_seed1/metrics
    python summarize_runs.py results/baseline_minmax_seed*/metrics
"""

import sys
from pathlib import Path

import numpy as np

METRICS = [("dice_3d", ""), ("hd_3d", " mm"), ("hd95_3d", " mm"), ("assd_3d", " mm")]


def per_organ(metrics_dir: Path, name: str):
    """Mean over patients for each organ, for one run. Returns (organs, values)."""
    data = np.load(metrics_dir / f"{name}.npz", allow_pickle=True)
    arr = data["metric"]  # (patients, classes)
    classes = [str(c) for c in data["classes"]]
    organs = [c for c in classes if c != "background"]
    values = [np.nanmean(arr[:, classes.index(o)]) for o in organs]
    return organs, np.array(values)


def main(dirs: list[Path]):
    for d in dirs:
        if not d.exists():
            sys.exit(f"Not found: {d}")

    print(f"Summarizing {len(dirs)} run(s):")
    for d in dirs:
        print(f"  {d}")

    for name, unit in METRICS:
        organs, _ = per_organ(dirs[0], name)
        stacked = np.stack([per_organ(d, name)[1] for d in dirs])  # (runs, organs)

        print(f"\n=== {name} ===")
        for i, organ in enumerate(organs):
            col = stacked[:, i]
            if len(dirs) == 1:
                print(f"  {organ:<10} {col[0]:.4f}{unit}")
            else:
                spread = col.max() - col.min()
                print(f"  {organ:<10} mean {col.mean():.4f} +/- {col.std():.4f}"
                      f"  (range {spread:.4f}, runs: {', '.join(f'{v:.4f}' for v in col)}){unit}")

    if len(dirs) == 1:
        print("\nOne run only, so there is no spread to report yet. "
              "Run more seeds to tell a real difference apart from run to run noise.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main([Path(a) for a in sys.argv[1:]])
