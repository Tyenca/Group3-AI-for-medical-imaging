"""
compare_cc.py
=============

Compares the metrics of a run before post-processing with the metrics after post-processing (`metrics_cc`),
averaged over the validation patients, per organ.

each .npz holds `metric` (patients x classes),
`patients` (the IDs) and `classes` (the class names). The per-patient layout
(one entry per Patient_XX) is accepted too.

Usage
-----
    python compare_cc.py
    python compare_cc.py --before runs/baseline_seed1/metrics --after results/baseline_seed1/metrics_cc
"""

import argparse
import warnings
from pathlib import Path

import numpy as np

METRICS = ["dice_3d", "hd95_3d", "assd_3d", "cldice_3d"]
FIVE = ["background", "esophagus", "heart", "trachea", "aorta"]


def load_metric(path: Path):
    """Returns (patients, values with shape (P, K, D), class names or None), sorted by patient ID."""
    data = np.load(path, allow_pickle=True)
    if "metric" in data.files:
        values = np.asarray(data["metric"], dtype=float)
        patients = [str(p) for p in data["patients"]] if "patients" in data.files else \
            [f"row{i}" for i in range(len(values))]
        names = [str(c) for c in data["classes"]] if "classes" in data.files else None
    else:
        patients = sorted(k for k in data.files if k.startswith("Patient_"))
        values = np.stack([np.asarray(data[p], dtype=float) for p in patients])
        names = None
    order = np.argsort(patients)
    patients = [patients[i] for i in order]
    values = values[order].reshape(len(patients), values.shape[1], -1)  # (P, K, D)
    return patients, values, names


def class_mean(values: np.ndarray) -> np.ndarray:
    """Mean over patients (and extra dimension) per class. NaN is ignored."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # a class that is NaN for every patient
        return np.nanmean(values, axis=(0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=Path, default=Path("runs/baseline_seed1/metrics"))
    ap.add_argument("--after", type=Path, default=Path("results/baseline_seed1/metrics_cc"))
    args = ap.parse_args()

    for m in METRICS:
        fb, fa = args.before / f"{m}.npz", args.after / f"{m}.npz"
        if not fb.exists() or not fa.exists():
            print(f"\n{m}: skipped (missing {fb if not fb.exists() else fa})")
            continue
        pb, vb, nb = load_metric(fb)
        pa, va, _ = load_metric(fa)
        assert pb == pa, f"{m}: the two folders do not contain the same patients: {pb} vs {pa}"
        before, after = class_mean(vb), class_mean(va)
        names = nb if nb and len(nb) == len(before) else FIVE if len(before) == 5 else \
            [f"class{i}" for i in range(len(before))]
        print(f"\n{m}  ({len(pb)} patients)")
        for n, b, a in zip(names, before, after):
            print(f"  {n:11s} before {b:8.3f}   after {a:8.3f}   change {a - b:+8.3f}")


if __name__ == "__main__":
    main()
