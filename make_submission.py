"""Assemble the submission folder required by the course readme.

    group-XX/
        test/pred/Patient_41.nii.gz ...
        val/pred/Patient_11.nii.gz ...
        val/gt/Patient_11.nii.gz ...
        val/dice_3d.npz ...
        group-XX.bundle
        bestmodel-group-XX.pkl

Two things this handles that are easy to get wrong:

1. Predictions are saved as 2D PNG slices, but the submission needs 3D NIfTI at
   native resolution. `stitch.py` does that and is invoked here.
2. `metrics_3d.py` writes one array plus separate patient and class lists. The
   readme wants an archive keyed by patient ID, each mapping to a length-K
   array. The two formats are not interchangeable, so the metrics are rewritten.

    python make_submission.py --run-dir results/zscore_seed1 --group 03 \
        --test-pred-dir results/zscore_seed1/best_epoch/test

On Snellius (not a git checkout) the bundle step is skipped: create it on a
laptop with `git bundle create group-XX/group-XX.bundle main`.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

METRIC_NAMES = ["dice_3d", "hd_3d", "hd95_3d", "assd_3d"]


def run(cmd: list[str]):
    print("  $ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def rewrite_metrics(src: Path, dest: Path):
    """Convert {metric, patients, classes} into an archive keyed by patient."""
    data = np.load(src, allow_pickle=True)
    patients = [str(p) for p in data["patients"]]
    per_patient = {p: data["metric"][i] for i, p in enumerate(patients)}
    np.savez(dest, **per_patient)
    return patients, data["metric"].shape


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="The run to submit, e.g. results/zscore_skip_seed1")
    ap.add_argument("--group", required=True, help="Group number, e.g. 03")
    ap.add_argument("--metrics-sub", default="metrics",
                    help="Which metrics folder inside the run to submit (default metrics)")
    ap.add_argument("--gt-root", type=Path, default=Path("data/segthor_train_full/train"))
    ap.add_argument("--test-scan-pattern", default="data/segthor_train_full/test/{id_}.nii.gz",
                    help="Test CT scans, used as the template for the stitched test volumes")
    ap.add_argument("--test-pred-dir", type=Path, default=None,
                    help="Predicted PNG slices for the test set, if it exists yet")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    tag = f"group-{args.group}"
    out = args.out or Path(tag)
    if out.exists():
        sys.exit(f"{out} already exists, remove it first so nothing stale is submitted")
    (out / "val" / "pred").mkdir(parents=True)
    (out / "val" / "gt").mkdir(parents=True)

    val_pred_src = args.run_dir / "best_epoch" / "val_cc"
    if args.metrics_sub == "metrics" or not val_pred_src.exists():
        val_pred_src = args.run_dir / "best_epoch" / "val"
    print(f"Validation predictions from {val_pred_src}")

    print("\n1. Stitching validation predictions into 3D volumes")
    run([sys.executable, "stitch.py",
         "--data_folder", val_pred_src,
         "--dest_folder", out / "val" / "pred",
         "--num_classes", "255",
         "--grp_regex", r"(Patient_\d\d)_\d\d\d\d",
         "--source_scan_pattern", str(args.gt_root / "{id_}" / "GT.nii.gz")])

    patients = sorted(p.stem.replace(".nii", "") for p in (out / "val" / "pred").glob("*.nii.gz"))
    print(f"   {len(patients)} volumes: {patients}")

    print("\n2. Copying matching ground truth")
    for p in patients:
        shutil.copyfile(args.gt_root / p / "GT.nii.gz", out / "val" / "gt" / f"{p}.nii.gz")
    print(f"   copied {len(patients)} ground truth volumes")

    print("\n3. Rewriting metrics into the per patient format the readme requires")
    src_metrics = args.run_dir / args.metrics_sub
    for name in METRIC_NAMES:
        src = src_metrics / f"{name}.npz"
        if not src.exists():
            print(f"   WARNING: {src} missing, skipped")
            continue
        pats, shape = rewrite_metrics(src, out / "val" / f"{name}.npz")
        print(f"   {name}: {len(pats)} patients, {shape[1]} classes each")

    print("\n4. Copying the best model")
    model = args.run_dir / "bestmodel.pkl"
    if model.exists():
        shutil.copyfile(model, out / f"bestmodel-{tag}.pkl")
        print(f"   {model} -> {out / f'bestmodel-{tag}.pkl'}")
    else:
        print(f"   WARNING: {model} not found. Copy it from the cluster.")

    print("\n5. Test set predictions")
    if args.test_pred_dir and args.test_pred_dir.exists():
        (out / "test" / "pred").mkdir(parents=True)
        run([sys.executable, "stitch.py",
             "--data_folder", args.test_pred_dir,
             "--dest_folder", out / "test" / "pred",
             "--num_classes", "255",
             "--grp_regex", r"(Patient_\d\d)_\d\d\d\d",
             "--source_scan_pattern", args.test_scan_pattern])
    else:
        print("   SKIPPED: no test predictions supplied. The submission is incomplete "
              "without them.")

    print("\n6. Git bundle")
    if not Path(".git").exists():
        print("   SKIPPED: not a git checkout, create the bundle on a laptop")
        return
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    run(["git", "bundle", "create", str(out / f"{tag}.bundle"), branch])

    print(f"\nAssembled {out}. Remaining gaps are printed above as WARNING or SKIPPED.")
    print(f"To archive:  tar cf - {out}/ | zstd -T0 -3 > {tag}.tar.zst")


if __name__ == "__main__":
    main()
