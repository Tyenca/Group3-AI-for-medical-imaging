#!/bin/bash
# Build the 2x2: preprocessing on/off by postprocessing on/off.
#
#   bash run_postprocessing.sh
#
# Postprocessing runs on predictions that already exist, so this needs no GPU
# and no SLURM job. It filters heart and trachea only, which are the two classes
# measured to benefit, and writes metrics_cc/ next to each run's existing
# metrics/ so the before and after numbers sit side by side.

set -e
cd "$(dirname "$0")"

for RUN in results/baseline_minmax_seed* results/zscore_skip_seed*; do
    [ -d "${RUN}/best_epoch/val" ] || { echo "skip ${RUN}, no predictions"; continue; }

    case "$(basename "${RUN}")" in
        baseline_minmax_*) VARIANT=baseline_minmax ;;
        zscore_skip_*)     VARIANT=zscore_skip ;;
    esac

    # The baseline was sliced before the directory rename, so accept either name.
    SLICE_DIR="data/SEGTHOR_${VARIANT}"
    [ -d "${SLICE_DIR}" ] || SLICE_DIR="data/SEGTHOR"

    echo "=== ${RUN}  (gt from ${SLICE_DIR}) ==="

    python -u postprocess_cc.py \
        --pred-dir "${RUN}/best_epoch/val" \
        --out-dir "${RUN}/best_epoch/val_cc" \
        --classes esophagus heart trachea aorta \
        --min-frac 0.20

    python -u metrics_3d.py \
        --pred-dir "${RUN}/best_epoch/val_cc" \
        --gt-dir "${SLICE_DIR}/val/gt" \
        --spacing-pkl "${SLICE_DIR}/spacing.pkl" \
        --out-dir "${RUN}/metrics_cc" > /dev/null

    echo "    wrote ${RUN}/metrics_cc"
done

echo
echo "Done. Every run now has metrics/ (no postprocessing) and metrics_cc/ (with)."
