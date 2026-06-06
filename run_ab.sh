#!/usr/bin/env bash
# Reproducible baseline-vs-motion A/B for GS-Mamba.
#
# Two MATCHED runs over the full curriculum (vimeo -> X4K N=4/3/2 -> mixed);
# the ONLY difference is the motion flags, so any PSNR/SSIM delta is attributable
# to the novel 3D-Gaussian synthesis.
#
# Usage:
#   bash run_ab.sh base      # control only
#   bash run_ab.sh motion    # motion only
#   bash run_ab.sh both      # both, sequentially (default)
# On SLURM, submit `base` and `motion` as two separate jobs to run them in parallel.
set -euo pipefail

# ============================== EDIT THESE ==============================
NGPU=4                                          # GPUs per node (torchrun --nproc_per_node)
VIMEO=/path/to/vimeo_triplet
X4K=/path/to/X4K
X4KTEST=/path/to/X4K/test
FLOW=/path/to/VFIMamba_S.pkl                    # gflow teacher; set "" to train without gflow
EPOCHS=100
BATCH=2                                         # per-GPU batch
X4K_CROP=                                       # e.g. 512 to crop X4K; empty = native ~768
MOTION_FRAMES_K=0                               # 0 = use all N; set 2 to cap motion memory
GAUSSIAN_FEAT_DIM=32                            # per-Gaussian latent width (motion run)
EVAL_FULL_EVERY=5
X4K_FRACTION=0.3                                # X4K has millions of samples; subsample per epoch
OUTDIR=./outputs
SEED=42
# =======================================================================

RUN="${1:-both}"

flow_args="";    [ -n "$FLOW" ]     && flow_args="--flow_ckpt $FLOW --flow_model_size S"
x4kcrop_args=""; [ -n "$X4K_CROP" ] && x4kcrop_args="--x4k-crop-size $X4K_CROP"

COMMON="--model gsmamba --epochs $EPOCHS --use_curriculum \
  --batch_size $BATCH --num_workers 4 \
  --vimeo_root $VIMEO --x4k_root $X4K --x4k_test_root $X4KTEST \
  $flow_args $x4kcrop_args \
  --eval_full_every $EVAL_FULL_EVERY --x4k_fraction $X4K_FRACTION \
  --output_dir $OUTDIR --seed $SEED"

run_base() {
  echo "=== [A] baseline (default 2-frame blend) -> $OUTDIR/base_full ==="
  torchrun --nproc_per_node="$NGPU" train.py $COMMON --exp_name base_full
}

run_motion() {
  echo "=== [B] motion (feature-augmented 3D synthesis) -> $OUTDIR/motion_full ==="
  torchrun --nproc_per_node="$NGPU" train.py $COMMON --exp_name motion_full \
    --predict-motion --gaussian-feat-dim "$GAUSSIAN_FEAT_DIM" --refine-real-coverage \
    --motion-frames-k "$MOTION_FRAMES_K"
}

case "$RUN" in
  base)   run_base ;;
  motion) run_motion ;;
  both)   run_base; run_motion ;;
  *) echo "usage: bash run_ab.sh [base|motion|both]"; exit 2 ;;
esac

# Resume after preemption: re-run the same line with
#   --resume $OUTDIR/<base_full|motion_full>/latest.pth
