# GS-Mamba

**N-frame Video Frame Interpolation via 3D Gaussian Splatting + Mamba.**

## Overview

GS-Mamba is a *novel-synthesis* approach to VFI. Instead of warping with 2D optical flow, it lifts each of N input frames into a **per-pixel 3D Gaussian** cloud, reasons across frames with a **Mamba** state-space model *in the 3D-Gaussian domain*, renders the in-between frame with a differentiable splat renderer, and refines it with a 2D U-Net. The camera is fixed (this is VFI, not novel-view synthesis) — the 3D representation exists to model **object motion, rotation, and self-occlusion** that 2D-flow methods struggle with.

- **Variable N input frames** (N = 2, 3, 4, …), trained via a curriculum (Vimeo → X4K → mixed).
- **Occlusion / disocclusion** handled by depth-compositing the splats.
- **Optional motion synthesis** (`--predict-motion`): each Gaussian predicts its own velocity (and rotation/acceleration), is forward-warped to the query time, and the warped clouds are merged — the mechanism that targets rotating/zooming objects.
- The U-Net refiner is kept at full strength (PSNR/SSIM objective); the coarse render is also deep-supervised so the 3D field carries the synthesis rather than leaning entirely on the U-Net.

## Installation

```bash
pip install -r requirements.txt
```
Training expects CUDA with `diff_gaussian_rasterization` and `mamba_ssm`; both gracefully fall back to slower reference paths when unavailable (e.g. CPU/MPS for smoke tests).

## Quick check

```bash
python smoke_tests.py --datasets none     # builds the model + runs fwd/bwd, incl. the motion path
```

## Training

Entry point is `train.py` (single GPU, or multi-GPU via `torchrun`).

**Vimeo-90K, single GPU**
```bash
python train.py --mode vimeo_only --exp_name my_exp \
    --vimeo_root /path/to/vimeo_triplet
```

**Multi-GPU (DDP), full curriculum**
```bash
torchrun --nproc_per_node=4 train.py \
    --exp_name my_exp --use_curriculum \
    --vimeo_root /path/to/vimeo_triplet \
    --x4k_root /path/to/X4K --x4k_test_root /path/to/X4K/test \
    --flow_ckpt /path/to/VFIMamba_S.pkl --flow_model_size S
```

**X4K with variable N (TEMPO-style)**
```bash
python train.py --mode x4k_only --no_curriculum \
    --x4k_root /path/to/X4K --x4k-steps 7 15 31 --x4k-n-frames 4 3 2
```
`--x4k-steps` and `--x4k-n-frames` are paired (e.g. step=7,N=4 → anchor spacing 8 with 4 anchors).

**Motion synthesis (the novel path — opt-in)**
```bash
python train.py --mode vimeo_only --exp_name motion \
    --predict-motion --gaussian-feat-dim 32 --refine-real-coverage \
    --vimeo_root /path/to/vimeo_triplet
```
Default (no `--predict-motion`) = the per-frame lift + 2-frame blend baseline.

**Baseline-vs-motion A/B** — two matched runs that differ *only* in the motion flags:
```bash
# edit the vars at the top of the script, then run as two jobs:
bash run_ab.sh base      # control
bash run_ab.sh motion    # novel synthesis
```

### Key arguments

| Argument | Description |
|---|---|
| `--mode` | `vimeo_only`, `x4k_only`, or `mixed` |
| `--use_curriculum` / `--no_curriculum` | curriculum (Vimeo → X4K N=4/3/2 → mixed) |
| `--batch_size`, `--epochs`, `--lr` | optimization |
| `--crop_size` / `--x4k-crop-size` | Vimeo / X4K spatial crop (memory lever) |
| `--use_amp` / `--no_amp` | mixed precision |
| `--flow_ckpt`, `--flow_model_size` | VFIMamba flow teacher for the gflow loss (training only) |
| `--predict-motion` | enable 3D-Gaussian motion synthesis |
| `--gaussian-feat-dim` | per-Gaussian latent width mixed in 3D (motion) |
| `--motion-accel` | also predict per-Gaussian acceleration (quadratic path) |
| `--motion-frames-k` | frames to advect+merge at t (`0` = all N; set e.g. `2` to cap memory) |
| `--motion-temporal-tau` | temporal opacity-weight softness (`≤0` → uniform) |
| `--refine-real-coverage` | feed the refiner a rendered coverage/alpha map |
| `--x4k_fraction` | fraction of X4K samples per epoch |
| `--eval_full_every` | run the full benchmark every N epochs |
| `--resume` | resume from a checkpoint |

## Evaluation

```bash
python eval.py --checkpoint outputs/my_exp/best.pth --dataset all \
    --vimeo_root /path/to/vimeo_triplet --x4k_root /path/to/X4K/test
```
`eval.py` rebuilds the exact architecture from the checkpoint's saved config, so motion/feature flags don't need to be re-passed. Datasets: `vimeo`, `x4k`, `all`.

## Architecture

```
N frames ─► SS2D encoder (per-frame, shared weights)
            └─► per-Gaussian head: 14 params  [+ F-dim latent when --predict-motion]
                 └─► 3D Gaussians per frame
                      └─► NFrameGaussianMamba — JOINT 3D-Morton ordering across frames
                          + bidirectional SSM (cross-frame reasoning in Gaussian space)
                          ├─ default: pick the two bounding frames → blend at t
                          └─ motion : per-Gaussian velocity/rotation → forward-warp every
                                      (or K nearest) frame to t → merge (depth-composited)
                               └─► differentiable splat render ─► U-Net refine ─► output
```

The VFIMamba flow network is a **training-only teacher** for the Gaussian-Flow loss; it is not used at inference (the model predicts its own motion).

## Losses

- **Photometric** on the refined output: L1 + SSIM + Laplacian pyramid.
- **Coarse-render deep supervision**: L1/SSIM on the pre-refine render, so the Gaussian field itself produces the interpolated frame.
- **Gaussian-Flow (gflow)**: frozen VFIMamba flow supervises motion early, then decays out (training only).
- **Regularization**: depth smoothness, opacity/scale.
- **Uncertainty weighting** (Kendall et al.) to balance the terms automatically.

## Notes on memory / scale

- X4K's sample index (millions of samples) is stored as a compact numpy array, so multi-worker DDP DataLoaders don't blow up host RAM via copy-on-write.
- For tight GPU memory at native X4K resolution, use `--x4k-crop-size` and/or `--motion-frames-k 2`.
