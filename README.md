# GS-Mamba

N-Frame Video Frame Interpolation via 3D Gaussian Splatting + Mamba

## Overview

GS-Mamba lifts N consecutive video frames to 3D Gaussian representations and uses Mamba (Selective State Space Models) for temporal interpolation. Unlike traditional 2-frame VFI methods that rely on 2D optical flow, GS-Mamba:

- Handles **variable N input frames** (N=2, 3, 4, ...)
- Naturally handles **occlusion** via 3D representation
- Captures **non-linear motion** through learned residuals

## Installation

```bash
pip install -r requirements.txt
```

## Training

### Single GPU (Vimeo-90K)
```bash
python -m gsmamba.train --exp_name my_exp --vimeo_root /path/to/vimeo_triplet
```

### Multi-GPU (DDP)
```bash
torchrun --nproc_per_node=4 -m gsmamba.train \
    --exp_name my_exp \
    --vimeo_root /path/to/vimeo_triplet
```

### X4K with Variable N-Frames (TEMPO-style)
```bash
python -m gsmamba.train --mode x4k_only \
    --x4k_root /path/to/x4k \
    --x4k-steps 7 15 31 \
    --x4k-n-frames 4 3 2 \
    --no_curriculum
```

The `--x4k-steps` and `--x4k-n-frames` are paired:
- step=7, n_frames=4 → anchor spacing of 8 frames, use 4 anchors
- step=15, n_frames=3 → anchor spacing of 16 frames, use 3 anchors
- step=31, n_frames=2 → anchor spacing of 32 frames, use 2 anchors

### Key Training Arguments

| Argument | Description |
|----------|-------------|
| `--exp_name` | Experiment name (creates output folder) |
| `--mode` | `vimeo_only`, `x4k_only`, or `mixed` |
| `--batch_size` | Batch size per GPU |
| `--epochs` | Number of training epochs |
| `--lr` | Learning rate |
| `--use_amp` / `--no_amp` | Enable/disable mixed precision |
| `--use_curriculum` / `--no_curriculum` | Enable/disable curriculum learning |
| `--eval_full_every` | Run full benchmark every N epochs |

## Evaluation

```bash
python -m gsmamba.eval \
    --checkpoint outputs/my_exp/best.pth \
    --dataset all \
    --vimeo_root /path/to/vimeo_triplet \
    --x4k_root /path/to/x4k/test
```

Datasets: `vimeo`, `x4k`, `all`

## Architecture

```
N Frames → SS2D Encoder → Temporal Mamba Fusion → Per-Frame Gaussians
                                ↓
Query timestep t → Gaussian Interpolator → Differentiable Rendering → Refinement → Output
```

## Loss Functions

- **Photometric**: L1, SSIM, Laplacian pyramid
- **Gaussian Flow**: 2D flow supervision (decays during training)
- **Regularization**: Depth smoothness, temporal consistency, opacity/scale
- **Uncertainty weighting**: Learns optimal loss weights automatically (Kendall et al.)
