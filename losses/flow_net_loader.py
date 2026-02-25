"""
VFIMamba Flow Network Loader

Loads a pretrained VFIMamba model for use as flow supervision
in the Gaussian Flow Loss.
"""

import sys
import os
import torch
import torch.nn as nn
from typing import Optional
from functools import partial


def load_vfimamba_flow_net(
    ckpt_path: str,
    model_size: str = "S",
    device: str = "cuda",
) -> nn.Module:
    """
    Load a pretrained VFIMamba model for flow estimation.

    Args:
        ckpt_path: Path to VFIMamba checkpoint (.pkl or .pth)
        model_size: 'S' for VFIMamba_S (F=16) or 'L' for VFIMamba (F=32)
        device: Device to load model on

    Returns:
        VFIMamba MultiScaleFlow model (frozen, eval mode)
    """
    # Add VFIMamba to path temporarily
    vfimamba_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'VFIMamba')
    vfimamba_dir = os.path.abspath(vfimamba_dir)

    if not os.path.isdir(vfimamba_dir):
        raise FileNotFoundError(
            f"VFIMamba directory not found at {vfimamba_dir}. "
            f"Make sure the VFIMamba repo is at the project root."
        )

    # Import VFIMamba components
    prev_path = sys.path.copy()
    sys.path.insert(0, vfimamba_dir)
    try:
        from model.feature_extractor import feature_extractor as mamba_extractor
        from model.flow_estimation import MultiScaleFlow as mamba_estimation
    finally:
        sys.path = prev_path

    # Model config based on size
    if model_size.upper() == "S":
        F, depth = 16, [2, 2, 2, 3, 3]
    elif model_size.upper() == "L":
        F, depth = 32, [2, 2, 2, 3, 3]
    else:
        raise ValueError(f"Unknown model_size: {model_size}. Use 'S' or 'L'.")

    backbone_cfg = {
        'embed_dims': [(2 ** i) * F for i in range(len(depth))],
        'depths': depth,
        'conv_stages': 3,
    }
    flow_cfg = {
        'embed_dims': [(2 ** i) * F for i in range(len(depth))],
        'motion_dims': [0, 0, 0, 8 * F // depth[-2], 16 * F // depth[-1]],
        'depths': depth,
        'num_heads': [8 * (2 ** i) * F // 32 for i in range(len(depth) - 3)],
        'window_sizes': [7, 7],
        'scales': [4 * (2 ** i) for i in range(len(depth) - 2)],
        'hidden_dims': [4 * F for i in range(len(depth) - 3)],
        'c': F,
        'M': False,
        'local_hidden_dims': 4 * F,
        'local_num': 2,
    }

    # Build model
    backbone = mamba_extractor(**backbone_cfg)
    flow_net = mamba_estimation(backbone, **flow_cfg)

    # Load checkpoint
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location='cpu')

    # Handle DDP-wrapped checkpoints (module. prefix)
    cleaned = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
        if "attn_mask" not in k and "HW" not in k
    }

    flow_net.load_state_dict(cleaned, strict=False)

    # Freeze and set to eval
    flow_net = flow_net.to(device)
    flow_net.eval()
    for param in flow_net.parameters():
        param.requires_grad = False

    print(f"Loaded VFIMamba-{model_size.upper()} flow network from {ckpt_path}")
    return flow_net
