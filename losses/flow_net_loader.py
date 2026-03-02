"""
VFIMamba Flow Network Loader

Loads a pretrained VFIMamba model for use as flow supervision
in the Gaussian Flow Loss.
"""

import sys
import os
import torch
import torch.nn as nn


def _import_flow_modules():
    """
    Import VFIMamba flow components.

    Preferred source:
      1) Local vendored package: optical_flow_model/*
      2) External VFIMamba checkout fallback
    """
    # 1) Vendored minimal flow package in this repo.
    try:
        from optical_flow_model.feature_extractor import feature_extractor as mamba_extractor
        from optical_flow_model.flow_estimation import MultiScaleFlow as mamba_estimation
        return mamba_extractor, mamba_estimation, "optical_flow_model"
    except Exception as vendored_err:
        vendored_msg = str(vendored_err)

    # 2) Backward-compatible fallback to full VFIMamba repo directory.
    losses_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(losses_dir, ".."))
    candidate_dirs = [
        os.path.join(repo_root, "VFIMamba"),               # preferred: GSMamba/VFIMamba
        os.path.join(repo_root, "..", "VFIMamba"),         # legacy: sibling repo
    ]

    prev_path = sys.path.copy()
    try:
        for vfimamba_dir in candidate_dirs:
            vfimamba_dir = os.path.abspath(vfimamba_dir)
            if not os.path.isdir(vfimamba_dir):
                continue
            sys.path.insert(0, vfimamba_dir)
            try:
                from model.feature_extractor import feature_extractor as mamba_extractor
                from model.flow_estimation import MultiScaleFlow as mamba_estimation
                return mamba_extractor, mamba_estimation, vfimamba_dir
            except Exception:
                # Try next candidate.
                continue
            finally:
                # Remove just-added path entry before trying next.
                if sys.path and sys.path[0] == vfimamba_dir:
                    sys.path.pop(0)
    finally:
        sys.path = prev_path

    raise FileNotFoundError(
        "Could not import VFIMamba flow modules. "
        "Tried local vendored package 'optical_flow_model' and VFIMamba directories: "
        f"{', '.join(os.path.abspath(p) for p in candidate_dirs)}. "
        f"Vendored import error: {vendored_msg}"
    )


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
    # Import VFIMamba components (vendored package first).
    mamba_extractor, mamba_estimation, import_source = _import_flow_modules()

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
    if isinstance(state_dict, dict):
        if 'state_dict' in state_dict and isinstance(state_dict['state_dict'], dict):
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict and isinstance(state_dict['model'], dict):
            state_dict = state_dict['model']

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

    print(
        f"Loaded VFIMamba-{model_size.upper()} flow network from {ckpt_path} "
        f"(source: {import_source})"
    )
    return flow_net
