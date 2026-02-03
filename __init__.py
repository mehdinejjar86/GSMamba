"""
GS-Mamba: N-Frame Video Interpolation via 3D Gaussian Splatting + Mamba

A novel VFI framework that lifts video frames to 3D Gaussian representations
and uses Mamba (Selective State Space Models) for temporal interpolation.

Usage:
    # Training
    python -m gsmamba.train --exp_name my_experiment
    python run_train.py --exp_name my_experiment

    # Evaluation
    python -m gsmamba.eval --checkpoint best.pth --dataset all
    python run_eval.py --checkpoint best.pth --dataset all
"""

from config import GSMambaConfig, get_config, get_full_config, FullConfig

__version__ = "0.1.0"
__all__ = [
    "GSMambaConfig",
    "get_config",
    "get_full_config",
    "FullConfig",
]
