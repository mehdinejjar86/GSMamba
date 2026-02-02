"""
GS-Mamba Loss Functions

Photometric, perceptual, geometric, Gaussian flow, and uncertainty-based losses.
"""

from .photometric import L1Loss, SSIMLoss, LaplacianLoss
from .perceptual import LPIPSLoss
from .geometric import DepthSmoothLoss, TemporalConsistencyLoss
from .gaussian_flow import GaussianFlowLoss
from .uncertainty import UncertaintyWeighting, ProgressiveUncertaintyWeighting
from .combined import GSMambaLoss, build_loss

__all__ = [
    "L1Loss",
    "SSIMLoss",
    "LaplacianLoss",
    "LPIPSLoss",
    "DepthSmoothLoss",
    "TemporalConsistencyLoss",
    "GaussianFlowLoss",
    "UncertaintyWeighting",
    "ProgressiveUncertaintyWeighting",
    "GSMambaLoss",
    "build_loss",
]
