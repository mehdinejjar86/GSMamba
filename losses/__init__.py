"""
GS-Mamba Loss Functions

Photometric, perceptual, geometric, Gaussian flow, and uncertainty-based losses.
"""

from losses.photometric import L1Loss, SSIMLoss, LaplacianLoss
from losses.perceptual import LPIPSLoss
from losses.geometric import DepthSmoothLoss, TemporalConsistencyLoss
from losses.gaussian_flow import GaussianFlowLoss
from losses.uncertainty import UncertaintyWeighting, ProgressiveUncertaintyWeighting
from losses.combined import GSMambaLoss, build_loss

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
