"""
Geometric Loss Functions

Depth smoothness, temporal consistency, and other geometric constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class DepthSmoothLoss(nn.Module):
    """
    Edge-aware depth smoothness loss.

    Encourages smooth depth predictions while preserving edges that
    align with image edges.

    Args:
        edge_weight: Weight for edge-aware term (default: 10.0)
    """

    def __init__(self, edge_weight: float = 10.0):
        super().__init__()
        self.edge_weight = edge_weight

    def forward(
            self,
            depth: torch.Tensor,
            image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute edge-aware depth smoothness loss.

        Args:
            depth: Depth map (B, 1, H, W)
            image: RGB image for edge guidance (B, 3, H, W)

        Returns:
            Smoothness loss
        """
        # Depth gradients
        depth_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        depth_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])

        # Image gradients (for edge-awareness)
        image_dx = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), dim=1, keepdim=True)
        image_dy = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), dim=1, keepdim=True)

        # Edge-aware weighting: smooth where image is smooth
        weight_x = torch.exp(-self.edge_weight * image_dx)
        weight_y = torch.exp(-self.edge_weight * image_dy)

        # Weighted smoothness
        smooth_x = depth_dx * weight_x
        smooth_y = depth_dy * weight_y

        return smooth_x.mean() + smooth_y.mean()


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss for Gaussian motion.

    Encourages smooth trajectories for Gaussians across frames.

    Args:
        motion_weight: Weight for motion smoothness (default: 1.0)
    """

    def __init__(self, motion_weight: float = 1.0):
        super().__init__()
        self.motion_weight = motion_weight

    def forward(
            self,
            gaussians_list: List[Dict[str, torch.Tensor]],
            gaussians_interp: Dict[str, torch.Tensor],
            t: float,
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss.

        Encourages the interpolated Gaussian positions to lie on a smooth
        trajectory between the source frames.

        Args:
            gaussians_list: List of Gaussian dicts for each input frame
            gaussians_interp: Interpolated Gaussians at timestep t
            t: Interpolation timestep in [0, 1]

        Returns:
            Temporal consistency loss
        """
        N = len(gaussians_list)

        if N < 2:
            return gaussians_interp['xyz'].new_zeros(())  # Maintains gradient chain


        # Find bounding frames
        idx0 = int(t * (N - 1))
        idx1 = min(idx0 + 1, N - 1)
        alpha = t * (N - 1) - idx0

        # Linear interpolation baseline
        xyz_0 = gaussians_list[idx0]['xyz']
        xyz_1 = gaussians_list[idx1]['xyz']
        xyz_linear = (1 - alpha) * xyz_0 + alpha * xyz_1

        # Loss: deviation from linear path
        xyz_interp = gaussians_interp['xyz']
        motion_loss = F.mse_loss(xyz_interp, xyz_linear)

        return self.motion_weight * motion_loss


class MultiViewConsistencyLoss(nn.Module):
    """
    Multi-view consistency loss.

    Encourages consistent depth predictions across multiple frames
    by checking that reprojected points match.

    Args:
        threshold: Distance threshold for valid correspondences (default: 0.1)
    """

    def __init__(self, threshold: float = 0.1):
        super().__init__()
        self.threshold = threshold

    def forward(
            self,
            gaussians_list: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute multi-view consistency loss.

        For corresponding Gaussians across frames, their 3D positions
        should be consistent (accounting for motion).

        Args:
            gaussians_list: List of Gaussian dicts for each frame

        Returns:
            Consistency loss
        """
        N = len(gaussians_list)

        if N < 2:
            return gaussians_list[0]['xyz'].new_zeros(())  # Maintains gradient chain

        loss = 0.0
        count = 0

        # Compare adjacent frames
        for i in range(N - 1):
            xyz_i = gaussians_list[i]['xyz']  # (B, num_points, 3)
            xyz_j = gaussians_list[i + 1]['xyz']

            # Motion between frames
            motion = xyz_j - xyz_i

            # Penalize large sudden motions (outliers)
            motion_norm = motion.norm(dim=-1)
            outlier_mask = motion_norm > self.threshold

            if outlier_mask.any():
                # Soft penalty for outlier motions
                outlier_loss = (motion_norm[outlier_mask] - self.threshold).mean()
                loss = loss + outlier_loss
                count += 1

        if count > 0:
            loss = loss / count

        return loss


class OpacityRegularizationLoss(nn.Module):
    """
    Opacity regularization to prevent trivial solutions.

    Encourages Gaussians to have meaningful opacities (not all 0 or 1).

    Args:
        target_mean: Target mean opacity (default: 0.5)
        entropy_weight: Weight for entropy term (default: 0.1)
    """

    def __init__(self, target_mean: float = 0.5, entropy_weight: float = 0.1):
        super().__init__()
        self.target_mean = target_mean
        self.entropy_weight = entropy_weight

    def forward(self, opacity: torch.Tensor) -> torch.Tensor:
        """
        Compute opacity regularization loss.

        Args:
            opacity: Gaussian opacities (B, N, 1) or (B, N)

        Returns:
            Regularization loss
        """
        if opacity.dim() == 3:
            opacity = opacity.squeeze(-1)

        # Mean opacity should be around target
        mean_loss = (opacity.mean() - self.target_mean) ** 2

        # Entropy: encourage diverse opacities (not all same value)
        # Binary entropy: -p*log(p) - (1-p)*log(1-p)
        eps = 1e-6
        entropy = -opacity * torch.log(opacity + eps) - (1 - opacity) * torch.log(1 - opacity + eps)
        entropy_loss = -entropy.mean()  # Negative because we want to maximize entropy

        return mean_loss + self.entropy_weight * entropy_loss


class ScaleRegularizationLoss(nn.Module):
    """
    Scale regularization to prevent degenerate Gaussians.

    Prevents Gaussians from becoming too large or too small.

    Args:
        min_scale: Minimum allowed scale (default: 1e-4)
        max_scale: Maximum allowed scale (default: 1.0)
    """

    def __init__(self, min_scale: float = 1e-4, max_scale: float = 1.0):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, scale: torch.Tensor) -> torch.Tensor:
        """
        Compute scale regularization loss.

        Args:
            scale: Gaussian scales (B, N, 3) - in log space or linear

        Returns:
            Regularization loss
        """
        # If scale is in log space, convert
        if scale.min() < 0:
            scale = torch.exp(scale)

        # Penalize scales outside allowed range
        too_small = F.relu(self.min_scale - scale)
        too_large = F.relu(scale - self.max_scale)

        return (too_small ** 2).mean() + (too_large ** 2).mean()
