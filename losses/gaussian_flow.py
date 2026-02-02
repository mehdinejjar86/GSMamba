"""
Gaussian Flow Loss

2D optical flow supervision for 3D Gaussian motion during early training.
Helps Gaussians converge faster by providing motion direction hints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class GaussianFlowLoss(nn.Module):
    """
    Gaussian Flow Loss.

    Supervises 3D Gaussian motion using 2D optical flow from a pretrained
    flow network. The idea is to:
    1. Predict 3D Gaussian motion between frames
    2. Project this 3D motion to 2D
    3. Compare with optical flow from a flow estimator

    This provides a bootstrap signal for Gaussian motion learning.
    The loss weight should decay over training as the 3D representation
    learns to transcend 2D flow limitations.

    Args:
        flow_net: Pretrained flow network (VFIMamba's MultiScaleFlow)
        image_size: Input image size for projection
        focal_length: Camera focal length (default: computed from image_size)
    """

    def __init__(
            self,
            flow_net: Optional[nn.Module] = None,
            image_size: Tuple[int, int] = (256, 256),
            focal_length: Optional[float] = None,
    ):
        super().__init__()

        self.flow_net = flow_net
        if flow_net is not None:
            self.flow_net.eval()
            for param in self.flow_net.parameters():
                param.requires_grad = False

        self.H, self.W = image_size
        self.focal_length = focal_length if focal_length else (self.H + self.W) / 2

        # Create pixel grid for projection
        y, x = torch.meshgrid(
            torch.arange(self.H, dtype=torch.float32),
            torch.arange(self.W, dtype=torch.float32),
            indexing='ij'
        )
        self.register_buffer('grid_x', x)
        self.register_buffer('grid_y', y)

    def _project_to_2d(
            self,
            xyz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project 3D points to 2D pixel coordinates.

        Args:
            xyz: 3D points (B, N, 3)

        Returns:
            2D pixel coordinates (B, N, 2)
        """
        # Simple pinhole projection
        # x_pixel = x * focal / z + cx
        # y_pixel = y * focal / z + cy
        z = xyz[..., 2:3].clamp(min=1e-6)
        x = xyz[..., 0:1] / z * self.focal_length + self.W / 2
        y = xyz[..., 1:2] / z * self.focal_length + self.H / 2

        return torch.cat([x, y], dim=-1)

    def _compute_flow_from_vfimamba(
            self,
            img0: torch.Tensor,
            img1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute optical flow using VFIMamba's flow network.

        Args:
            img0: First frame (B, 3, H, W)
            img1: Second frame (B, 3, H, W)

        Returns:
            Optical flow (B, 2, H, W)
        """
        if self.flow_net is None:
            raise ValueError("Flow network not provided")

        with torch.no_grad():
            # VFIMamba expects concatenated input
            x = torch.cat([img0, img1], dim=1)  # (B, 6, H, W)

            # Forward pass returns (flow_list, mask_list, merged, pred)
            flow_list, _, _, _ = self.flow_net(x)

            # Get finest scale flow (forward direction)
            flow = flow_list[-1][:, :2]  # (B, 2, H, W)

        return flow

    def _sample_flow_at_gaussians(
            self,
            flow: torch.Tensor,
            pixel_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample optical flow at Gaussian pixel locations.

        Args:
            flow: Optical flow (B, 2, H, W)
            pixel_coords: Pixel coordinates (B, N, 2)

        Returns:
            Sampled flow (B, N, 2)
        """
        B, N, _ = pixel_coords.shape
        H, W = flow.shape[2:]

        # Normalize coordinates to [-1, 1] for grid_sample
        grid = pixel_coords.clone()
        grid[..., 0] = 2 * grid[..., 0] / (W - 1) - 1
        grid[..., 1] = 2 * grid[..., 1] / (H - 1) - 1
        grid = grid.view(B, N, 1, 2)  # (B, N, 1, 2)

        # Sample
        sampled = F.grid_sample(
            flow, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # (B, 2, N, 1)

        return sampled.squeeze(-1).permute(0, 2, 1)  # (B, N, 2)

    def forward(
            self,
            gaussians_0: Dict[str, torch.Tensor],
            gaussians_1: Dict[str, torch.Tensor],
            img0: torch.Tensor,
            img1: torch.Tensor,
            use_precomputed_flow: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Gaussian Flow loss.

        Args:
            gaussians_0: Gaussians at frame 0
            gaussians_1: Gaussians at frame 1
            img0: Image at frame 0 (B, 3, H, W)
            img1: Image at frame 1 (B, 3, H, W)
            use_precomputed_flow: Optional precomputed flow (B, 2, H, W)

        Returns:
            Gaussian Flow loss
        """
        # Get 3D positions
        xyz_0 = gaussians_0['xyz']  # (B, N, 3)
        xyz_1 = gaussians_1['xyz']  # (B, N, 3)

        # Compute 3D motion
        delta_3d = xyz_1 - xyz_0  # (B, N, 3)

        # Project 3D motion to 2D
        # For small motions: delta_2d ≈ delta_xy * focal / z
        z_0 = xyz_0[..., 2:3].clamp(min=1e-6)
        delta_2d_pred = delta_3d[..., :2] * self.focal_length / z_0  # (B, N, 2)

        # Get optical flow
        if use_precomputed_flow is not None:
            flow_2d = use_precomputed_flow
        else:
            flow_2d = self._compute_flow_from_vfimamba(img0, img1)

        # Project Gaussian centers to get pixel coordinates
        pixel_coords = self._project_to_2d(xyz_0)  # (B, N, 2)

        # Sample flow at Gaussian locations
        flow_at_gaussians = self._sample_flow_at_gaussians(flow_2d, pixel_coords)  # (B, N, 2)

        # Loss: predicted 2D motion should match optical flow
        # Use L1 loss for robustness
        loss = F.l1_loss(delta_2d_pred, flow_at_gaussians)

        return loss


def get_gflow_weight(epoch: int, total_epochs: int, max_weight: float = 0.1, decay_fraction: float = 0.5) -> float:
    """
    Compute Gaussian Flow loss weight with cosine decay.

    The weight starts at max_weight and decays to 0 over the first
    decay_fraction of training.

    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
        max_weight: Maximum weight at start of training
        decay_fraction: Fraction of training over which to decay

    Returns:
        Current loss weight
    """
    decay_epochs = int(total_epochs * decay_fraction)

    if epoch >= decay_epochs:
        return 0.0

    # Cosine decay
    progress = epoch / decay_epochs
    weight = 0.5 * (1 + math.cos(math.pi * progress)) * max_weight

    return weight


class GaussianFlowLossWithSchedule(nn.Module):
    """
    Gaussian Flow Loss with built-in weight scheduling.

    Wrapper around GaussianFlowLoss that handles the decay schedule.

    Args:
        flow_net: Pretrained flow network
        image_size: Input image size
        max_weight: Maximum loss weight
        decay_fraction: Fraction of training for decay
    """

    def __init__(
            self,
            flow_net: Optional[nn.Module] = None,
            image_size: Tuple[int, int] = (256, 256),
            max_weight: float = 0.1,
            decay_fraction: float = 0.5,
    ):
        super().__init__()

        self.gflow_loss = GaussianFlowLoss(flow_net, image_size)
        self.max_weight = max_weight
        self.decay_fraction = decay_fraction
        self.current_epoch = 0
        self.total_epochs = 100  # Will be updated by trainer

    def set_training_info(self, current_epoch: int, total_epochs: int):
        """Update training progress information."""
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs

    def forward(
            self,
            gaussians_0: Dict[str, torch.Tensor],
            gaussians_1: Dict[str, torch.Tensor],
            img0: torch.Tensor,
            img1: torch.Tensor,
            use_precomputed_flow: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute weighted Gaussian Flow loss.

        Returns:
            Tuple of (weighted_loss, current_weight)
        """
        weight = get_gflow_weight(
            self.current_epoch,
            self.total_epochs,
            self.max_weight,
            self.decay_fraction
        )

        if weight < 1e-6:
            return torch.tensor(0.0, device=img0.device), 0.0

        loss = self.gflow_loss(gaussians_0, gaussians_1, img0, img1, use_precomputed_flow)

        return weight * loss, weight
