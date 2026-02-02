"""
Gaussian Prediction Head

Predicts per-pixel 3D Gaussian parameters from fused features.
Each pixel produces a 3D Gaussian with position, scale, rotation, opacity, and color.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class GaussianHead(nn.Module):
    """
    Per-pixel Gaussian prediction head.

    Predicts 3D Gaussian parameters for each pixel from feature maps.

    Output channels (11 total):
        - depth (1): Inverse depth (disparity) - positive via softplus
        - depth_scale (1): Gaussian extent in depth direction - positive via softplus
        - xy_offset (2): Sub-pixel position offset - unbounded (tanh for stability)
        - scale_xy (2): Gaussian extent in image plane - positive via softplus
        - rotation (1): 2D rotation angle (simplified from quaternion)
        - color (3): RGB color - [0,1] via sigmoid
        - opacity (1): Alpha transparency - [0,1] via sigmoid

    Args:
        in_channels: Input feature channels
        hidden_channels: Hidden layer channels (default: in_channels)
        init_depth: Initial depth value bias (default: 1.0)
        init_scale: Initial scale value bias (default: 0.01)
        init_opacity: Initial opacity logit (default: -2.0, ~0.12 after sigmoid)
    """

    # Channel indices for output parsing
    DEPTH_IDX = 0
    DEPTH_SCALE_IDX = 1
    XY_OFFSET_START = 2
    XY_OFFSET_END = 4
    SCALE_XY_START = 4
    SCALE_XY_END = 6
    ROTATION_IDX = 6
    COLOR_START = 7
    COLOR_END = 10
    OPACITY_IDX = 10

    OUT_CHANNELS = 11

    def __init__(
            self,
            in_channels: int,
            hidden_channels: Optional[int] = None,
            init_depth: float = 1.0,
            init_scale: float = 0.01,
            init_opacity: float = -2.0,
    ):
        super().__init__()

        hidden_channels = hidden_channels or in_channels

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, self.OUT_CHANNELS, 1)

        # Initialize biases for stable training
        self._init_biases(init_depth, init_scale, init_opacity)

    def _init_biases(self, init_depth: float, init_scale: float, init_opacity: float):
        """Initialize output biases for reasonable initial predictions."""
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        with torch.no_grad():
            # Depth: softplus(x) ≈ init_depth when x ≈ log(exp(init_depth) - 1)
            self.conv2.bias[self.DEPTH_IDX] = math.log(math.exp(init_depth) - 1 + 1e-6)

            # Depth scale
            self.conv2.bias[self.DEPTH_SCALE_IDX] = math.log(math.exp(init_scale) - 1 + 1e-6)

            # XY offset: start at 0
            self.conv2.bias[self.XY_OFFSET_START:self.XY_OFFSET_END] = 0

            # Scale XY
            self.conv2.bias[self.SCALE_XY_START:self.SCALE_XY_END] = math.log(math.exp(init_scale) - 1 + 1e-6)

            # Rotation: start at 0
            self.conv2.bias[self.ROTATION_IDX] = 0

            # Color: sigmoid(0) = 0.5 (gray)
            self.conv2.bias[self.COLOR_START:self.COLOR_END] = 0

            # Opacity: sigmoid(init_opacity)
            self.conv2.bias[self.OPACITY_IDX] = init_opacity

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict Gaussian parameters from features.

        Args:
            features: Input features (B, C, H, W)

        Returns:
            Dictionary with Gaussian parameters:
                - depth: (B, 1, H, W) inverse depth
                - depth_scale: (B, 1, H, W) depth-direction scale
                - xy_offset: (B, 2, H, W) sub-pixel offset
                - scale_xy: (B, 2, H, W) in-plane scale
                - rotation: (B, 1, H, W) rotation angle
                - color: (B, 3, H, W) RGB color
                - opacity: (B, 1, H, W) alpha
        """
        x = self.act(self.conv1(features))
        out = self.conv2(x)

        # Apply activations
        depth = F.softplus(out[:, self.DEPTH_IDX:self.DEPTH_IDX+1])
        depth_scale = F.softplus(out[:, self.DEPTH_SCALE_IDX:self.DEPTH_SCALE_IDX+1])
        xy_offset = torch.tanh(out[:, self.XY_OFFSET_START:self.XY_OFFSET_END])
        scale_xy = F.softplus(out[:, self.SCALE_XY_START:self.SCALE_XY_END])
        rotation = out[:, self.ROTATION_IDX:self.ROTATION_IDX+1]  # Unbounded angle
        color = torch.sigmoid(out[:, self.COLOR_START:self.COLOR_END])
        opacity = torch.sigmoid(out[:, self.OPACITY_IDX:self.OPACITY_IDX+1])

        return {
            'depth': depth,
            'depth_scale': depth_scale,
            'xy_offset': xy_offset,
            'scale_xy': scale_xy,
            'rotation': rotation,
            'color': color,
            'opacity': opacity,
        }


class MultiScaleGaussianHead(nn.Module):
    """
    Multi-scale Gaussian prediction with feature fusion.

    Predicts Gaussians at multiple scales and fuses them for
    a dense final prediction.

    Args:
        in_channels_list: Input channels at each scale
        out_resolution: Output resolution (H, W)
        fusion_channels: Channels for fusion (default: 64)
    """

    def __init__(
            self,
            in_channels_list: List[int],
            out_resolution: Tuple[int, int] = (256, 256),
            fusion_channels: int = 64,
    ):
        super().__init__()

        self.num_scales = len(in_channels_list)
        self.out_resolution = out_resolution

        # Per-scale projection
        self.scale_projs = nn.ModuleList([
            nn.Conv2d(c, fusion_channels, 1) for c in in_channels_list
        ])

        # Fusion and final prediction
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels * self.num_scales, fusion_channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(fusion_channels, fusion_channels, 3, 1, 1),
            nn.GELU(),
        )

        self.head = GaussianHead(fusion_channels)

    def forward(self, multi_scale_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict Gaussians from multi-scale features.

        Args:
            multi_scale_features: List of features at each scale

        Returns:
            Dictionary with Gaussian parameters at output resolution
        """
        H, W = self.out_resolution

        # Project and upsample each scale
        projected = []
        for feat, proj in zip(multi_scale_features, self.scale_projs):
            p = proj(feat)
            p = F.interpolate(p, size=(H, W), mode='bilinear', align_corners=False)
            projected.append(p)

        # Concatenate and fuse
        fused = torch.cat(projected, dim=1)
        fused = self.fusion(fused)

        # Predict Gaussians
        return self.head(fused)


class GaussianAssembler(nn.Module):
    """
    Assembles per-pixel Gaussian predictions into 3D Gaussian representation.

    Converts 2D predictions (depth, offset, etc.) to 3D coordinates
    using camera unprojection.

    Args:
        image_size: Input image size (H, W)
        focal_length: Camera focal length (default: computed from image size)
    """

    def __init__(
            self,
            image_size: Tuple[int, int] = (256, 256),
            focal_length: Optional[float] = None,
    ):
        super().__init__()

        H, W = image_size
        self.H = H
        self.W = W

        # Default focal length: approximate 45 degree FoV
        self.focal_length = focal_length if focal_length else (H + W) / 2

        # Create pixel coordinate grid
        y, x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij'
        )
        # Normalize to [-1, 1] centered
        x = (x - W / 2) / self.focal_length
        y = (y - H / 2) / self.focal_length

        self.register_buffer('grid_x', x)
        self.register_buffer('grid_y', y)

    def forward(self, gaussian_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Assemble 3D Gaussians from per-pixel predictions.

        Args:
            gaussian_params: Dictionary from GaussianHead

        Returns:
            Dictionary with 3D Gaussian representation:
                - xyz: (B, N, 3) 3D positions
                - scale: (B, N, 3) 3D scales
                - rotation: (B, N, 1) rotation angles
                - opacity: (B, N, 1) alpha values
                - color: (B, N, 3) RGB colors
            where N = H * W
        """
        B = gaussian_params['depth'].shape[0]
        H, W = gaussian_params['depth'].shape[2:]

        # Get depth (convert from disparity)
        disparity = gaussian_params['depth']  # (B, 1, H, W)
        depth = 1.0 / (disparity + 1e-6)  # (B, 1, H, W)

        # Get sub-pixel offset
        xy_offset = gaussian_params['xy_offset']  # (B, 2, H, W)

        # Compute 3D positions via unprojection
        # Add offset to grid coordinates
        x = self.grid_x.unsqueeze(0).unsqueeze(0) + xy_offset[:, 0:1] / self.focal_length
        y = self.grid_y.unsqueeze(0).unsqueeze(0) + xy_offset[:, 1:2] / self.focal_length

        # Unproject: X = x * Z, Y = y * Z
        X = x * depth
        Y = y * depth
        Z = depth

        xyz = torch.cat([X, Y, Z], dim=1)  # (B, 3, H, W)
        xyz = xyz.view(B, 3, -1).permute(0, 2, 1)  # (B, N, 3)

        # Assemble 3D scale
        scale_xy = gaussian_params['scale_xy']  # (B, 2, H, W)
        depth_scale = gaussian_params['depth_scale']  # (B, 1, H, W)
        scale = torch.cat([scale_xy, depth_scale], dim=1)  # (B, 3, H, W)
        scale = scale.view(B, 3, -1).permute(0, 2, 1)  # (B, N, 3)

        # Flatten other parameters
        rotation = gaussian_params['rotation'].view(B, 1, -1).permute(0, 2, 1)  # (B, N, 1)
        opacity = gaussian_params['opacity'].view(B, 1, -1).permute(0, 2, 1)  # (B, N, 1)
        color = gaussian_params['color'].view(B, 3, -1).permute(0, 2, 1)  # (B, N, 3)

        return {
            'xyz': xyz,
            'scale': scale,
            'rotation': rotation,
            'opacity': opacity,
            'color': color,
        }

    def get_pixel_coords(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get pixel coordinate grids."""
        return self.grid_x, self.grid_y
