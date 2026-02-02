"""
UNet Refinement Network

Refines Gaussian-rendered frames with 2D convolutions.
Adapted from VFIMamba's Unet for GS-Mamba.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


def conv(in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> nn.Module:
    """Conv + PReLU block."""
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.PReLU(out_planes)
    )


def deconv(in_planes: int, out_planes: int, kernel_size: int = 4, stride: int = 2, padding: int = 1) -> nn.Module:
    """Transposed conv + PReLU block."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.PReLU(out_planes)
    )


class Conv2(nn.Module):
    """Two conv layers with stride for downsampling."""

    def __init__(self, in_planes: int, out_planes: int, stride: int = 2):
        super().__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetRefine(nn.Module):
    """
    UNet refinement network for GS-Mamba.

    Takes a coarse rendered frame from Gaussian splatting and refines it
    using information from input frames.

    Architecture:
        - Encoder: 4 downsampling stages
        - Decoder: 4 upsampling stages with skip connections
        - Output: Residual RGB

    Args:
        base_channels: Base channel count (default: 32)
        in_frames: Number of input frames to use for refinement (default: 2)
        use_features: Whether to use multi-scale encoder features (default: False)
        feature_channels: Channel counts at each feature scale (if use_features)
    """

    def __init__(
            self,
            base_channels: int = 32,
            in_frames: int = 2,
            use_features: bool = False,
            feature_channels: Optional[List[int]] = None,
    ):
        super().__init__()

        c = base_channels
        self.use_features = use_features

        # Input channels:
        # - rendered frame: 3
        # - depth map: 1
        # - opacity map: 1
        # - input frames: in_frames * 3
        # - mask: 1
        base_in = 3 + 1 + 1 + in_frames * 3 + 1  # 3 + 1 + 1 + 6 + 1 = 12 for in_frames=2

        # Encoder
        self.down0 = Conv2(base_in, 2 * c)
        self.down1 = Conv2(4 * c if use_features else 2 * c, 4 * c)
        self.down2 = Conv2(8 * c if use_features else 4 * c, 8 * c)
        self.down3 = Conv2(16 * c if use_features else 8 * c, 16 * c)

        # Bottleneck
        # If using features, additional feature channels from encoder
        bottleneck_in = 32 * c if use_features else 16 * c
        self.bottleneck = Conv2(bottleneck_in, 16 * c, stride=1)

        # Decoder
        self.up0 = deconv(16 * c, 8 * c)
        self.up1 = deconv(16 * c, 4 * c)  # 8c + 8c skip
        self.up2 = deconv(8 * c, 2 * c)   # 4c + 4c skip
        self.up3 = deconv(4 * c, c)       # 2c + 2c skip

        # Output (residual)
        self.conv_out = nn.Conv2d(c, 3, 3, 1, 1)

        # Feature projections (if using encoder features)
        if use_features and feature_channels:
            self.feat_projs = nn.ModuleList([
                nn.Conv2d(fc, 2 * c * (2 ** i), 1) if fc else None
                for i, fc in enumerate(feature_channels[:4])
            ])
        else:
            self.feat_projs = None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
            self,
            rendered: torch.Tensor,
            depth: torch.Tensor,
            opacity: torch.Tensor,
            input_frames: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Refine rendered frame.

        Args:
            rendered: Coarse rendered frame (B, 3, H, W)
            depth: Depth map from renderer (B, 1, H, W)
            opacity: Accumulated opacity (B, 1, H, W)
            input_frames: Input frames (B, N, 3, H, W) or (B, N*3, H, W)
            mask: Optional blending mask (B, 1, H, W)
            features: Optional multi-scale encoder features

        Returns:
            Refined frame (B, 3, H, W)
        """
        B = rendered.shape[0]
        H, W = rendered.shape[2:]

        # Prepare input frames
        if input_frames.dim() == 5:
            # (B, N, 3, H, W) -> (B, N*3, H, W)
            N = input_frames.shape[1]
            input_frames = input_frames.view(B, -1, H, W)

        # Default mask
        if mask is None:
            mask = torch.ones(B, 1, H, W, device=rendered.device)

        # Concatenate inputs
        x = torch.cat([rendered, depth, opacity, input_frames, mask], dim=1)

        # Encoder
        s0 = self.down0(x)

        if self.use_features and features and self.feat_projs:
            # Add projected features at each scale
            if self.feat_projs[0] and len(features) > 0:
                f0 = self.feat_projs[0](features[0])
                f0 = F.interpolate(f0, size=s0.shape[2:], mode='bilinear', align_corners=False)
                s0 = torch.cat([s0, f0], dim=1)

        s1 = self.down1(s0)

        if self.use_features and features and self.feat_projs:
            if self.feat_projs[1] and len(features) > 1:
                f1 = self.feat_projs[1](features[1])
                f1 = F.interpolate(f1, size=s1.shape[2:], mode='bilinear', align_corners=False)
                s1 = torch.cat([s1, f1], dim=1)

        s2 = self.down2(s1)

        if self.use_features and features and self.feat_projs:
            if self.feat_projs[2] and len(features) > 2:
                f2 = self.feat_projs[2](features[2])
                f2 = F.interpolate(f2, size=s2.shape[2:], mode='bilinear', align_corners=False)
                s2 = torch.cat([s2, f2], dim=1)

        s3 = self.down3(s2)

        if self.use_features and features and self.feat_projs:
            if self.feat_projs[3] and len(features) > 3:
                f3 = self.feat_projs[3](features[3])
                f3 = F.interpolate(f3, size=s3.shape[2:], mode='bilinear', align_corners=False)
                s3 = torch.cat([s3, f3], dim=1)

        # Bottleneck
        x = self.bottleneck(s3)

        # Decoder with skip connections
        x = self.up0(x)
        x = self.up1(torch.cat([x, s2], dim=1))
        x = self.up2(torch.cat([x, s1], dim=1))
        x = self.up3(torch.cat([x, s0], dim=1))

        # Output residual
        residual = self.conv_out(x)
        residual = torch.tanh(residual)  # [-1, 1]

        # Add residual to rendered frame
        refined = rendered + residual
        refined = refined.clamp(0, 1)

        return refined


class LightweightRefine(nn.Module):
    """
    Lightweight refinement network.

    A simpler, faster alternative to full UNet for real-time applications.

    Args:
        in_channels: Total input channels
        hidden_channels: Hidden layer channels
        num_layers: Number of refinement layers
    """

    def __init__(
            self,
            in_channels: int = 12,
            hidden_channels: int = 64,
            num_layers: int = 4,
    ):
        super().__init__()

        layers = [conv(in_channels, hidden_channels)]
        for _ in range(num_layers - 2):
            layers.append(conv(hidden_channels, hidden_channels))
        layers.append(nn.Conv2d(hidden_channels, 3, 3, 1, 1))

        self.net = nn.Sequential(*layers)

    def forward(
            self,
            rendered: torch.Tensor,
            depth: torch.Tensor,
            opacity: torch.Tensor,
            input_frames: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> torch.Tensor:
        """
        Lightweight refinement.

        Args:
            rendered: Coarse rendered frame (B, 3, H, W)
            depth: Depth map (B, 1, H, W)
            opacity: Opacity map (B, 1, H, W)
            input_frames: Input frames (B, N*3, H, W)
            mask: Optional mask (B, 1, H, W)

        Returns:
            Refined frame (B, 3, H, W)
        """
        B, _, H, W = rendered.shape

        if input_frames.dim() == 5:
            input_frames = input_frames.view(B, -1, H, W)

        if mask is None:
            mask = torch.ones(B, 1, H, W, device=rendered.device)

        x = torch.cat([rendered, depth, opacity, input_frames, mask], dim=1)
        residual = self.net(x)
        residual = torch.tanh(residual)

        refined = rendered + residual
        return refined.clamp(0, 1)
