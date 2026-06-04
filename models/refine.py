"""
UNet Refinement Network

Refines Gaussian-rendered frames with 2D convolutions.
Adapted from VFIMamba's Unet for GS-Mamba.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
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
    """

    def __init__(
            self,
            base_channels: int = 32,
            in_frames: int = 2,
    ):
        super().__init__()

        c = base_channels

        # Input channels:
        # - rendered frame: 3
        # - depth map: 1
        # - opacity map: 1
        # - input frames: in_frames * 3
        # - mask: 1
        base_in = 3 + 1 + 1 + in_frames * 3 + 1  # 3 + 1 + 1 + 6 + 1 = 12 for in_frames=2

        # Encoder
        self.down0 = Conv2(base_in, 2 * c)
        self.down1 = Conv2(2 * c, 4 * c)
        self.down2 = Conv2(4 * c, 8 * c)
        self.down3 = Conv2(8 * c, 16 * c)

        # Bottleneck
        self.bottleneck = Conv2(16 * c, 16 * c, stride=1)

        # Decoder
        self.up0 = deconv(16 * c, 8 * c)
        self.up1 = deconv(16 * c, 4 * c)  # 8c + 8c skip
        self.up2 = deconv(8 * c, 2 * c)   # 4c + 4c skip
        self.up3 = deconv(4 * c, c)       # 2c + 2c skip

        # Output (residual)
        self.conv_out = nn.Conv2d(c, 3, 3, 1, 1)

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
    ) -> torch.Tensor:
        """
        Refine rendered frame.

        Args:
            rendered: Coarse rendered frame (B, 3, H, W)
            depth: Depth map from renderer (B, 1, H, W)
            opacity: Accumulated opacity (B, 1, H, W)
            input_frames: Input frames (B, N, 3, H, W) or (B, N*3, H, W)
            mask: Optional blending mask (B, 1, H, W)

        Returns:
            Refined frame (B, 3, H, W)
        """
        B = rendered.shape[0]
        H, W = rendered.shape[2:]

        # Prepare input frames
        if input_frames.dim() == 5:
            # (B, N, 3, H, W) -> (B, N*3, H, W)
            input_frames = input_frames.view(B, -1, H, W)

        # Default mask
        if mask is None:
            mask = torch.ones(B, 1, H, W, device=rendered.device)

        # Concatenate inputs
        x = torch.cat([rendered, depth, opacity, input_frames, mask], dim=1)

        # Encoder
        s0 = self.down0(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)

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
