"""
VSSBlock and BiMambaBlock

Vision Selective State Space blocks for spatial feature processing.
Adapted from VFIMamba for GS-Mamba.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional

from modules.ss2d import SS2D


class ChannelAttention(nn.Module):
    """
    Channel attention module (squeeze-and-excitation style).

    Args:
        num_feat: Number of input/output channels
        squeeze_factor: Channel reduction factor (default: 16)
    """

    def __init__(self, num_feat: int, squeeze_factor: int = 16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)


class CAB(nn.Module):
    """
    Channel Attention Block.

    Applies convolutions followed by channel attention.

    Args:
        num_feat: Number of channels
        compress_ratio: Compression ratio for intermediate features (default: 3)
        squeeze_factor: Squeeze factor for channel attention (default: 30)
    """

    def __init__(self, num_feat: int, compress_ratio: int = 3, squeeze_factor: int = 30):
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cab(x)


class VSSBlock(nn.Module):
    """
    Vision Selective State Space Block.

    Combines SS2D (selective scan) with channel attention for
    comprehensive spatial feature processing.

    Args:
        hidden_dim: Feature dimension
        drop_path: Drop path rate (default: 0.0)
        norm_layer: Normalization layer (default: LayerNorm)
        attn_drop_rate: Attention dropout rate (default: 0.0)
        d_state: SSM state dimension (default: 16)
        mlp_ratio: MLP expansion ratio (default: 2.0)
    """

    def __init__(
            self,
            hidden_dim: int,
            drop_path: float = 0.0,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0.0,
            d_state: int = 16,
            mlp_ratio: float = 2.0,
            **kwargs,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Pre-norm for SS2D
        self.ln_1 = norm_layer(hidden_dim)

        # Selective scan attention
        self.self_attention = SS2D(
            d_model=hidden_dim,
            d_state=d_state,
            expand=mlp_ratio,
            dropout=attn_drop_rate,
            **kwargs
        )

        # Learnable skip scale
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))

        # Channel attention block
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, C, H, W)
        """
        # Convert to (B, H, W, C) for LayerNorm
        x = x.permute(0, 2, 3, 1).contiguous()

        # SS2D branch with residual
        x_norm = self.ln_1(x)
        x = x * self.skip_scale + self.self_attention(x_norm)

        # CAB branch with residual
        x_norm2 = self.ln_2(x).permute(0, 3, 1, 2).contiguous()
        x = x * self.skip_scale2 + self.conv_blk(x_norm2).permute(0, 2, 3, 1).contiguous()

        # Convert back to (B, C, H, W)
        return x.permute(0, 3, 1, 2).contiguous()


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba Block.

    Stacks multiple VSSBlocks for deeper feature processing.

    Args:
        dim: Feature dimension
        depth: Number of VSSBlocks to stack
        norm_layer: Normalization layer (default: LayerNorm)
        d_state: SSM state dimension (default: 16)
    """

    def __init__(
            self,
            dim: int,
            depth: int,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                norm_layer=norm_layer,
                d_state=d_state,
                **kwargs
            )
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all blocks.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, C, H, W)
        """
        for blk in self.blocks:
            x = blk(x)
        return x


class ConvBlock(nn.Module):
    """
    Convolutional block for early stages.

    Uses standard convolutions instead of Mamba for local feature extraction.

    Args:
        in_dim: Input channels
        out_dim: Output channels
        depths: Number of conv layers (default: 2)
        act_layer: Activation layer (default: PReLU)
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            depths: int = 2,
            act_layer: Callable[..., nn.Module] = nn.PReLU
    ):
        super().__init__()

        layers = []
        for i in range(depths):
            in_channels = in_dim if i == 0 else out_dim
            layers.extend([
                nn.Conv2d(in_channels, out_dim, 3, 1, 1),
                act_layer(out_dim),
            ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class OverlapPatchEmbed(nn.Module):
    """
    Overlapping patch embedding with downsampling.

    Args:
        patch_size: Patch size (default: 3)
        stride: Stride for downsampling (default: 2)
        in_chans: Input channels
        embed_dim: Output embedding dimension
    """

    def __init__(
            self,
            patch_size: int = 3,
            stride: int = 2,
            in_chans: int = 3,
            embed_dim: int = 768
    ):
        super().__init__()

        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2
        )
        self.norm = nn.LayerNorm(in_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, embed_dim, H//stride, W//stride)
        """
        B, C, H, W = x.shape

        # Pre-norm
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x).permute(0, 3, 1, 2).contiguous()

        # Project and downsample
        x = self.proj(x)

        return x
