"""
Temporal Fusion Module

Fuses information across N frames using bidirectional Mamba.
This module takes per-frame features and produces temporally-aware features
that incorporate information from past and future frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math

from ..modules.temporal_ssm import TemporalSSMBlock, TemporalPositionEncoding


class TemporalFusion(nn.Module):
    """
    Temporal Fusion using bidirectional Mamba.

    Processes features from N frames and produces temporally-fused features
    where each frame's features incorporate context from all other frames.

    Args:
        dim: Feature dimension
        num_layers: Number of temporal SSM layers (default: 4)
        d_state: SSM state dimension (default: 16)
        bidirectional: Use bidirectional processing (default: True)
        dropout: Dropout rate (default: 0.0)
    """

    def __init__(
            self,
            dim: int,
            num_layers: int = 4,
            d_state: int = 16,
            bidirectional: bool = True,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_layers = num_layers

        # Position encoding for temporal sequence
        self.pos_encoding = TemporalPositionEncoding(dim, max_len=100)

        # Stack of temporal SSM blocks
        self.layers = nn.ModuleList([
            TemporalSSMBlock(
                d_model=dim,
                d_state=d_state,
                bidirectional=bidirectional,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(dim)

    def forward(
            self,
            features: torch.Tensor,
            timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse features across N frames.

        Args:
            features: Per-frame features (B, N, C, H, W)
            timestamps: Optional frame timestamps (B, N) in [0, 1]

        Returns:
            Fused features (B, N, C, H, W)
        """
        B, N, C, H, W = features.shape

        # Reshape to process spatial locations as batch
        # (B, N, C, H, W) -> (B*H*W, N, C)
        x = features.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, N, C)
        x = x.view(B * H * W, N, C)

        # Add temporal position encoding
        x = self.pos_encoding(x, timestamps.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W).reshape(B * H * W, N) if timestamps is not None else None)

        # Process through temporal SSM layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Reshape back
        # (B*H*W, N, C) -> (B, N, C, H, W)
        x = x.view(B, H, W, N, C)
        x = x.permute(0, 3, 4, 1, 2).contiguous()

        return x


class MultiScaleTemporalFusion(nn.Module):
    """
    Multi-scale temporal fusion.

    Applies temporal fusion at multiple feature scales with
    appropriate dimension handling.

    Args:
        dims: Feature dimensions at each scale
        num_layers: Number of temporal layers (can be int or list)
        d_state: SSM state dimension
        bidirectional: Use bidirectional processing
        scales_to_fuse: Which scales to apply fusion (default: all)
    """

    def __init__(
            self,
            dims: List[int],
            num_layers: int = 4,
            d_state: int = 16,
            bidirectional: bool = True,
            scales_to_fuse: Optional[List[int]] = None,
    ):
        super().__init__()

        self.dims = dims
        self.num_scales = len(dims)
        self.scales_to_fuse = scales_to_fuse if scales_to_fuse else list(range(len(dims)))

        # Create fusion module for each scale
        self.fusion_modules = nn.ModuleList()
        for i, dim in enumerate(dims):
            if i in self.scales_to_fuse:
                # Fewer layers for larger scales (more efficient)
                n_layers = max(1, num_layers - i)
                self.fusion_modules.append(
                    TemporalFusion(
                        dim=dim,
                        num_layers=n_layers,
                        d_state=d_state,
                        bidirectional=bidirectional,
                    )
                )
            else:
                self.fusion_modules.append(None)

    def forward(
            self,
            multi_scale_features: List[torch.Tensor],
            timestamps: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Apply temporal fusion at each scale.

        Args:
            multi_scale_features: List of features at each scale,
                                  each (B, N, C_i, H_i, W_i)
            timestamps: Optional frame timestamps (B, N)

        Returns:
            List of fused features at each scale
        """
        fused_features = []

        for i, (features, fusion) in enumerate(zip(multi_scale_features, self.fusion_modules)):
            if fusion is not None:
                fused = fusion(features, timestamps)
            else:
                fused = features
            fused_features.append(fused)

        return fused_features


class CrossFrameAttention(nn.Module):
    """
    Cross-frame attention for explicit frame-to-frame feature matching.

    Alternative to pure temporal SSM - uses attention to explicitly
    correlate features between frames.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.0)
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(
            self,
            features: torch.Tensor,
            timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-frame attention.

        Args:
            features: Per-frame features (B, N, C, H, W)
            timestamps: Optional (unused, for API compatibility)

        Returns:
            Features with cross-frame attention (B, N, C, H, W)
        """
        B, N, C, H, W = features.shape
        residual = features

        # Reshape for attention: (B*H*W, N, C)
        x = features.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, N, C)
        x = self.norm(x)

        # QKV projection
        qkv = self.qkv(x).reshape(B * H * W, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*H*W, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        x = (attn @ v).transpose(1, 2).reshape(B * H * W, N, C)
        x = self.proj(x)
        x = self.dropout(x)

        # Reshape back and add residual
        x = x.view(B, H, W, N, C).permute(0, 3, 4, 1, 2).contiguous()
        x = residual + x

        return x


class HybridTemporalFusion(nn.Module):
    """
    Hybrid temporal fusion combining SSM and attention.

    Uses Mamba SSM for efficient sequence modeling with occasional
    attention layers for explicit long-range dependencies.

    Args:
        dim: Feature dimension
        num_ssm_layers: Number of SSM layers
        num_attn_layers: Number of attention layers
        d_state: SSM state dimension
        num_heads: Attention heads
    """

    def __init__(
            self,
            dim: int,
            num_ssm_layers: int = 3,
            num_attn_layers: int = 1,
            d_state: int = 16,
            num_heads: int = 8,
    ):
        super().__init__()

        self.pos_encoding = TemporalPositionEncoding(dim)

        # Interleave SSM and attention
        self.layers = nn.ModuleList()
        total_layers = num_ssm_layers + num_attn_layers
        attn_positions = set(range(num_ssm_layers, total_layers))  # Attention at end

        for i in range(total_layers):
            if i in attn_positions:
                self.layers.append(('attn', CrossFrameAttention(dim, num_heads)))
            else:
                self.layers.append(('ssm', TemporalSSMBlock(dim, d_state)))

        self.norm = nn.LayerNorm(dim)

    def forward(
            self,
            features: torch.Tensor,
            timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply hybrid temporal fusion.

        Args:
            features: Per-frame features (B, N, C, H, W)
            timestamps: Optional frame timestamps

        Returns:
            Fused features (B, N, C, H, W)
        """
        B, N, C, H, W = features.shape

        for layer_type, layer in self.layers:
            if layer_type == 'ssm':
                # Reshape for SSM: (B*H*W, N, C)
                x = features.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, N, C)
                if timestamps is not None:
                    t = timestamps.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W).reshape(B * H * W, N)
                else:
                    t = None
                x = self.pos_encoding(x, t)
                x = layer(x)
                features = x.view(B, H, W, N, C).permute(0, 3, 4, 1, 2).contiguous()
            else:
                # Attention operates directly on (B, N, C, H, W)
                features = layer(features, timestamps)

        # Final norm
        x = features.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, N, C)
        x = self.norm(x)
        features = x.view(B, H, W, N, C).permute(0, 3, 4, 1, 2).contiguous()

        return features
