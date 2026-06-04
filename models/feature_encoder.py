"""
Feature Encoder

Shared SS2D backbone for extracting multi-scale features from N input frames.
Based on VFIMamba's MambaFeature but adapted for variable N frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

from modules.vss_block import BiMambaBlock, ConvBlock, OverlapPatchEmbed


class FeatureEncoder(nn.Module):
    """
    Multi-scale feature encoder using SS2D (Mamba) blocks.

    Extracts hierarchical features from N input frames using shared weights.
    Early stages use convolutions for local features, later stages use
    BiMambaBlock for global context via selective scan.

    Args:
        in_chans: Number of input channels (default: 3 for RGB)
        embed_dims: Feature dimensions at each stage
        depths: Number of blocks at each stage
        conv_stages: Number of stages using conv instead of Mamba (default: 2)
        d_state: SSM state dimension (default: 16)
    """

    def __init__(
            self,
            in_chans: int = 3,
            embed_dims: List[int] = [32, 64, 128, 256, 512],
            depths: List[int] = [2, 2, 2, 3, 3],
            conv_stages: int = 2,
            d_state: int = 16,
    ):
        super().__init__()

        self.num_stages = len(embed_dims)
        self.embed_dims = embed_dims
        self.conv_stages = conv_stages

        # Build stages
        for i in range(self.num_stages):
            if i == 0:
                # First stage: no downsampling
                block = ConvBlock(in_chans, embed_dims[i], depths[i])
            else:
                # Downsampling + feature extraction
                if i < conv_stages:
                    # Conv stages
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3, 2, 1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i], embed_dims[i], depths[i])
                else:
                    # Mamba stages
                    patch_embed = OverlapPatchEmbed(
                        patch_size=3,
                        stride=2,
                        in_chans=embed_dims[i-1],
                        embed_dim=embed_dims[i]
                    )
                    block = BiMambaBlock(embed_dims[i], depths[i], d_state=d_state)

                setattr(self, f"patch_embed{i}", patch_embed)

            setattr(self, f"block{i}", block)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_single(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from a single frame.

        Args:
            x: Input frame (B, C, H, W)

        Returns:
            List of features at each scale
        """
        features = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i}", None)
            block = getattr(self, f"block{i}", None)

            if i > 0 and patch_embed is not None:
                x = patch_embed(x)

            x = block(x)
            features.append(x)

        return features

    def forward(self, frames: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Extract features from N frames.

        Args:
            frames: Input frames (B, N, C, H, W) or list of (B, C, H, W) tensors

        Returns:
            List of N feature lists, where each feature list contains
            features at all scales: [[F0_s1, F0_s2, ...], [F1_s1, F1_s2, ...], ...]
        """
        if isinstance(frames, torch.Tensor):
            if frames.dim() == 5:
                # (B, N, C, H, W) format
                B, N, C, H, W = frames.shape
                frame_list = [frames[:, i] for i in range(N)]
            else:
                # Single frame (B, C, H, W)
                frame_list = [frames]
        else:
            # List of frames
            frame_list = frames

        # Process all frames with shared weights
        all_features = []
        for frame in frame_list:
            features = self.forward_single(frame)
            all_features.append(features)

        return all_features

    def forward_batched(self, frames: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """
        Extract features from N frames using batch processing for efficiency.

        Concatenates all frames along batch dimension, processes once,
        then splits back.

        Args:
            frames: Input frames (B, N, C, H, W)

        Returns:
            Tuple of:
                - List of features at each scale, each (B*N, C_i, H_i, W_i)
                - N (number of frames)
        """
        B, N, C, H, W = frames.shape

        # Flatten to (B*N, C, H, W)
        x = frames.view(B * N, C, H, W)

        # Extract features
        features = self.forward_single(x)

        return features, N
