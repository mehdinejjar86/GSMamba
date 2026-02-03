"""
GS-Mamba: N-Frame Video Interpolation via 3D Gaussian Splatting + Mamba

Main model class that combines all components:
1. Feature Encoder (SS2D backbone)
2. Temporal Fusion (Bidirectional Mamba)
3. Gaussian Prediction Head
4. Gaussian Interpolator
5. Differentiable Gaussian Renderer
6. UNet Refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from config import GSMambaConfig
from models.feature_encoder import FeatureEncoder
from models.temporal_fusion import MultiScaleTemporalFusion, TemporalFusion
from models.gaussian_head import GaussianHead, GaussianAssembler, MultiScaleGaussianHead
from models.gaussian_interpolator import GaussianInterpolator
from models.renderer import GaussianRenderer
from models.refine import UNetRefine


class GSMamba(nn.Module):
    """
    GS-Mamba: N-Frame Video Interpolation via 3D Gaussian Splatting + Mamba.

    Takes N consecutive video frames and produces interpolated frames at
    arbitrary timesteps by:
    1. Lifting frames to 3D Gaussian representations
    2. Fusing temporal information via bidirectional Mamba
    3. Interpolating Gaussians to query timestep
    4. Rendering via differentiable Gaussian splatting
    5. Refining with 2D UNet

    Args:
        config: GSMambaConfig with model parameters
    """

    def __init__(self, config: Optional[GSMambaConfig] = None):
        super().__init__()

        if config is None:
            config = GSMambaConfig()

        self.config = config

        # 1. Feature Encoder (shared SS2D backbone)
        self.encoder = FeatureEncoder(
            in_chans=3,
            embed_dims=config.embed_dims,
            depths=config.depths,
            conv_stages=config.conv_stages,
            d_state=config.d_state,
        )

        # 2. Temporal Fusion (at each scale)
        self.temporal_fusion = MultiScaleTemporalFusion(
            dims=config.embed_dims,
            num_layers=config.temporal_num_layers,
            d_state=config.d_state,
            bidirectional=True,
            # Only fuse at Mamba stages (skip conv stages)
            scales_to_fuse=list(range(config.conv_stages, len(config.embed_dims))),
        )

        # 3. Gaussian Prediction Head
        self.gaussian_head = MultiScaleGaussianHead(
            in_channels_list=config.embed_dims,
            out_resolution=config.image_size,
            fusion_channels=config.embed_dims[0] * 2,
        )

        # 4. Gaussian Assembler (2D predictions -> 3D Gaussians)
        self.gaussian_assembler = GaussianAssembler(
            image_size=config.image_size,
        )

        # 5. Gaussian Interpolator
        self.interpolator = GaussianInterpolator(
            hidden_dim=config.temporal_hidden_dim,
            num_layers=2,
            use_residual=True,
        )

        # 6. Differentiable Gaussian Renderer
        self.renderer = GaussianRenderer(
            image_size=config.image_size,
            fov=config.default_fov,
            sh_degree=config.sh_degree,
        )

        # 7. UNet Refinement (optional)
        self.use_refinement = config.use_refinement
        if config.use_refinement:
            self.refine = UNetRefine(
                base_channels=config.refine_channels,
                in_frames=2,  # Use 2 nearest frames for refinement
                use_features=False,
            )

    def encode_frames(
            self,
            frames: torch.Tensor,
            timestamps: Optional[torch.Tensor] = None,
    ) -> Tuple[List[List[torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        """
        Encode N frames to multi-scale features and Gaussians.

        Args:
            frames: Input frames (B, N, 3, H, W)
            timestamps: Frame timestamps (B, N) in [0, 1]

        Returns:
            Tuple of:
                - Multi-scale fused features for each frame
                - List of N Gaussian dicts
        """
        B, N, C, H, W = frames.shape
        device = frames.device

        # Default timestamps: uniform spacing
        if timestamps is None:
            timestamps = torch.linspace(0, 1, N, device=device).unsqueeze(0).expand(B, -1)

        # Extract per-frame features (shared encoder)
        all_features = self.encoder(frames)  # List of N feature lists

        # Reorganize: for each scale, stack features from all N frames
        # all_features[n][s] -> features[s] of shape (B, N, C_s, H_s, W_s)
        num_scales = len(self.config.embed_dims)
        multi_scale_features = []

        for s in range(num_scales):
            scale_features = torch.stack(
                [all_features[n][s] for n in range(N)],
                dim=1
            )  # (B, N, C_s, H_s, W_s)
            multi_scale_features.append(scale_features)

        # Temporal fusion at each scale
        fused_features = self.temporal_fusion(multi_scale_features, timestamps)

        # Predict Gaussians for each frame
        gaussians_list = []
        for n in range(N):
            # Get features for frame n at each scale
            frame_features = [fused_features[s][:, n] for s in range(num_scales)]

            # Predict per-pixel Gaussian parameters
            gaussian_params = self.gaussian_head(frame_features)

            # Assemble to 3D Gaussians
            gaussians = self.gaussian_assembler(gaussian_params)
            gaussians_list.append(gaussians)

        return fused_features, gaussians_list

    def interpolate_and_render(
            self,
            gaussians_list: List[Dict[str, torch.Tensor]],
            t: Union[float, torch.Tensor],
            timestamps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Interpolate Gaussians to timestep t and render.

        Args:
            gaussians_list: List of N Gaussian dicts
            t: Query timestep in [0, 1]
            timestamps: Frame timestamps (B, N)

        Returns:
            Dict with rendered image and depth
        """
        # Interpolate Gaussians
        gaussians_t = self.interpolator(gaussians_list, t, timestamps)

        # Render
        render_output = self.renderer(gaussians_t)

        return {
            'gaussians': gaussians_t,
            'render': render_output['render'],
            'depth': render_output['depth'],
        }

    def forward(
            self,
            frames: torch.Tensor,
            t: Union[float, torch.Tensor] = 0.5,
            timestamps: Optional[torch.Tensor] = None,
            return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: frames -> interpolated frame at timestep t.

        Args:
            frames: Input frames (B, N, 3, H, W) where N >= 2
            t: Query timestep in [0, 1] (default: 0.5 for middle frame)
            timestamps: Optional frame timestamps (B, N)
            return_intermediates: Whether to return intermediate outputs

        Returns:
            Dict with:
                - pred: Final interpolated frame (B, 3, H, W)
                - render: Coarse rendered frame (before refinement)
                - depth: Depth map
                - gaussians: Interpolated Gaussians (if return_intermediates)
                - all_gaussians: Gaussians for all input frames (if return_intermediates)
        """
        B, N, C, H, W = frames.shape
        device = frames.device

        # Default timestamps
        if timestamps is None:
            timestamps = torch.linspace(0, 1, N, device=device).unsqueeze(0).expand(B, -1)

        # Encode frames and get Gaussians
        fused_features, gaussians_list = self.encode_frames(frames, timestamps)

        # Interpolate and render
        render_output = self.interpolate_and_render(gaussians_list, t, timestamps)

        rendered = render_output['render']
        depth = render_output['depth']

        # Refinement
        if self.use_refinement:
            # Find two nearest frames to timestep t
            if isinstance(t, float):
                t_tensor = torch.tensor([t], device=device).expand(B)
            else:
                t_tensor = t

            # Get nearest frame indices
            t_expanded = t_tensor.view(B, 1)
            diffs = (timestamps - t_expanded).abs()
            _, nearest_indices = diffs.topk(2, dim=1, largest=False)

            # Gather nearest frames
            idx0 = nearest_indices[:, 0]
            idx1 = nearest_indices[:, 1]

            nearest_frames = torch.stack([
                frames[b, idx0[b]] for b in range(B)
            ] + [
                frames[b, idx1[b]] for b in range(B)
            ], dim=0).view(B, 2, C, H, W)

            # Compute opacity map (sum of Gaussian opacities projected to image)
            opacity = render_output['gaussians']['opacity'].mean(dim=1, keepdim=True)  # (B, 1)
            opacity_map = torch.ones(B, 1, H, W, device=device) * opacity.view(B, 1, 1, 1)

            # Refine
            pred = self.refine(
                rendered=rendered,
                depth=depth,
                opacity=opacity_map,
                input_frames=nearest_frames,
            )
        else:
            pred = rendered

        # Prepare output
        output = {
            'pred': pred,
            'render': rendered,
            'depth': depth,
        }

        if return_intermediates:
            output['gaussians'] = render_output['gaussians']
            output['all_gaussians'] = gaussians_list
            output['fused_features'] = fused_features

        return output

    def inference(
            self,
            frames: torch.Tensor,
            t: Union[float, torch.Tensor] = 0.5,
            timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Simplified inference: returns only the interpolated frame.

        Args:
            frames: Input frames (B, N, 3, H, W)
            t: Query timestep
            timestamps: Frame timestamps

        Returns:
            Interpolated frame (B, 3, H, W)
        """
        with torch.no_grad():
            output = self.forward(frames, t, timestamps, return_intermediates=False)
        return output['pred']

    def multi_frame_inference(
            self,
            frames: torch.Tensor,
            num_interpolations: int = 1,
            timestamps: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Generate multiple interpolated frames.

        Args:
            frames: Input frames (B, N, 3, H, W)
            num_interpolations: Number of frames to interpolate between each pair
            timestamps: Frame timestamps

        Returns:
            List of interpolated frames
        """
        B, N, C, H, W = frames.shape
        device = frames.device

        if timestamps is None:
            timestamps = torch.linspace(0, 1, N, device=device).unsqueeze(0).expand(B, -1)

        # Encode once
        with torch.no_grad():
            fused_features, gaussians_list = self.encode_frames(frames, timestamps)

        # Generate interpolated frames
        interpolated = []
        for i in range(N - 1):
            t_start = timestamps[0, i].item()
            t_end = timestamps[0, i + 1].item()

            for j in range(1, num_interpolations + 1):
                t = t_start + (t_end - t_start) * j / (num_interpolations + 1)

                with torch.no_grad():
                    render_output = self.interpolate_and_render(gaussians_list, t, timestamps)

                    if self.use_refinement:
                        nearest_frames = frames[:, [i, i + 1]]
                        opacity_map = torch.ones(B, 1, H, W, device=device)
                        pred = self.refine(
                            rendered=render_output['render'],
                            depth=render_output['depth'],
                            opacity=opacity_map,
                            input_frames=nearest_frames,
                        )
                    else:
                        pred = render_output['render']

                interpolated.append(pred)

        return interpolated


def build_model(config_name: str = "gsmamba") -> GSMamba:
    """
    Build a GS-Mamba model from config name.

    Args:
        config_name: One of "gsmamba", "gsmamba_small", "gsmamba_large"

    Returns:
        GSMamba model instance
    """
    from config import get_config
    config = get_config(config_name)
    return GSMamba(config)
