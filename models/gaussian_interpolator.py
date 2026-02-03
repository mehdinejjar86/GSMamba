"""
Gaussian Interpolator

Interpolates Gaussian parameters between N discrete frames to produce
Gaussians at arbitrary query timesteps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math

from modules.temporal_ssm import TemporalSSMBlock, TemporalPositionEncoding


class GaussianInterpolator(nn.Module):
    """
    Timestep-conditioned Gaussian interpolator.

    Given Gaussians at N discrete times, predicts Gaussians at any query time t.
    Uses learned interpolation with parameter-specific handling.

    Different parameters have different temporal behaviors:
        - Position (xyz): Smooth motion trajectories
        - Scale: Usually stable, small changes
        - Rotation: Angular interpolation
        - Opacity: Can change sharply (occlusion/disocclusion)
        - Color: May have view-dependent changes

    Args:
        hidden_dim: Hidden dimension for processing (default: 128)
        num_layers: Number of processing layers (default: 2)
        use_residual: Add residual from linear interpolation (default: True)
    """

    def __init__(
            self,
            hidden_dim: int = 128,
            num_layers: int = 2,
            use_residual: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_residual = use_residual

        # Parameter embedding dimensions
        xyz_dim = 3
        scale_dim = 3
        rot_dim = 1
        opacity_dim = 1
        color_dim = 3
        total_dim = xyz_dim + scale_dim + rot_dim + opacity_dim + color_dim  # 11

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
        )

        # Parameter embeddings
        self.param_embed = nn.Linear(total_dim * 2, hidden_dim)  # *2 for start/end frame params

        # Temporal processing layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),  # +time embedding
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        # Parameter-specific decoders
        self.xyz_decoder = nn.Linear(hidden_dim, xyz_dim)
        self.scale_decoder = nn.Linear(hidden_dim, scale_dim)
        self.rot_decoder = nn.Linear(hidden_dim, rot_dim)
        self.opacity_decoder = nn.Linear(hidden_dim, opacity_dim)
        self.color_decoder = nn.Linear(hidden_dim, color_dim)

    def _find_bounding_frames(
            self,
            t: torch.Tensor,
            timestamps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find the two frames that bound the query timestep.

        Args:
            t: Query timestep (B,) or scalar
            timestamps: Frame timestamps (B, N) or (N,)

        Returns:
            Tuple of (idx0, idx1, alpha) where alpha is interpolation weight
        """
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(0)

        B, N = timestamps.shape

        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=timestamps.device)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        t = t.view(B, 1)

        # Find indices where t falls between consecutive timestamps
        # For each t, find largest i such that timestamps[i] <= t
        diff = timestamps - t  # (B, N)
        diff[diff > 0] = float('-inf')  # Mask future frames

        idx0 = diff.argmax(dim=1)  # (B,)

        # idx1 is the next frame (or same if at end)
        idx1 = (idx0 + 1).clamp(max=N - 1)

        # Compute interpolation weight
        t0 = timestamps.gather(1, idx0.unsqueeze(1)).squeeze(1)  # (B,)
        t1 = timestamps.gather(1, idx1.unsqueeze(1)).squeeze(1)  # (B,)

        # Avoid division by zero when t0 == t1
        dt = (t1 - t0).clamp(min=1e-6)
        alpha = ((t.squeeze(1) - t0) / dt).clamp(0, 1)  # (B,)

        return idx0, idx1, alpha

    def _gather_gaussians(
            self,
            gaussians: List[Dict[str, torch.Tensor]],
            indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Gather Gaussians at specified frame indices.

        Args:
            gaussians: List of N Gaussian dicts, each with (B, num_points, C) tensors
            indices: Frame indices to gather (B,)

        Returns:
            Gaussian dict with gathered parameters
        """
        B = indices.shape[0]
        N = len(gaussians)

        result = {}
        for key in gaussians[0].keys():
            # Stack all frames: (N, B, num_points, C)
            stacked = torch.stack([g[key] for g in gaussians], dim=0)

            # Gather: for each batch element, select the appropriate frame
            # indices: (B,) -> (1, B, 1, 1)
            idx = indices.view(1, B, 1, 1).expand(1, B, stacked.shape[2], stacked.shape[3])
            gathered = stacked.gather(0, idx).squeeze(0)  # (B, num_points, C)

            result[key] = gathered

        return result

    def forward(
            self,
            gaussians: List[Dict[str, torch.Tensor]],
            t: Union[float, torch.Tensor],
            timestamps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Interpolate Gaussians to query timestep.

        Args:
            gaussians: List of N Gaussian dicts from each frame
                      Each dict has keys: xyz, scale, rotation, opacity, color
                      Each value has shape (B, num_points, C)
            t: Query timestep in [0, 1] (scalar or (B,) tensor)
            timestamps: Optional explicit frame timestamps (B, N) or (N,)
                       If None, assumes uniform spacing [0, 1/(N-1), 2/(N-1), ..., 1]

        Returns:
            Interpolated Gaussian dict at timestep t
        """
        N = len(gaussians)
        B = gaussians[0]['xyz'].shape[0]
        num_points = gaussians[0]['xyz'].shape[1]
        device = gaussians[0]['xyz'].device

        # Default timestamps: uniform spacing
        if timestamps is None:
            timestamps = torch.linspace(0, 1, N, device=device)

        # Find bounding frames
        idx0, idx1, alpha = self._find_bounding_frames(t, timestamps)

        # Gather Gaussians at bounding frames
        g0 = self._gather_gaussians(gaussians, idx0)
        g1 = self._gather_gaussians(gaussians, idx1)

        # Linear interpolation baseline
        alpha_expanded = alpha.view(B, 1, 1)

        xyz_linear = (1 - alpha_expanded) * g0['xyz'] + alpha_expanded * g1['xyz']
        scale_linear = (1 - alpha_expanded) * g0['scale'] + alpha_expanded * g1['scale']
        rot_linear = self._angle_interp(g0['rotation'], g1['rotation'], alpha_expanded)
        opacity_linear = (1 - alpha_expanded) * g0['opacity'] + alpha_expanded * g1['opacity']
        color_linear = (1 - alpha_expanded) * g0['color'] + alpha_expanded * g1['color']

        if not self.use_residual:
            return {
                'xyz': xyz_linear,
                'scale': scale_linear,
                'rotation': rot_linear,
                'opacity': opacity_linear,
                'color': color_linear,
            }

        # Learned residual prediction
        # Concatenate all parameters from both bounding frames
        params_0 = torch.cat([g0['xyz'], g0['scale'], g0['rotation'], g0['opacity'], g0['color']], dim=-1)
        params_1 = torch.cat([g1['xyz'], g1['scale'], g1['rotation'], g1['opacity'], g1['color']], dim=-1)
        params_concat = torch.cat([params_0, params_1], dim=-1)  # (B, num_points, 22)

        # Embed parameters
        h = self.param_embed(params_concat)  # (B, num_points, hidden_dim)

        # Embed timestep
        t_tensor = alpha.view(B, 1, 1).expand(B, num_points, 1)
        t_embed = self.time_embed(t_tensor)  # (B, num_points, hidden_dim//4)

        # Process through layers
        for layer in self.layers:
            h_with_t = torch.cat([h, t_embed], dim=-1)
            h = h + layer(h_with_t)

        # Decode residuals
        xyz_residual = self.xyz_decoder(h)
        scale_residual = self.scale_decoder(h)
        rot_residual = self.rot_decoder(h)
        opacity_residual = self.opacity_decoder(h)
        color_residual = self.color_decoder(h)

        # Add residuals to linear interpolation
        return {
            'xyz': xyz_linear + xyz_residual,
            'scale': F.softplus(scale_linear + scale_residual),
            'rotation': rot_linear + 0.1 * rot_residual,  # Small rotation adjustment
            'opacity': torch.sigmoid(torch.logit(opacity_linear.clamp(1e-6, 1-1e-6)) + opacity_residual),
            'color': torch.sigmoid(torch.logit(color_linear.clamp(1e-6, 1-1e-6)) + color_residual),
        }

    @staticmethod
    def _angle_interp(a0: torch.Tensor, a1: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Interpolate angles handling wrap-around.

        Args:
            a0, a1: Start and end angles
            alpha: Interpolation weight

        Returns:
            Interpolated angle
        """
        # Compute shortest path
        diff = a1 - a0

        # Wrap to [-pi, pi]
        diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi

        return a0 + alpha * diff


class SSMGaussianInterpolator(nn.Module):
    """
    Gaussian interpolator using temporal SSM.

    Instead of finding bounding frames, processes all N frames through
    a temporal SSM and queries the hidden state at the desired timestep.

    Args:
        gaussian_dim: Dimension of Gaussian parameters (default: 11)
        hidden_dim: SSM hidden dimension (default: 128)
        d_state: SSM state dimension (default: 16)
        num_layers: Number of SSM layers (default: 2)
    """

    def __init__(
            self,
            gaussian_dim: int = 11,
            hidden_dim: int = 128,
            d_state: int = 16,
            num_layers: int = 2,
    ):
        super().__init__()

        self.gaussian_dim = gaussian_dim

        # Input embedding
        self.input_embed = nn.Linear(gaussian_dim, hidden_dim)

        # Temporal SSM layers
        self.ssm_layers = nn.ModuleList([
            TemporalSSMBlock(
                d_model=hidden_dim,
                d_state=d_state,
                bidirectional=True,
            )
            for _ in range(num_layers)
        ])

        # Position encoding
        self.pos_encoding = TemporalPositionEncoding(hidden_dim)

        # Query timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, gaussian_dim)

    def forward(
            self,
            gaussians: List[Dict[str, torch.Tensor]],
            t: Union[float, torch.Tensor],
            timestamps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Interpolate using SSM.

        Args:
            gaussians: List of N Gaussian dicts
            t: Query timestep
            timestamps: Frame timestamps

        Returns:
            Interpolated Gaussians
        """
        N = len(gaussians)
        B = gaussians[0]['xyz'].shape[0]
        num_points = gaussians[0]['xyz'].shape[1]
        device = gaussians[0]['xyz'].device

        if timestamps is None:
            timestamps = torch.linspace(0, 1, N, device=device)

        # Stack all Gaussian parameters: (B, N, num_points, gaussian_dim)
        params_list = []
        for g in gaussians:
            params = torch.cat([g['xyz'], g['scale'], g['rotation'], g['opacity'], g['color']], dim=-1)
            params_list.append(params)
        params = torch.stack(params_list, dim=1)

        # Reshape for processing: (B * num_points, N, gaussian_dim)
        params = params.permute(0, 2, 1, 3).contiguous()
        params = params.view(B * num_points, N, self.gaussian_dim)

        # Embed
        h = self.input_embed(params)

        # Add position encoding
        if timestamps.dim() == 1:
            timestamps_expanded = timestamps.unsqueeze(0).expand(B * num_points, -1)
        else:
            timestamps_expanded = timestamps.unsqueeze(1).expand(-1, num_points, -1).reshape(B * num_points, N)
        h = self.pos_encoding(h, timestamps_expanded)

        # Process through SSM layers
        for layer in self.ssm_layers:
            h = layer(h)

        # Interpolate hidden states to query timestep
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=device)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.view(B, 1).expand(B, num_points).reshape(B * num_points, 1)

        # Linear interpolation of hidden states based on t
        # Find position in sequence
        t_normalized = t * (N - 1)  # Scale to [0, N-1]
        idx_float = t_normalized.squeeze(-1)
        idx_low = idx_float.floor().long().clamp(0, N - 2)
        idx_high = (idx_low + 1).clamp(max=N - 1)
        alpha = (idx_float - idx_low.float()).unsqueeze(-1)

        # Gather hidden states
        h_low = h.gather(1, idx_low.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.input_embed.out_features)).squeeze(1)
        h_high = h.gather(1, idx_high.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.input_embed.out_features)).squeeze(1)

        # Interpolate
        h_interp = (1 - alpha) * h_low + alpha * h_high

        # Add timestep embedding
        t_embed = self.time_embed(t)
        h_interp = h_interp + t_embed

        # Project to output
        out = self.output_proj(h_interp)  # (B * num_points, gaussian_dim)
        out = out.view(B, num_points, self.gaussian_dim)

        # Split into parameters
        return {
            'xyz': out[..., :3],
            'scale': F.softplus(out[..., 3:6]),
            'rotation': out[..., 6:7],
            'opacity': torch.sigmoid(out[..., 7:8]),
            'color': torch.sigmoid(out[..., 8:11]),
        }
