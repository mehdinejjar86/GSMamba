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
        rot_dim = 4      # Full SO(3) quaternion
        opacity_dim = 1
        color_dim = 3
        total_dim = xyz_dim + scale_dim + rot_dim + opacity_dim + color_dim  # 14

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
            t = torch.tensor(t, device=timestamps.device, dtype=timestamps.dtype)
        else:
            t = t.to(device=timestamps.device, dtype=timestamps.dtype)

        # Normalize t to shape (B,), supporting scalar, (1,), (B,), or (B,1)
        if t.dim() == 0:
            t = t.expand(B)
        elif t.dim() == 1:
            if t.shape[0] == 1 and B > 1:
                t = t.expand(B)
            elif t.shape[0] != B:
                raise ValueError(f"t must have shape (B,) with B={B}, got {tuple(t.shape)}")
        elif t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(1)
            if t.shape[0] == 1 and B > 1:
                t = t.expand(B)
            elif t.shape[0] != B:
                raise ValueError(f"t must have shape (B,1) with B={B}, got {tuple(t.shape)}")
        else:
            raise ValueError(f"Unsupported t shape: {tuple(t.shape)}")

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
            flow: Optional[torch.Tensor] = None,
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
            flow: Optional optical flow (B, 4, H, W) with fwd=[:, :2] and bwd=[:, 2:4].
                  When provided, bounding-frame Gaussian attribute maps are warped to t
                  before interpolation for better correspondence at motion boundaries.

        Returns:
            Interpolated Gaussian dict at timestep t
        """
        N = len(gaussians)
        B = gaussians[0]['xyz'].shape[0]
        num_points = gaussians[0]['xyz'].shape[1]
        device = gaussians[0]['xyz'].device

        # Default timestamps: uniform spacing
        if timestamps is None:
            timestamps = torch.linspace(0, 1, N, device=device).unsqueeze(0).expand(B, -1)
        elif timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(0).expand(B, -1)
        elif timestamps.dim() == 2 and timestamps.shape[0] == 1 and B > 1:
            timestamps = timestamps.expand(B, -1)
        elif timestamps.dim() == 2 and timestamps.shape[0] != B:
            raise ValueError(
                f"timestamps batch size must match gaussians batch size {B}, "
                f"got {timestamps.shape[0]}"
            )

        # Find bounding frames
        idx0, idx1, alpha = self._find_bounding_frames(t, timestamps)

        # Gather Gaussians at bounding frames
        g0 = self._gather_gaussians(gaussians, idx0)
        g1 = self._gather_gaussians(gaussians, idx1)

        # Fix C: Flow-guided correspondence warp
        if flow is not None:
            H, W = flow.shape[2], flow.shape[3]
            fwd_flow = flow[:, :2]   # f_a → t
            bwd_flow = flow[:, 2:4]  # f_b → t
            for key in list(g0.keys()):
                g0[key] = self._from_map(self._warp_map(self._to_map(g0[key], H, W), fwd_flow))
                g1[key] = self._from_map(self._warp_map(self._to_map(g1[key], H, W), bwd_flow))

        # Interpolation baseline
        alpha_expanded = alpha.view(B, 1, 1)

        xyz_linear = (1 - alpha_expanded) * g0['xyz'] + alpha_expanded * g1['xyz']
        scale_linear = (1 - alpha_expanded) * g0['scale'] + alpha_expanded * g1['scale']
        rot_linear = self._slerp(g0['rotation'], g1['rotation'], alpha_expanded)
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
        params_0 = torch.cat([g0['xyz'], g0['scale'], g0['rotation'], g0['opacity'], g0['color']], dim=-1)
        params_1 = torch.cat([g1['xyz'], g1['scale'], g1['rotation'], g1['opacity'], g1['color']], dim=-1)
        params_concat = torch.cat([params_0, params_1], dim=-1)  # (B, num_points, 28)

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

        # Add residuals to interpolation baseline
        return {
            'xyz': xyz_linear + xyz_residual,
            'scale': F.softplus(scale_linear + scale_residual),
            'rotation': F.normalize(rot_linear + 0.1 * rot_residual, dim=-1),  # re-normalize quaternion
            'opacity': torch.sigmoid(torch.logit(opacity_linear.clamp(1e-6, 1-1e-6)) + opacity_residual),
            'color': torch.sigmoid(torch.logit(color_linear.clamp(1e-6, 1-1e-6)) + color_residual),
        }

    @staticmethod
    def _slerp(q0: torch.Tensor, q1: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Spherical linear interpolation between unit quaternions.

        Args:
            q0: Start quaternions (..., 4)
            q1: End quaternions (..., 4)
            alpha: Interpolation weight (..., 1) in [0, 1]

        Returns:
            Interpolated unit quaternion (..., 4)
        """
        dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        # Ensure shortest arc
        q1 = torch.where(dot < 0, -q1, q1)
        dot = dot.abs()
        omega = torch.acos(dot.clamp(max=1.0 - 1e-6))
        sin_omega = torch.sin(omega).clamp(min=1e-6)
        # Fall back to linear interp when quaternions are nearly identical
        slerp = (torch.sin((1 - alpha) * omega) / sin_omega * q0
                 + torch.sin(alpha * omega) / sin_omega * q1)
        lerp = (1 - alpha) * q0 + alpha * q1
        return F.normalize(torch.where(sin_omega < 1e-6, lerp, slerp), dim=-1)

    @staticmethod
    def _to_map(tensor: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Reshape (B, N, C) -> (B, C, H, W)."""
        B, N, C = tensor.shape
        return tensor.permute(0, 2, 1).reshape(B, C, H, W)

    @staticmethod
    def _from_map(tensor: torch.Tensor) -> torch.Tensor:
        """Reshape (B, C, H, W) -> (B, N, C)."""
        B, C, H, W = tensor.shape
        return tensor.reshape(B, C, -1).permute(0, 2, 1)

    @staticmethod
    def _warp_map(feat_map: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp a spatial feature map by optical flow.

        Args:
            feat_map: (B, C, H, W)
            flow: (B, 2, H, W) displacement in pixels (dx, dy)

        Returns:
            Warped feature map (B, C, H, W)
        """
        B, C, H, W = feat_map.shape
        # Normalize pixel displacements to [-1, 1]
        flow_norm = torch.stack([
            flow[:, 0] / (W / 2),
            flow[:, 1] / (H / 2),
        ], dim=1)  # (B, 2, H, W)

        # Base identity grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feat_map.device, dtype=feat_map.dtype),
            torch.linspace(-1, 1, W, device=feat_map.device, dtype=feat_map.dtype),
            indexing='ij',
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

        sampling_grid = grid + flow_norm.permute(0, 2, 3, 1)  # (B, H, W, 2)
        return F.grid_sample(feat_map, sampling_grid, mode='bilinear',
                             padding_mode='border', align_corners=True)


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
            gaussian_dim: int = 14,
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

        # Split into parameters (xyz=3, scale=3, rotation=4, opacity=1, color=3 → total 14)
        return {
            'xyz': out[..., :3],
            'scale': F.softplus(out[..., 3:6]),
            'rotation': F.normalize(out[..., 6:10], dim=-1),  # unit quaternion
            'opacity': torch.sigmoid(out[..., 10:11]),
            'color': torch.sigmoid(out[..., 11:14]),
        }
