"""
N-Frame Gaussian Mamba

Refines the merged N-frame Gaussian cloud with a Mamba SSM (3D Morton ordered)
and interpolates to an arbitrary query timestep.  Operates directly in Gaussian
parameter space (B, N·HW, 14) — O(D·N) CUDA state, no OOM at high resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math

from modules.temporal_ssm import TemporalSSMBlock, TemporalPositionEncoding


def _find_bounding_frames(
        t: Union[float, torch.Tensor],
        timestamps: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find the two frames that bound query timestep t.

    Args:
        t: Query timestep — scalar, (B,), (1,), or (B,1)
        timestamps: Frame timestamps (B, N) or (N,)

    Returns:
        (idx0, idx1, alpha) — idx0/idx1: (B,) frame indices; alpha: (B,) weight in [0,1]
    """
    if timestamps.dim() == 1:
        timestamps = timestamps.unsqueeze(0)

    B, N = timestamps.shape

    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, device=timestamps.device, dtype=timestamps.dtype)
    else:
        t = t.to(device=timestamps.device, dtype=timestamps.dtype)

    if t.dim() == 0:
        t = t.expand(B)
    elif t.dim() == 1:
        if t.shape[0] == 1 and B > 1:
            t = t.expand(B)
    elif t.dim() == 2 and t.shape[1] == 1:
        t = t.squeeze(1)
        if t.shape[0] == 1 and B > 1:
            t = t.expand(B)

    t = t.view(B, 1)

    diff = timestamps - t       # (B, N)
    diff[diff > 0] = float('-inf')
    idx0 = diff.argmax(dim=1)  # (B,)
    idx1 = (idx0 + 1).clamp(max=N - 1)

    t0 = timestamps.gather(1, idx0.unsqueeze(1)).squeeze(1)
    t1 = timestamps.gather(1, idx1.unsqueeze(1)).squeeze(1)

    dt = (t1 - t0).clamp(min=1e-6)
    alpha = ((t.squeeze(1) - t0) / dt).clamp(0, 1)  # (B,)

    return idx0, idx1, alpha


def morton_encode_3d(xyz: torch.Tensor, bits: int = 10) -> torch.Tensor:
    """
    Compute 3D Morton (Z-order) codes for a batch of 3D points.

    Normalises xyz per-batch element before quantising so absolute scale
    and scene size don't matter.

    Args:
        xyz:  (B, N, 3) float tensor — Gaussian xyz coordinates
        bits: bits per dimension (10 → 30-bit code, handles up to 1024³ grid)

    Returns:
        (B, N) int64 Morton codes — argsort of this gives spatial ordering
    """
    xyz_min = xyz.amin(dim=1, keepdim=True)            # (B, 1, 3)
    xyz_max = xyz.amax(dim=1, keepdim=True)            # (B, 1, 3)
    xyz_n = (xyz - xyz_min) / (xyz_max - xyz_min + 1e-6)  # (B, N, 3) in [0, 1]
    q = (xyz_n * ((1 << bits) - 1)).long().clamp(0, (1 << bits) - 1)  # quantise

    def spread(v: torch.Tensor) -> torch.Tensor:
        """Interleave bits with two zeros between each bit (10-bit input → 30-bit)."""
        v = v & 0x3ff
        v = (v | (v << 16)) & 0x030000ff
        v = (v | (v <<  8)) & 0x0300f00f
        v = (v | (v <<  4)) & 0x030c30c3
        v = (v | (v <<  2)) & 0x09249249
        return v

    x, y, z = q[..., 0], q[..., 1], q[..., 2]
    return spread(x) | (spread(y) << 1) | (spread(z) << 2)  # (B, N) int64


def so3_exp_to_quat(omega: torch.Tensor) -> torch.Tensor:
    """
    Map an axis-angle rotation vector to a unit quaternion (w, x, y, z).

    omega: (..., 3) — direction = rotation axis, magnitude = rotation angle.
    Returns (..., 4) unit quaternion. omega=0 -> identity [1, 0, 0, 0].
    """
    theta = omega.norm(dim=-1, keepdim=True)                  # (..., 1) angle
    half = 0.5 * theta
    # ratio = sin(half)/theta, with the stable small-angle limit -> 0.5
    ratio = torch.where(theta > 1e-6, torch.sin(half) / theta.clamp(min=1e-6),
                        torch.full_like(theta, 0.5))
    w = torch.cos(half)                                       # (..., 1)
    xyz = omega * ratio                                       # (..., 3)
    return torch.cat([w, xyz], dim=-1)                        # (..., 4)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of quaternions (w, x, y, z), broadcasting over leading dims."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


class NFrameGaussianMamba(nn.Module):
    """
    Mamba SSM over the merged N-frame Gaussian cloud.

    Operates directly in Gaussian parameter space (B, N·HW, 14) — O(D·N) CUDA
    state, so there is no OOM at high resolution.

    Memory pattern:
        Old TemporalFusion:   (B·H·W, N, C_feat) — huge batch, tiny seq → OOM
        NFrameGaussianMamba:  (B, N·H·W, 14)     — small batch, long seq → O(D·N) CUDA

    At 768×768, N=2, B=2:
        Old deltaA: (1.18M, 2, 256, 16) ≈ 36 GB  → OOM
        New state:  (2, 28, 16)          ≈ negligible

    Args:
        d_state: SSM state dimension (default: 16)
        expand: Inner dimension multiplier (default: 2, → d_inner = 28)
        num_layers: Number of TemporalSSMBlock layers (default: 2)
    """

    GAUSSIAN_DIM = 14  # xyz(3) + scale(3) + rotation(4) + opacity(1) + color(3)

    def __init__(self, d_state: int = 16, expand: int = 2, num_layers: int = 2,
                 motion_frames_k: int = 0, feat_dim: int = 0, motion_accel: bool = False,
                 motion_temporal_tau: float = 1.0):
        super().__init__()
        self.motion_frames_k = motion_frames_k
        self.feat_dim = feat_dim                       # per-Gaussian latent width (Phase 4); 0 = off
        self.motion_accel = motion_accel
        self.motion_temporal_tau = motion_temporal_tau
        D = self.GAUSSIAN_DIM + feat_dim               # token = params(14) [+ latent(F)]
        self.D = D
        self.pos_encoding = TemporalPositionEncoding(D, max_len=100)
        self.layers = nn.ModuleList([
            TemporalSSMBlock(d_model=D, d_state=d_state, expand=expand, bidirectional=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(D)

        # Motion head (Phase 4): reads the cross-frame-mixed token -> per-Gaussian motion
        # = velocity(3) + angular velocity(3) [+ acceleration(3)]. Zero-init so synthesis
        # starts motion-free. Only built when latent mixing is on (feat_dim > 0).
        self.motion_head = None
        if feat_dim > 0:
            motion_out = 6 + (3 if motion_accel else 0)
            self.motion_head = nn.Linear(D, motion_out)
            nn.init.zeros_(self.motion_head.weight)
            nn.init.zeros_(self.motion_head.bias)

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
    def _pack(gaussians_list: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Pack N Gaussian dicts into (B, N·HW, 14)."""
        return torch.cat([
            torch.cat([g['xyz'], g['scale'], g['rotation'], g['opacity'], g['color']], dim=-1)
            for g in gaussians_list
        ], dim=1)

    def forward(
            self,
            gaussians_list: List[Dict[str, torch.Tensor]],
            t: Union[float, torch.Tensor],
            timestamps: Optional[torch.Tensor] = None,
            flow: Optional[torch.Tensor] = None,  # accepted for API compat, not used
    ) -> Dict[str, torch.Tensor]:
        """
        Refine N Gaussian clouds with Mamba and interpolate to timestep t.

        Args:
            gaussians_list: N dicts, each with (B, HW, C) tensors
                            keys: xyz, scale, rotation, opacity, color
            t: Query timestep in [0, 1]
            timestamps: Frame timestamps (B, N) or (N,). None → uniform.
            flow: Unused (kept for drop-in compatibility / future internal motion).

        Returns:
            Gaussian dict at timestep t, shapes (B, HW, C).
        """
        B = gaussians_list[0]['xyz'].shape[0]
        N = len(gaussians_list)
        HW = gaussians_list[0]['xyz'].shape[1]
        device = gaussians_list[0]['xyz'].device

        if timestamps is None:
            timestamps = torch.linspace(0, 1, N, device=device).unsqueeze(0).expand(B, -1)
        elif timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(0).expand(B, -1)
        elif timestamps.dim() == 2 and timestamps.shape[0] == 1 and B > 1:
            timestamps = timestamps.expand(B, -1)

        # ---- Default path (no latent / no motion): per-frame Morton-sort, block-concat,
        # SSM, 2-frame Morton-rank blend.  Unchanged & checkpoint-safe. ----
        if self.feat_dim == 0:
            sorted_frames = []
            for n in range(N):
                params_n = torch.cat([
                    gaussians_list[n]['xyz'], gaussians_list[n]['scale'],
                    gaussians_list[n]['rotation'], gaussians_list[n]['opacity'],
                    gaussians_list[n]['color'],
                ], dim=-1)                                          # (B, HW, 14)
                codes = morton_encode_3d(params_n[..., :3])
                perm = codes.argsort(dim=1)
                sorted_frames.append(params_n.gather(1, perm.unsqueeze(-1).expand_as(params_n)))
            all_params = torch.cat(sorted_frames, dim=1)           # (B, N·HW, 14)
            ts_per_pos = timestamps.unsqueeze(-1).expand(B, N, HW).reshape(B, N * HW)
            all_params = self.pos_encoding(all_params, ts_per_pos)
            for layer in self.layers:
                all_params = layer(all_params)
            all_params = self.norm(all_params)

            idx0, idx1, alpha = _find_bounding_frames(t, timestamps)
            a = alpha.view(B, 1, 1)
            g0 = torch.stack([all_params[b, idx0[b] * HW:(idx0[b] + 1) * HW] for b in range(B)])
            g1 = torch.stack([all_params[b, idx1[b] * HW:(idx1[b] + 1) * HW] for b in range(B)])
            return {
                'xyz':      (1 - a) * g0[..., :3] + a * g1[..., :3],
                'scale':    F.softplus((1 - a) * g0[..., 3:6] + a * g1[..., 3:6]),
                'rotation': self._slerp(F.normalize(g0[..., 6:10], dim=-1),
                                        F.normalize(g1[..., 6:10], dim=-1), a),
                'opacity':  torch.sigmoid((1 - a) * g0[..., 10:11] + a * g1[..., 10:11]),
                'color':    torch.sigmoid((1 - a) * g0[..., 11:14] + a * g1[..., 11:14]),
            }

        # ---- Motion path (Phase 4): JOINT Morton ordering + latent mixing + motion head ----
        # A joint Morton sort over ALL selected frames' Gaussians makes correspondences
        # (and static content) adjacent in the sequence — not block-separated by frame — so
        # the SSM can actually mix them.  Δ uses each token's ACTUAL timestamp, so
        # non-uniform frame spacing is handled correctly.
        if not torch.is_tensor(t):
            tq = torch.full((B,), float(t), device=device, dtype=timestamps.dtype)
        else:
            tq = t.to(device=device, dtype=timestamps.dtype).reshape(-1)
            if tq.numel() == 1:
                tq = tq.expand(B)

        # Per-frame packed tokens [params(14) + feat(F)] -> (N, B, HW, D)
        per_frame = torch.stack([
            torch.cat([gaussians_list[n]['xyz'], gaussians_list[n]['scale'],
                       gaussians_list[n]['rotation'], gaussians_list[n]['opacity'],
                       gaussians_list[n]['color'], gaussians_list[n]['feat']], dim=-1)
            for n in range(N)
        ], dim=0)                                                  # (N, B, HW, D)

        # Choose frames to advect+merge: all N, or the K nearest to t (per batch element).
        delta_f = tq.view(B, 1) - timestamps                       # (B, N)
        K = self.motion_frames_k
        if K and 0 < K < N:
            sel = (-delta_f.abs()).topk(K, dim=1).indices          # (B, M)
        else:
            sel = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # (B, N)
        M = sel.shape[1]
        bo = torch.arange(B, device=device).view(B, 1)
        tok_f = per_frame[sel, bo]                                 # (B, M, HW, D)
        ts_sel = timestamps.gather(1, sel)                         # (B, M)

        # Spacing-aware temporal opacity weight over selected frames (convex; tau<=0 -> uniform).
        if self.motion_temporal_tau > 0:
            avg_gap = ((timestamps.amax(dim=1) - timestamps.amin(dim=1)).clamp(min=1e-6)
                       / max(N - 1, 1))                            # (B,)
            tau_scale = (self.motion_temporal_tau * avg_gap).view(B, 1).clamp(min=1e-6)
            fw = torch.softmax(-(tq.view(B, 1) - ts_sel).abs() / tau_scale, dim=1)  # (B, M)
        else:
            fw = torch.full((B, M), 1.0 / M, device=device, dtype=timestamps.dtype)

        # Flatten to one token set, carrying per-token timestamp + temporal weight.
        D = self.D
        tok = tok_f.reshape(B, M * HW, D)
        ts_tok = ts_sel.unsqueeze(-1).expand(B, M, HW).reshape(B, M * HW)
        w_tok = fw.unsqueeze(-1).expand(B, M, HW).reshape(B, M * HW)

        # JOINT Morton sort across frames so correspondences are adjacent in the sequence.
        codes = morton_encode_3d(tok[..., :3])                    # (B, M·HW)
        perm = codes.argsort(dim=1)                               # (B, M·HW)
        tok = tok.gather(1, perm.unsqueeze(-1).expand_as(tok))
        ts_tok = ts_tok.gather(1, perm)
        w_tok = w_tok.gather(1, perm)

        # Temporal position encoding (actual timestamps) -> SSM mix -> norm.
        tok = self.pos_encoding(tok, ts_tok)
        for layer in self.layers:
            tok = layer(tok)
        tok = self.norm(tok)

        # Per-token motion from the cross-frame-mixed token.
        motion = self.motion_head(tok)                            # (B, M·HW, 6 or 9)
        v = motion[..., 0:3]
        w = motion[..., 3:6]
        acc = motion[..., 6:9] if self.motion_accel else None

        # Advect + rotate each token to t by its own Δ = t - t_token.
        d = (tq.view(B, 1) - ts_tok).unsqueeze(-1)               # (B, M·HW, 1)
        xyz = tok[..., :3] + v * d
        if acc is not None:
            xyz = xyz + 0.5 * acc * (d * d)
        q_t = F.normalize(quat_mul(so3_exp_to_quat(w * d),
                                   F.normalize(tok[..., 6:10], dim=-1)), dim=-1)
        scale = F.softplus(tok[..., 3:6])
        opac = torch.sigmoid(tok[..., 10:11]) * w_tok.unsqueeze(-1)
        color = torch.sigmoid(tok[..., 11:14])

        # Merged cloud at time t (all selected, advected tokens); rasterizer composites by depth.
        return {
            'xyz': xyz, 'scale': scale, 'rotation': q_t, 'opacity': opac, 'color': color,
        }
