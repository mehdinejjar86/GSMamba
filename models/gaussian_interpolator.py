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
                 motion_frames_k: int = 0, feat_dim: int = 0, motion_accel: bool = False):
        super().__init__()
        self.motion_frames_k = motion_frames_k
        self.feat_dim = feat_dim                       # per-Gaussian latent width (Phase 4); 0 = off
        self.motion_accel = motion_accel
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

        use_motion = self.feat_dim > 0   # Phase 4: latent-augmented mixing + Mamba motion head

        # Per-Gaussian token = 14 params [+ F latent when motion is on].
        pack_keys = ['xyz', 'scale', 'rotation', 'opacity', 'color']
        if use_motion:
            pack_keys.append('feat')

        # Pack each frame sorted by 3D Morton code so spatially proximate Gaussians
        # from different frames become adjacent in the Mamba sequence.  Keep the
        # permutations so we can unsort back to pixel order for motion synthesis.
        sorted_frames = []
        perms = []
        for n in range(N):
            params_n = torch.cat([gaussians_list[n][k] for k in pack_keys], dim=-1)  # (B, HW, D)
            codes = morton_encode_3d(params_n[..., :3])            # (B, HW) — xyz only
            perm  = codes.argsort(dim=1)                           # (B, HW)
            perms.append(perm)
            sorted_frames.append(
                params_n.gather(1, perm.unsqueeze(-1).expand_as(params_n))
            )
        all_params = torch.cat(sorted_frames, dim=1)               # (B, N·HW, D)

        # Temporal position encoding — each HW block shares its frame's timestamp
        ts_per_pos = timestamps.unsqueeze(-1).expand(B, N, HW).reshape(B, N * HW)
        all_params = self.pos_encoding(all_params, ts_per_pos)

        # Mamba refinement: (B, N·HW, 14) — CUDA kernel, O(D·N) state
        for layer in self.layers:
            all_params = layer(all_params)
        all_params = self.norm(all_params)

        # Find the two bounding frames for time t
        idx0, idx1, alpha = _find_bounding_frames(t, timestamps)
        a = alpha.view(B, 1, 1)  # broadcast over (HW, C)

        if not use_motion:
            # ---- Default synthesis: blend the two Morton-sorted bounding blocks ----
            g0 = torch.stack([all_params[b, idx0[b] * HW:(idx0[b] + 1) * HW] for b in range(B)])
            g1 = torch.stack([all_params[b, idx1[b] * HW:(idx1[b] + 1) * HW] for b in range(B)])

            xyz_t   = (1 - a) * g0[..., :3]   + a * g1[..., :3]
            scale_t = F.softplus((1 - a) * g0[..., 3:6]  + a * g1[..., 3:6])
            rot_t   = self._slerp(
                          F.normalize(g0[..., 6:10], dim=-1),
                          F.normalize(g1[..., 6:10], dim=-1),
                          a)
            opac_t  = torch.sigmoid((1 - a) * g0[..., 10:11] + a * g1[..., 10:11])
            color_t = torch.sigmoid((1 - a) * g0[..., 11:14] + a * g1[..., 11:14])

            return {
                'xyz': xyz_t,
                'scale': scale_t,
                'rotation': rot_t,
                'opacity': opac_t,
                'color': color_t,
            }

        # ---- Motion synthesis (Phase 4): mixed-token motion head + forward-warp + merge ----
        # Unsort each frame's cross-frame-mixed token back to pixel order, read per-Gaussian
        # motion from the mixed token (the motion head), advect+rotate every (or the K nearest)
        # frame's Gaussians to time t, and merge the warped clouds weighted by temporal
        # proximity.  The rasterizer composites by depth (motion, rotation, disocclusion).
        refined = []
        for n in range(N):
            block_n = all_params[:, n * HW:(n + 1) * HW, :]        # (B, HW, D) sorted
            inv = perms[n].argsort(dim=1)                          # inverse perm -> pixel order
            refined.append(block_n.gather(1, inv.unsqueeze(-1).expand_as(block_n)))
        refined = torch.stack(refined, dim=0)                      # (N, B, HW, D) pixel order

        # Per-Gaussian motion read from the cross-frame-mixed tokens.
        motion = self.motion_head(refined)                         # (N, B, HW, 6 or 9)
        vel = motion[..., 0:3]
        ang = motion[..., 3:6]
        acc = motion[..., 6:9] if self.motion_accel else None

        # Per-frame time delta to the (unclamped) query time t:  Δ_n = t - t_n
        if not torch.is_tensor(t):
            tq = torch.full((B,), float(t), device=device, dtype=timestamps.dtype)
        else:
            tq = t.to(device=device, dtype=timestamps.dtype).reshape(-1)
            if tq.numel() == 1:
                tq = tq.expand(B)
        delta = tq.view(B, 1) - timestamps                        # (B, N)

        # Frames to warp: all N, or the K nearest to t (per batch element).
        K = self.motion_frames_k
        if K and 0 < K < N:
            sel = (-delta.abs()).topk(K, dim=1).indices           # (B, K)
        else:
            sel = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # (B, N)
        nf = sel.shape[1]
        bo = torch.arange(B, device=device).view(B, 1)            # (B, 1), broadcasts with sel

        g_sel = refined[sel, bo]                                   # (B, nf, HW, D)  (use [:14])
        v_sel = vel[sel, bo]                                       # (B, nf, HW, 3)
        w_sel = ang[sel, bo]                                       # (B, nf, HW, 3)
        a_sel = acc[sel, bo] if acc is not None else None
        d_sel = delta.gather(1, sel)                              # (B, nf)

        # Temporal-proximity opacity weights over the selected frames.
        wgt = torch.softmax(-d_sel.abs() / 0.1, dim=1)           # (B, nf)
        d = d_sel.view(B, nf, 1, 1)                               # broadcast over (HW, ·)

        # Advect positions: x(t) = x_n + v_n·Δ (+ ½·a_n·Δ²)
        xyz = g_sel[..., :3] + v_sel * d
        if a_sel is not None:
            xyz = xyz + 0.5 * a_sel * (d * d)

        # Rotate orientations to t: q(t) = exp(ω_n·Δ) ⊗ q_n
        q_t = F.normalize(
            quat_mul(so3_exp_to_quat(w_sel * d), F.normalize(g_sel[..., 6:10], dim=-1)),
            dim=-1)

        scale = F.softplus(g_sel[..., 3:6])
        opac = torch.sigmoid(g_sel[..., 10:11]) * wgt.view(B, nf, 1, 1)
        color = torch.sigmoid(g_sel[..., 11:14])

        # Merge the nf warped clouds -> (B, nf·HW, ·); rasterizer composites by depth.
        return {
            'xyz':      xyz.reshape(B, nf * HW, 3),
            'scale':    scale.reshape(B, nf * HW, 3),
            'rotation': q_t.reshape(B, nf * HW, 4),
            'opacity':  opac.reshape(B, nf * HW, 1),
            'color':    color.reshape(B, nf * HW, 3),
        }
