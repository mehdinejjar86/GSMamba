"""
SS2D - Selective Scan 2D

Adapted from VFIMamba for GS-Mamba.
Core Mamba block for 2D spatial feature processing with 4-directional scanning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from typing import Optional

# Try to import the optimized selective scan, fall back to reference if not available
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None
    print("Warning: mamba_ssm not found. Using reference implementation (slower).")


class SS2D(nn.Module):
    """
    Selective Scan 2D Block.

    Processes 2D spatial features using Mamba's selective state space mechanism.
    Uses 4-directional scanning (HW, WH, reversed) for comprehensive spatial coverage.

    Args:
        d_model: Input/output feature dimension
        d_state: SSM state dimension (default: 16)
        d_conv: Convolution kernel size (default: 3)
        expand: Expansion factor for inner dimension (default: 2.0)
        dt_rank: Rank for delta projections (default: auto)
        dropout: Dropout rate (default: 0.0)
    """

    def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 3,
            expand: float = 2.0,
            dt_rank: str = "auto",
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            dt_init: str = "random",
            dt_scale: float = 1.0,
            dt_init_floor: float = 1e-4,
            dropout: float = 0.0,
            conv_bias: bool = True,
            bias: bool = False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # Depthwise conv for local context
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # SSM projections for 4 directions
        self.x_proj_weight = nn.Parameter(
            torch.randn(4, self.d_inner, self.dt_rank + self.d_state * 2, **factory_kwargs) * 0.02
        )

        # Delta (timestep) projections
        self.dt_projs_weight = nn.Parameter(
            torch.randn(4, self.d_inner, self.dt_rank, **factory_kwargs) * 0.02
        )
        self.dt_projs_bias = nn.Parameter(self._init_dt_bias(d_inner=self.d_inner, copies=4, **factory_kwargs))

        # SSM parameters
        self.A_logs = self._init_A_log(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self._init_D(self.d_inner, copies=4, merge=True)

        # Output projection
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # Use optimized kernel if available
        self.selective_scan = selective_scan_fn if selective_scan_fn else self._selective_scan_ref

    @staticmethod
    def _init_dt_bias(d_inner, copies=4, dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        """Initialize delta bias for stable training."""
        dt = torch.exp(
            torch.rand(copies, d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        return inv_dt

    @staticmethod
    def _init_A_log(d_state, d_inner, copies=1, merge=True, device=None):
        """Initialize A matrix in log space."""
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def _init_D(d_inner, copies=1, merge=True, device=None):
        """Initialize D (skip connection) parameter."""
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def _selective_scan_ref(self, u, delta, A, B, C, D, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
        """Reference implementation of selective scan (slower, for fallback)."""
        batch, dim, length = u.shape
        n_state = A.shape[1]

        if delta_bias is not None:
            delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)
        if delta_softplus:
            delta = F.softplus(delta)

        # Discretize
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        deltaB = torch.einsum('bdl,bknl->bdkln', delta, B.unsqueeze(2).expand(-1, -1, dim, -1, -1))

        # Scan
        x = torch.zeros(batch, dim, n_state, device=u.device, dtype=u.dtype)
        ys = []
        for i in range(length):
            x = deltaA[:, :, i] * x + deltaB[:, :, :, i] * u[:, :, i:i+1]
            y = torch.einsum('bdn,bkn->bd', x, C[:, :, :, i])
            ys.append(y)
        y = torch.stack(ys, dim=-1)

        if D is not None:
            y = y + u * D.unsqueeze(0).unsqueeze(-1)

        return y

    def forward_core(self, x: torch.Tensor):
        """
        Core selective scan with 4-directional processing.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Tuple of 4 directional outputs
        """
        B, C, H, W = x.shape
        L = H * W
        K = 4  # 4 directions

        # Flatten spatial dimensions
        x_flat = x.view(B, C, L)

        # Create 4 directional sequences
        x_hw = x_flat  # H*W order
        x_wh = x.transpose(2, 3).contiguous().view(B, C, L)  # W*H order
        x_hw_rev = torch.flip(x_hw, dims=[-1])  # Reversed H*W
        x_wh_rev = torch.flip(x_wh, dims=[-1])  # Reversed W*H

        xs = torch.stack([x_hw, x_wh, x_hw_rev, x_wh_rev], dim=1)  # (B, 4, C, L)

        # Project to SSM parameters
        x_dbl = torch.einsum("bkcl,kcd->bkdl", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        # Delta projection
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_projs_weight)

        # Prepare for selective scan
        xs_flat = xs.view(B, -1, L)  # (B, 4*C, L)
        dts_flat = dts.contiguous().view(B, -1, L)
        Bs_flat = Bs.view(B, K, -1, L)
        Cs_flat = Cs.view(B, K, -1, L)
        Ds = self.Ds.view(-1)
        As = -torch.exp(self.A_logs).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.view(-1)

        # Selective scan
        out_y = self.selective_scan(
            xs_flat, dts_flat,
            As, Bs_flat, Cs_flat, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        # Reverse the reversed directions
        y_hw = out_y[:, 0]
        y_wh = out_y[:, 1]
        y_hw_rev = torch.flip(out_y[:, 2], dims=[-1])
        y_wh_rev = torch.flip(out_y[:, 3], dims=[-1])

        # Reshape W*H outputs back to H*W
        y_wh = y_wh.view(B, -1, W, H).transpose(2, 3).contiguous().view(B, -1, L)
        y_wh_rev = y_wh_rev.view(B, -1, W, H).transpose(2, 3).contiguous().view(B, -1, L)

        return y_hw, y_wh, y_hw_rev, y_wh_rev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, H, W, C)

        Returns:
            Output tensor (B, H, W, C)
        """
        B, H, W, C = x.shape

        # Input projection and split
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Conv for local context
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        # 4-directional selective scan
        y1, y2, y3, y4 = self.forward_core(x)

        # Combine directions
        y = y1 + y2 + y3 + y4
        y = y.view(B, -1, H, W).permute(0, 2, 3, 1).contiguous()

        # Output projection with gating
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        out = self.dropout(out)

        return out
