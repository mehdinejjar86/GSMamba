"""
Temporal SSM Block

1D Mamba-style selective state space model for temporal sequence processing.
Used for cross-frame fusion in GS-Mamba.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


class TemporalSSMBlock(nn.Module):
    """
    Temporal Selective State Space Block.

    Processes sequences along the temporal dimension using Mamba's
    selective scan mechanism. Designed for variable-length frame sequences.

    Args:
        d_model: Input/output feature dimension
        d_state: SSM state dimension (default: 16)
        d_conv: Causal conv kernel size (default: 4)
        expand: Expansion factor (default: 2)
        bidirectional: Use bidirectional processing (default: True)
    """

    def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            bidirectional: bool = True,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)
        self.bidirectional = bidirectional

        # Input projection (expands to 2x for gating)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal convolution for local temporal context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        self.act = nn.SiLU()

        # SSM projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt_proj bias for stable training
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # A matrix (log space for stability)
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Backward direction (if bidirectional)
        if bidirectional:
            self.conv1d_back = nn.Conv1d(
                self.d_inner, self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.d_inner
            )
            self.x_proj_back = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
            self.dt_proj_back = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            with torch.no_grad():
                self.dt_proj_back.bias.copy_(inv_dt)

            self.A_log_back = nn.Parameter(torch.log(A.clone()))
            self.A_log_back._no_weight_decay = True
            self.D_back = nn.Parameter(torch.ones(self.d_inner))
            self.D_back._no_weight_decay = True

            # Merge projection for bidirectional
            self.merge_proj = nn.Linear(self.d_inner * 2, self.d_inner, bias=False)

    def _ssm_forward(self, x, conv1d, x_proj, dt_proj, A_log, D):
        """
        Single-direction SSM forward pass.

        Args:
            x: Input (B, L, D_inner)
            conv1d: Causal conv layer
            x_proj: Projection for dt, B, C
            dt_proj: Delta projection
            A_log: Log of A matrix
            D: Skip connection

        Returns:
            Output (B, L, D_inner)
        """
        B, L, _ = x.shape

        # Causal conv
        x_conv = x.transpose(1, 2)  # (B, D, L)
        x_conv = conv1d(x_conv)[:, :, :L]  # Causal: remove future padding
        x_conv = x_conv.transpose(1, 2)  # (B, L, D)
        x_conv = self.act(x_conv)

        # Project to SSM parameters
        x_dbl = x_proj(x_conv)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Delta projection
        dt = dt_proj(dt)  # (B, L, D_inner)
        dt = F.softplus(dt)

        # Get A, D
        A = -torch.exp(A_log)  # (D_inner, d_state)

        # Selective scan
        y = self._selective_scan(x_conv, dt, A, B_ssm, C_ssm, D)

        return y

    def _selective_scan(self, u, delta, A, B, C, D):
        """
        Selective scan implementation.

        Args:
            u: Input (B, L, D)
            delta: Time step (B, L, D)
            A: State matrix (D, N)
            B: Input modulation (B, L, N)
            C: Output modulation (B, L, N)
            D: Skip connection (D,)

        Returns:
            Output (B, L, D)
        """
        B_batch, L, D_dim = u.shape
        N = A.shape[1]

        # Discretize
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, D, N)

        # Scan
        x = torch.zeros(B_batch, D_dim, N, device=u.device, dtype=u.dtype)
        ys = []

        for i in range(L):
            x = deltaA[:, i] * x + deltaB[:, i] * u[:, i:i+1].transpose(1, 2)
            y = (x * C[:, i].unsqueeze(1)).sum(-1)  # (B, D)
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (B, L, D)

        # Skip connection
        y = y + u * D.unsqueeze(0).unsqueeze(0)

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, L, D) where L is sequence length (number of frames)

        Returns:
            Output tensor (B, L, D)
        """
        residual = x
        x = self.norm(x)

        # Input projection
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (B, L, D_inner) each

        # Forward direction
        y_fwd = self._ssm_forward(
            x, self.conv1d, self.x_proj, self.dt_proj, self.A_log, self.D
        )

        if self.bidirectional:
            # Backward direction (reverse sequence)
            x_back = torch.flip(x, dims=[1])
            y_back = self._ssm_forward(
                x_back, self.conv1d_back, self.x_proj_back,
                self.dt_proj_back, self.A_log_back, self.D_back
            )
            y_back = torch.flip(y_back, dims=[1])  # Reverse back

            # Merge forward and backward
            y = self.merge_proj(torch.cat([y_fwd, y_back], dim=-1))
        else:
            y = y_fwd

        # Gating
        y = y * F.silu(z)

        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)

        return residual + y


class TemporalPositionEncoding(nn.Module):
    """
    Sinusoidal position encoding for temporal sequences.

    Encodes the relative position (timestamp) of each frame in the sequence.

    Args:
        d_model: Feature dimension
        max_len: Maximum sequence length (default: 100)
    """

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor, timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add position encoding to input.

        Args:
            x: Input tensor (B, L, D)
            timestamps: Optional explicit timestamps (B, L). If None, uses sequential positions.

        Returns:
            Output tensor with position encoding added (B, L, D)
        """
        if timestamps is None:
            # Use sequential positions
            return x + self.pe[:, :x.size(1)]
        else:
            # Use explicit timestamps (normalized to [0, max_len-1])
            B, L = timestamps.shape
            indices = (timestamps * (self.pe.size(1) - 1)).long().clamp(0, self.pe.size(1) - 1)
            pe = self.pe[0, indices.view(-1)].view(B, L, -1)
            return x + pe
