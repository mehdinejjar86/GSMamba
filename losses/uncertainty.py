# gsmamba/losses/uncertainty.py
"""
Uncertainty-Based Loss Weighting for GS-Mamba

Implements Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses
for Scene Geometry and Semantics" (CVPR 2018)

Instead of hand-tuning fixed loss weights, this module learns optimal weights
automatically based on homoscedastic uncertainty:

    L_total = Σᵢ (1/(2σᵢ²) * Lᵢ + log(σᵢ))

Where σᵢ is a learnable parameter per loss that represents task uncertainty.
Higher uncertainty → lower effective weight.
"""

import math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn


class UncertaintyWeighting(nn.Module):
    """
    Learnable loss weighting based on homoscedastic uncertainty.

    Each loss component gets a learnable log-variance parameter (log_var = log(σ²)).
    The effective weight is the precision: 1/σ² = exp(-log_var).

    The total loss becomes:
        L = Σᵢ (0.5 * precision_i * loss_i + 0.5 * log_var_i)

    The log_var term prevents σ → ∞ (which would make weight → 0).

    Usage:
        uncertainty = UncertaintyWeighting(['l1', 'ssim', 'lap'])
        total, weighted = uncertainty({'l1': l1_loss, 'ssim': ssim_loss, 'lap': lap_loss})
    """

    def __init__(
        self,
        loss_names: List[str],
        initial_log_vars: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            loss_names: List of loss component names
            initial_log_vars: Initial log-variance values per loss.
                             Lower values → higher initial weight.
                             log_var=0 → σ=1 → weight=1
                             log_var=1 → σ≈1.65 → weight≈0.37
                             log_var=2 → σ≈2.72 → weight≈0.14
        """
        super().__init__()
        self.loss_names = loss_names

        # Default initial log-variances (tuned for PSNR/SSIM maximization)
        defaults = {
            'l1': 0.0,       # σ=1.0, high weight (PSNR)
            'ssim': 0.0,     # σ=1.0, high weight (SSIM metric!)
            'lap': 1.0,      # σ≈1.65, medium weight
            'recon': 1.5,    # σ≈2.1, lower weight
            'depth': 2.0,    # σ≈2.7, regularization
            'temporal': 2.0, # σ≈2.7, regularization
            'gflow': 1.5,    # σ≈2.1, auxiliary
            'opacity_reg': 2.5,  # σ≈3.5, minimal
            'scale_reg': 2.5,    # σ≈3.5, minimal
        }

        if initial_log_vars:
            defaults.update(initial_log_vars)

        # Create learnable parameters
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(defaults.get(name, 0.0), dtype=torch.float32))
            for name in loss_names
        })

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply uncertainty weighting to losses.

        Args:
            losses: Dict of {loss_name: loss_value} (unweighted losses)

        Returns:
            total_loss: Sum of uncertainty-weighted losses
            weighted_losses: Dict of {loss_name: weighted_loss_value}
        """
        device = next(iter(losses.values())).device
        total = torch.tensor(0.0, device=device)
        weighted = {}

        for name, loss in losses.items():
            if name in self.log_vars:
                log_var = self.log_vars[name]
                # Precision = 1/σ² = exp(-log_var)
                precision = torch.exp(-log_var)
                # Weighted loss = 0.5 * precision * loss + 0.5 * log_var
                weighted_loss = 0.5 * precision * loss + 0.5 * log_var
                weighted[name] = weighted_loss
                total = total + weighted_loss
            else:
                # Unknown loss, add directly
                weighted[name] = loss
                total = total + loss

        return total, weighted

    def get_weights(self) -> Dict[str, float]:
        """
        Get current effective weights (precision = 1/σ²).

        Returns:
            Dict of {loss_name: effective_weight}
        """
        return {
            name: torch.exp(-lv).item()
            for name, lv in self.log_vars.items()
        }

    def get_sigmas(self) -> Dict[str, float]:
        """
        Get current σ values.

        Returns:
            Dict of {loss_name: sigma}
        """
        return {
            name: torch.exp(0.5 * lv).item()
            for name, lv in self.log_vars.items()
        }

    def get_log_vars(self) -> Dict[str, float]:
        """
        Get current log-variance values.

        Returns:
            Dict of {loss_name: log_var}
        """
        return {
            name: lv.item()
            for name, lv in self.log_vars.items()
        }

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get comprehensive statistics for logging.

        Returns:
            Dict with 'weights', 'sigmas', 'log_vars' sub-dicts
        """
        return {
            'weights': self.get_weights(),
            'sigmas': self.get_sigmas(),
            'log_vars': self.get_log_vars(),
        }

    def __repr__(self) -> str:
        weights = self.get_weights()
        items = [f"{name}={w:.3f}" for name, w in sorted(weights.items(), key=lambda x: -x[1])]
        return f"UncertaintyWeighting(weights=[{', '.join(items)}])"


class ProgressiveUncertaintyWeighting(UncertaintyWeighting):
    """
    Uncertainty weighting with progressive warmup.

    During warmup, uses fixed weights. After warmup, transitions to learned weights.
    """

    def __init__(
        self,
        loss_names: List[str],
        initial_log_vars: Optional[Dict[str, float]] = None,
        fixed_weights: Optional[Dict[str, float]] = None,
        warmup_epochs: int = 10,
    ):
        """
        Args:
            loss_names: List of loss component names
            initial_log_vars: Initial log-variance values
            fixed_weights: Fixed weights to use during warmup
            warmup_epochs: Number of epochs before transitioning to learned weights
        """
        super().__init__(loss_names, initial_log_vars)

        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        # Default fixed weights for warmup
        self.fixed_weights = fixed_weights or {
            'l1': 1.0,
            'ssim': 1.0,
            'lap': 0.1,
            'recon': 0.1,
            'depth': 0.001,
            'temporal': 0.01,
            'gflow': 0.05,
            'opacity_reg': 0.001,
            'scale_reg': 0.001,
        }

    def set_epoch(self, epoch: int):
        """Set current epoch for warmup scheduling."""
        self.current_epoch = epoch

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply weighting based on warmup status."""
        if self.current_epoch < self.warmup_epochs:
            # During warmup: use fixed weights
            return self._apply_fixed_weights(losses)
        else:
            # After warmup: use learned uncertainty weights
            return super().forward(losses)

    def _apply_fixed_weights(
        self,
        losses: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply fixed weights during warmup."""
        device = next(iter(losses.values())).device
        total = torch.tensor(0.0, device=device)
        weighted = {}

        for name, loss in losses.items():
            weight = self.fixed_weights.get(name, 1.0)
            weighted_loss = weight * loss
            weighted[name] = weighted_loss
            total = total + weighted_loss

        return total, weighted

    def is_warmup(self) -> bool:
        """Check if still in warmup phase."""
        return self.current_epoch < self.warmup_epochs
