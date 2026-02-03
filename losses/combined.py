"""
Combined GS-Mamba Loss

Combines all loss components with configurable weights.
Supports both fixed weights and uncertainty-based learned weights (Kendall et al.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from losses.photometric import L1Loss, SSIMLoss, LaplacianLoss
from losses.perceptual import LPIPSLoss
from losses.geometric import (
    DepthSmoothLoss,
    TemporalConsistencyLoss,
    OpacityRegularizationLoss,
    ScaleRegularizationLoss,
)
from losses.gaussian_flow import GaussianFlowLoss, get_gflow_weight
from losses.uncertainty import UncertaintyWeighting, ProgressiveUncertaintyWeighting


class GSMambaLoss(nn.Module):
    """
    Combined loss for GS-Mamba training.

    Combines:
    - Photometric: L1, SSIM, Laplacian pyramid
    - Perceptual: LPIPS (disabled by default for PSNR/SSIM focus)
    - Reconstruction: Input frame reconstruction
    - Geometric: Depth smoothness, temporal consistency
    - Gaussian Flow: 2D flow supervision (with decay)

    Supports two weighting modes:
    1. Fixed weights (manual tuning)
    2. Uncertainty-based learned weights (Kendall et al.)

    Args:
        w_photo: Weight for L1 loss (default: 1.0)
        w_ssim: Weight for SSIM loss (default: 1.0)
        w_lap: Weight for Laplacian loss (default: 0.1)
        w_lpips: Weight for LPIPS loss (default: 0.0)
        w_recon: Weight for reconstruction loss (default: 0.1)
        w_depth: Weight for depth smoothness (default: 0.001)
        w_temporal: Weight for temporal consistency (default: 0.01)
        w_gflow_max: Max weight for Gaussian flow (default: 0.05)
        gflow_decay_fraction: Fraction of training for gflow decay (default: 0.3)
        w_opacity_reg: Weight for opacity regularization (default: 0.001)
        w_scale_reg: Weight for scale regularization (default: 0.001)
        use_lpips: Whether to use LPIPS loss (default: False)
        use_gflow: Whether to use Gaussian flow loss (default: True)
        use_uncertainty_weighting: Whether to use learned loss weights (default: True)
        uncertainty_warmup_epochs: Epochs before transitioning to learned weights (default: 5)
        initial_log_vars: Initial log-variance values for uncertainty weighting
        flow_net: Optional pretrained flow network for gflow loss
        image_size: Image size for renderer
    """

    def __init__(
            self,
            w_photo: float = 1.0,
            w_ssim: float = 1.0,
            w_lap: float = 0.1,
            w_lpips: float = 0.0,
            w_recon: float = 0.1,
            w_depth: float = 0.001,
            w_temporal: float = 0.01,
            w_gflow_max: float = 0.05,
            gflow_decay_fraction: float = 0.3,
            w_opacity_reg: float = 0.001,
            w_scale_reg: float = 0.001,
            use_lpips: bool = False,
            use_gflow: bool = True,
            use_uncertainty_weighting: bool = True,
            uncertainty_warmup_epochs: int = 5,
            initial_log_vars: Optional[Dict[str, float]] = None,
            flow_net: Optional[nn.Module] = None,
            image_size: Tuple[int, int] = (256, 256),
    ):
        super().__init__()

        # Fixed weights (used during warmup or when uncertainty disabled)
        self.w_photo = w_photo
        self.w_ssim = w_ssim
        self.w_lap = w_lap
        self.w_lpips = w_lpips
        self.w_recon = w_recon
        self.w_depth = w_depth
        self.w_temporal = w_temporal
        self.w_gflow_max = w_gflow_max
        self.gflow_decay_fraction = gflow_decay_fraction
        self.w_opacity_reg = w_opacity_reg
        self.w_scale_reg = w_scale_reg

        # Loss modules
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()
        self.lap_loss = LaplacianLoss()

        self.use_lpips = use_lpips
        if use_lpips:
            self.lpips_loss = LPIPSLoss()
        else:
            self.lpips_loss = None

        self.depth_loss = DepthSmoothLoss()
        self.temporal_loss = TemporalConsistencyLoss()
        self.opacity_reg = OpacityRegularizationLoss()
        self.scale_reg = ScaleRegularizationLoss()

        self.use_gflow = use_gflow
        if use_gflow:
            self.gflow_loss = GaussianFlowLoss(flow_net, image_size)
        else:
            self.gflow_loss = None

        # Uncertainty-based weighting
        self.use_uncertainty_weighting = use_uncertainty_weighting
        if use_uncertainty_weighting:
            # Loss names that will be weighted by uncertainty
            loss_names = ['l1', 'ssim', 'lap', 'recon', 'depth', 'temporal', 'gflow', 'opacity_reg', 'scale_reg']

            # Fixed weights for warmup phase
            fixed_weights = {
                'l1': w_photo,
                'ssim': w_ssim,
                'lap': w_lap,
                'recon': w_recon,
                'depth': w_depth,
                'temporal': w_temporal,
                'gflow': w_gflow_max,
                'opacity_reg': w_opacity_reg,
                'scale_reg': w_scale_reg,
            }

            self.uncertainty = ProgressiveUncertaintyWeighting(
                loss_names=loss_names,
                initial_log_vars=initial_log_vars,
                fixed_weights=fixed_weights,
                warmup_epochs=uncertainty_warmup_epochs,
            )
        else:
            self.uncertainty = None

        # Training state
        self.current_epoch = 0
        self.total_epochs = 100

    def set_epoch(self, epoch: int, total_epochs: int):
        """Update current training epoch for loss scheduling."""
        self.current_epoch = epoch
        self.total_epochs = total_epochs

        # Update uncertainty weighting epoch
        if self.uncertainty is not None:
            self.uncertainty.set_epoch(epoch)

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            render: torch.Tensor,
            depth: torch.Tensor,
            input_frames: torch.Tensor,
            gaussians_list: List[Dict[str, torch.Tensor]],
            gaussians_interp: Dict[str, torch.Tensor],
            t: float,
            return_components: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            pred: Predicted frame after refinement (B, 3, H, W)
            target: Ground truth frame (B, 3, H, W)
            render: Coarse rendered frame before refinement (B, 3, H, W)
            depth: Depth map (B, 1, H, W)
            input_frames: Input frames (B, N, 3, H, W)
            gaussians_list: List of Gaussian dicts for each input frame
            gaussians_interp: Interpolated Gaussians at timestep t
            t: Interpolation timestep
            return_components: Whether to return individual loss components

        Returns:
            Dict with 'total' loss and optionally individual components
        """
        B, N, C, H, W = input_frames.shape
        device = pred.device

        # Compute RAW (unweighted) losses first
        raw_losses = {}

        # ========== Photometric Losses ==========
        raw_losses['l1'] = self.l1_loss(pred, target)
        raw_losses['ssim'] = self.ssim_loss(pred, target)
        raw_losses['lap'] = self.lap_loss(pred, target)

        # ========== Perceptual Loss ==========
        # Note: LPIPS is handled separately (not in uncertainty weighting)
        # as it conflicts with PSNR/SSIM optimization
        lpips_loss = torch.tensor(0.0, device=device)
        if self.use_lpips and self.lpips_loss is not None:
            lpips_loss = self.w_lpips * self.lpips_loss(pred, target)

        # ========== Reconstruction Loss ==========
        recon_loss = torch.tensor(0.0, device=device)
        if len(gaussians_list) > 0:
            # Import renderer (avoid circular import)
            from models.renderer import GaussianRenderer
            renderer = GaussianRenderer(image_size=(H, W))

            for i, gaussians in enumerate(gaussians_list):
                rendered_i = renderer(gaussians)['render']
                target_i = input_frames[:, i]
                recon_loss = recon_loss + self.l1_loss(rendered_i, target_i)

            recon_loss = recon_loss / len(gaussians_list)

        raw_losses['recon'] = recon_loss

        # ========== Geometric Losses ==========
        raw_losses['depth'] = self.depth_loss(depth, pred)
        raw_losses['temporal'] = self.temporal_loss(
            gaussians_list, gaussians_interp, t
        ) if len(gaussians_list) > 0 else torch.tensor(0.0, device=device)

        # ========== Gaussian Flow Loss ==========
        # Note: gflow has its own decay schedule, compute raw loss here
        gflow_loss = torch.tensor(0.0, device=device)
        if self.use_gflow and self.gflow_loss is not None and len(gaussians_list) >= 2:
            gflow_loss = self.gflow_loss(
                gaussians_list[0],
                gaussians_list[-1],
                input_frames[:, 0],
                input_frames[:, -1],
            )
        raw_losses['gflow'] = gflow_loss

        # ========== Regularization ==========
        if len(gaussians_list) > 0:
            all_opacities = torch.cat([g['opacity'] for g in gaussians_list], dim=1)
            raw_losses['opacity_reg'] = self.opacity_reg(all_opacities)

            all_scales = torch.cat([g['scale'] for g in gaussians_list], dim=1)
            raw_losses['scale_reg'] = self.scale_reg(all_scales)
        else:
            raw_losses['opacity_reg'] = torch.tensor(0.0, device=device)
            raw_losses['scale_reg'] = torch.tensor(0.0, device=device)

        # ========== Apply Weighting ==========
        if self.use_uncertainty_weighting and self.uncertainty is not None:
            # Apply gflow decay before uncertainty weighting
            gflow_decay = get_gflow_weight(
                self.current_epoch,
                self.total_epochs,
                1.0,  # Max multiplier
                self.gflow_decay_fraction
            )
            raw_losses['gflow'] = raw_losses['gflow'] * gflow_decay

            # Use uncertainty-based learned weights
            total_loss, weighted_losses = self.uncertainty(raw_losses)

            # Add LPIPS separately (not in uncertainty weighting)
            weighted_losses['lpips'] = lpips_loss
            total_loss = total_loss + lpips_loss
        else:
            # Use fixed weights
            weighted_losses = {}
            weighted_losses['l1'] = self.w_photo * raw_losses['l1']
            weighted_losses['ssim'] = self.w_ssim * raw_losses['ssim']
            weighted_losses['lap'] = self.w_lap * raw_losses['lap']
            weighted_losses['lpips'] = lpips_loss
            weighted_losses['recon'] = self.w_recon * raw_losses['recon']
            weighted_losses['depth'] = self.w_depth * raw_losses['depth']
            weighted_losses['temporal'] = self.w_temporal * raw_losses['temporal']

            # Apply gflow decay
            gflow_weight = get_gflow_weight(
                self.current_epoch,
                self.total_epochs,
                self.w_gflow_max,
                self.gflow_decay_fraction
            )
            weighted_losses['gflow'] = gflow_weight * raw_losses['gflow']

            weighted_losses['opacity_reg'] = self.w_opacity_reg * raw_losses['opacity_reg']
            weighted_losses['scale_reg'] = self.w_scale_reg * raw_losses['scale_reg']

            total_loss = sum(weighted_losses.values())

        # ========== Return ==========
        weighted_losses['total'] = total_loss

        if return_components:
            return weighted_losses
        else:
            return {'total': total_loss}

    def get_uncertainty_stats(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Get uncertainty weighting statistics for logging.

        Returns:
            Dict with 'weights', 'sigmas', 'log_vars' if uncertainty enabled, else None
        """
        if self.uncertainty is not None:
            return self.uncertainty.get_stats()
        return None

    def is_uncertainty_warmup(self) -> bool:
        """Check if still in uncertainty warmup phase."""
        if self.uncertainty is not None:
            return self.uncertainty.is_warmup()
        return False


def build_loss(loss_config, model_config=None) -> GSMambaLoss:
    """
    Build loss function from config.

    Args:
        loss_config: LossConfig with loss weights and uncertainty settings
        model_config: Optional GSMambaConfig for image_size

    Returns:
        GSMambaLoss instance
    """
    image_size = (256, 256)
    if model_config is not None:
        image_size = model_config.image_size

    return GSMambaLoss(
        w_photo=loss_config.w_photo,
        w_ssim=loss_config.w_ssim,
        w_lap=loss_config.w_lap,
        w_lpips=loss_config.w_lpips,
        w_recon=loss_config.w_recon,
        w_depth=loss_config.w_depth,
        w_temporal=loss_config.w_temporal,
        w_gflow_max=loss_config.w_gflow_max,
        gflow_decay_fraction=loss_config.gflow_decay_fraction,
        w_opacity_reg=loss_config.w_opacity_reg,
        w_scale_reg=loss_config.w_scale_reg,
        use_lpips=loss_config.use_lpips,
        use_gflow=loss_config.use_gflow,
        use_uncertainty_weighting=loss_config.use_uncertainty_weighting,
        uncertainty_warmup_epochs=loss_config.uncertainty_warmup_epochs,
        initial_log_vars=loss_config.initial_log_vars,
        image_size=image_size,
    )
