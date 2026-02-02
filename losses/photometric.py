"""
Photometric Loss Functions

L1, SSIM, and Laplacian pyramid losses for image reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math


class L1Loss(nn.Module):
    """Simple L1 loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) loss.

    SSIM measures structural similarity between images, considering
    luminance, contrast, and structure.

    Args:
        window_size: Size of the Gaussian window (default: 11)
        sigma: Standard deviation of Gaussian (default: 1.5)
        channel: Number of channels (default: 3)
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5, channel: int = 3):
        super().__init__()

        self.window_size = window_size
        self.channel = channel

        # Create Gaussian window
        gauss = torch.Tensor([
            math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()

        # Create 2D window
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)

        self.register_buffer('window', _2D_window.expand(channel, 1, window_size, window_size))

    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute SSIM between two images."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        window = self.window.to(img1.device).type_as(img1)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 ** 2, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss (1 - SSIM).

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            SSIM loss (1 - SSIM)
        """
        return 1 - self._ssim(pred, target)


class LaplacianLoss(nn.Module):
    """
    Laplacian Pyramid Loss.

    Computes L1 loss at multiple scales of a Laplacian pyramid,
    providing multi-scale gradient supervision.

    Args:
        max_levels: Number of pyramid levels (default: 5)
        channels: Number of input channels (default: 3)
    """

    def __init__(self, max_levels: int = 5, channels: int = 3):
        super().__init__()

        self.max_levels = max_levels
        self.channels = channels

        # Gaussian kernel for building pyramid
        kernel = torch.tensor([
            [1., 4., 6., 4., 1.],
            [4., 16., 24., 16., 4.],
            [6., 24., 36., 24., 6.],
            [4., 16., 24., 16., 4.],
            [1., 4., 6., 4., 1.]
        ]) / 256.

        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        self.register_buffer('kernel', kernel)

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample image by 2x using Gaussian blur."""
        x = F.pad(x, (2, 2, 2, 2), mode='reflect')
        x = F.conv2d(x, self.kernel.to(x.device), groups=self.channels)
        return x[:, :, ::2, ::2]

    def _upsample(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample image by 2x."""
        B, C, H, W = x.shape
        out = torch.zeros(B, C, H * 2, W * 2, device=x.device, dtype=x.dtype)
        out[:, :, ::2, ::2] = x
        out = F.pad(out, (2, 2, 2, 2), mode='reflect')
        return F.conv2d(out, self.kernel.to(x.device) * 4, groups=self.channels)

    def _laplacian_pyramid(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Build Laplacian pyramid."""
        pyramid = []
        current = x

        for _ in range(self.max_levels - 1):
            down = self._downsample(current)
            up = self._upsample(down)

            # Handle size mismatch
            if up.shape != current.shape:
                up = F.interpolate(up, size=current.shape[2:], mode='bilinear', align_corners=False)

            diff = current - up
            pyramid.append(diff)
            current = down

        # Add the final low-frequency residual
        pyramid.append(current)

        return pyramid

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Laplacian pyramid loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            Sum of L1 losses at each pyramid level
        """
        pred_pyramid = self._laplacian_pyramid(pred)
        target_pyramid = self._laplacian_pyramid(target)

        loss = 0.0
        for pred_level, target_level in zip(pred_pyramid, target_pyramid):
            loss = loss + F.l1_loss(pred_level, target_level)

        return loss


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (smooth L1).

    A differentiable approximation to L1 loss that is smooth near zero.

    Args:
        eps: Small constant for numerical stability (default: 1e-6)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff ** 2 + self.eps ** 2))


class CombinedPhotometricLoss(nn.Module):
    """
    Combined photometric loss.

    Combines L1/Charbonnier loss with SSIM and Laplacian losses.

    Args:
        w_l1: Weight for L1 loss (default: 1.0)
        w_ssim: Weight for SSIM loss (default: 0.2)
        w_lap: Weight for Laplacian loss (default: 0.5)
        use_charbonnier: Use Charbonnier instead of L1 (default: False)
    """

    def __init__(
            self,
            w_l1: float = 1.0,
            w_ssim: float = 0.2,
            w_lap: float = 0.5,
            use_charbonnier: bool = False,
    ):
        super().__init__()

        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_lap = w_lap

        self.l1_loss = CharbonnierLoss() if use_charbonnier else L1Loss()
        self.ssim_loss = SSIMLoss()
        self.lap_loss = LaplacianLoss()

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute combined photometric loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)
            return_components: Whether to return individual loss components

        Returns:
            Combined loss or (combined_loss, loss_dict) if return_components
        """
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        lap = self.lap_loss(pred, target)

        total = self.w_l1 * l1 + self.w_ssim * ssim + self.w_lap * lap

        if return_components:
            return total, {
                'l1': l1,
                'ssim': ssim,
                'lap': lap,
            }
        return total
