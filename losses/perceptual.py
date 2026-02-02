"""
Perceptual Loss

LPIPS (Learned Perceptual Image Patch Similarity) loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Try to import LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


class LPIPSLoss(nn.Module):
    """
    LPIPS perceptual loss.

    Uses pretrained VGG or AlexNet features to compute perceptual similarity.

    Args:
        net: Network to use ('vgg', 'alex', 'squeeze')
        spatial: Return spatial map or scalar (default: False)
    """

    def __init__(self, net: str = 'vgg', spatial: bool = False):
        super().__init__()

        if not LPIPS_AVAILABLE:
            print("Warning: lpips not installed. LPIPSLoss will use VGG fallback.")
            self.lpips = None
            self.vgg = VGGPerceptualLoss()
        else:
            self.lpips = lpips.LPIPS(net=net, spatial=spatial)
            self.lpips.eval()
            # Freeze LPIPS weights
            for param in self.lpips.parameters():
                param.requires_grad = False
            self.vgg = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute LPIPS loss.

        Args:
            pred: Predicted image (B, 3, H, W) in [0, 1]
            target: Target image (B, 3, H, W) in [0, 1]

        Returns:
            LPIPS loss value
        """
        if self.lpips is not None:
            # LPIPS expects images in [-1, 1]
            pred_scaled = pred * 2 - 1
            target_scaled = target * 2 - 1
            return self.lpips(pred_scaled, target_scaled).mean()
        else:
            return self.vgg(pred, target)


class VGGPerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss (fallback when LPIPS not available).

    Uses features from VGG16 pretrained on ImageNet.

    Args:
        layers: Which VGG layers to use
        weights: Weights for each layer
    """

    def __init__(
            self,
            layers: Optional[list] = None,
            weights: Optional[list] = None,
    ):
        super().__init__()

        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        except:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True)

        self.layers = layers or [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3
        self.weights = weights or [1.0, 1.0, 1.0, 1.0]

        # Extract features
        features = list(vgg.features.children())
        self.slices = nn.ModuleList()

        prev_layer = 0
        for layer in self.layers:
            self.slices.append(nn.Sequential(*features[prev_layer:layer + 1]))
            prev_layer = layer + 1

        # Freeze weights
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize image for VGG."""
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute VGG perceptual loss.

        Args:
            pred: Predicted image (B, 3, H, W) in [0, 1]
            target: Target image (B, 3, H, W) in [0, 1]

        Returns:
            Perceptual loss
        """
        pred = self._normalize(pred)
        target = self._normalize(target)

        loss = 0.0
        x = pred
        y = target

        for slice_module, weight in zip(self.slices, self.weights):
            x = slice_module(x)
            with torch.no_grad():
                y = slice_module(y)
            loss = loss + weight * F.l1_loss(x, y)

        return loss
