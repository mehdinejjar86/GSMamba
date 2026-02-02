#!/usr/bin/env python3
"""
GS-Mamba Evaluation Script

Evaluates GS-Mamba on multiple benchmarks:
1. Vimeo-90K Triplet test set (standard 2x interpolation)
2. X4K1000FPS test set (cascaded 8x interpolation)

X4K Cascaded Evaluation:
- Uses frames 0 and 32 as initial anchors
- Hierarchical prediction with increasing N:
  * Level 0: N=2 (0, 32) → predict frame 16
  * Level 1: N=3 (0, 16, 32) → predict frames 8, 24
  * Level 2: N=4 (closest anchors) → predict frames 4, 12, 20, 28

Usage:
    # Evaluate on Vimeo (from project root)
    python -m gsmamba.eval --checkpoint best.pth --dataset vimeo --vimeo_root ./datasets/vimeo_triplet

    # Evaluate on X4K (cascaded 8x)
    python -m gsmamba.eval --checkpoint best.pth --dataset x4k --x4k_root ./datasets/x4k/test

    # Evaluate on both
    python -m gsmamba.eval --checkpoint best.pth --dataset all

    # Or use the convenience script
    python run_eval.py --checkpoint best.pth --dataset all
"""

import os
import sys
import cv2
import math
import glob
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from torchvision.utils import save_image

torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from .models.gs_mamba import GSMamba, build_model
from .config import get_config, get_full_config


# ==============================================================================
# Metrics
# ==============================================================================

def ssim_matlab(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Compute SSIM (MATLAB-compatible implementation).

    Args:
        img1, img2: Tensors of shape [1, 1, H, W] (single channel)

    Returns:
        SSIM value
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    img1 = img1.float()
    img2 = img2.float()

    # Gaussian window 11x11, sigma=1.5
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).float().unsqueeze(0).unsqueeze(0).to(img1.device)

    mu1 = torch.nn.functional.conv2d(img1, window, padding=5, groups=1)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=5, groups=1)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(img1 ** 2, window, padding=5, groups=1) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 ** 2, window, padding=5, groups=1) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=5, groups=1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute PSNR between prediction and ground truth."""
    mse = torch.mean((pred - gt) ** 2)
    if mse < 1e-10:
        return 100.0
    return -10 * math.log10(mse.cpu().item())


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute SSIM for RGB image (average across channels)."""
    ssim_vals = []
    for c in range(3):
        ssim_vals.append(ssim_matlab(pred[:, c:c+1], gt[:, c:c+1]))
    return np.mean(ssim_vals)


def compute_lpips(pred: torch.Tensor, gt: torch.Tensor, lpips_fn) -> float:
    """Compute LPIPS between prediction and ground truth."""
    if lpips_fn is None:
        return 0.0
    # LPIPS expects [-1, 1] range
    pred_lpips = pred * 2 - 1
    gt_lpips = gt * 2 - 1
    return lpips_fn(pred_lpips, gt_lpips).item()


# ==============================================================================
# Utilities
# ==============================================================================

class InputPadder:
    """Pads images to be divisible by divisor."""

    def __init__(self, dims, divisor=32):
        self.ht, self.wd = dims[-2:]
        pad_ht = (divisor - (self.ht % divisor)) % divisor
        pad_wd = (divisor - (self.wd % divisor)) % divisor
        self._pad = [0, pad_wd, 0, pad_ht]

    def pad(self, *inputs):
        return [torch.nn.functional.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        return x[..., :self.ht, :self.wd]


def load_frame(path: str) -> np.ndarray:
    """Load frame as numpy array [H, W, 3] BGR in [0, 1]."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load: {path}")
    return img.astype(np.float32) / 255.0


def numpy_to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy [H, W, 3] BGR to tensor [1, 3, H, W] RGB."""
    # BGR to RGB
    img_rgb = img[..., ::-1].copy()
    return torch.from_numpy(img_rgb.transpose(2, 0, 1)[None]).float().to(device)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor [1, 3, H, W] RGB to numpy [H, W, 3] BGR uint8."""
    img = tensor[0].cpu().numpy().transpose(1, 2, 0)
    # RGB to BGR
    img_bgr = img[..., ::-1]
    return (img_bgr * 255).clip(0, 255).round().astype(np.uint8)


# ==============================================================================
# GS-Mamba Predictor for Cascaded Evaluation
# ==============================================================================

class GSMambaPredictor:
    """
    GS-Mamba model wrapper for cascaded X4K evaluation.

    Implements hierarchical cascaded prediction:
    - Level 0: N=2 (endpoints) → predict middle
    - Level 1: N=3 (endpoints + middle) → predict quarters
    - Level 2: N=4 (closest 4 anchors) → predict eighths
    """

    def __init__(self, model: GSMamba, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    def predict_frame(
        self,
        anchor_frames: Dict[int, torch.Tensor],
        anchor_indices: List[int],
        target_idx: int,
        total_frames: int = 33
    ) -> torch.Tensor:
        """
        Predict a single frame using closest anchors.

        Args:
            anchor_frames: dict {frame_idx: tensor [1, 3, H, W]}
            anchor_indices: list of available anchor indices
            target_idx: target frame index to predict
            total_frames: total frame count (33 for X4K test)

        Returns:
            Predicted frame tensor [1, 3, H, W]
        """
        sorted_anchors = sorted(anchor_indices)

        # Find anchors that bracket the target
        before = [i for i in sorted_anchors if i < target_idx]
        after = [i for i in sorted_anchors if i > target_idx]

        if not before or not after:
            raise ValueError(f"Target {target_idx} not bracketed by anchors {sorted_anchors}")

        # Select up to 4 anchors (closest to target)
        selected = []
        selected.append(before[-1])  # Closest before
        selected.append(after[0])    # Closest after

        remaining_before = before[:-1][::-1]
        remaining_after = after[1:]

        while len(selected) < 4 and (remaining_before or remaining_after):
            if remaining_before:
                selected.append(remaining_before.pop(0))
            if len(selected) < 4 and remaining_after:
                selected.append(remaining_after.pop(0))

        selected = sorted(selected)
        N = len(selected)

        # Build input tensors
        frames_list = [anchor_frames[i] for i in selected]
        frames = torch.stack(frames_list, dim=1)  # [1, N, 3, H, W]

        # Compute normalized times
        first_idx, last_idx = selected[0], selected[-1]
        anchor_times = torch.tensor(
            [(i - first_idx) / (last_idx - first_idx) for i in selected],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # [1, N]

        target_time = (target_idx - first_idx) / (last_idx - first_idx)

        # Predict
        with torch.no_grad():
            output = self.model(
                frames=frames,
                t=target_time,
                timestamps=anchor_times,
                return_intermediates=False,
            )
            pred = output['pred']

        return pred

    def predict_8x(
        self,
        frame_0: torch.Tensor,
        frame_32: torch.Tensor,
        padder: Optional[InputPadder] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Predict 7 intermediate frames using cascaded approach.

        Args:
            frame_0: First frame tensor [1, 3, H, W]
            frame_32: Last frame tensor [1, 3, H, W]
            padder: Optional InputPadder for unpadding

        Returns:
            dict {target_idx: predicted_frame} for indices 4, 8, 12, 16, 20, 24, 28
        """
        anchor_frames = {0: frame_0, 32: frame_32}
        predictions = {}

        # Level 0: N=2 (0, 32) → predict 16
        pred_16 = self._predict_with_n(anchor_frames, [0, 32], target_idx=16)
        anchor_frames[16] = pred_16
        predictions[16] = padder.unpad(pred_16) if padder else pred_16

        # Level 1: N=3 (0, 16, 32) → predict 8, 24
        for target_idx in [8, 24]:
            pred = self._predict_with_n(anchor_frames, [0, 16, 32], target_idx=target_idx)
            anchor_frames[target_idx] = pred
            predictions[target_idx] = padder.unpad(pred) if padder else pred

        # Level 2: N=4 (closest anchors) → predict 4, 12, 20, 28
        for target_idx in [4, 12, 20, 28]:
            pred = self.predict_frame(
                anchor_frames,
                list(anchor_frames.keys()),
                target_idx
            )
            predictions[target_idx] = padder.unpad(pred) if padder else pred

        return predictions

    def _predict_with_n(
        self,
        anchor_frames: Dict[int, torch.Tensor],
        anchor_indices: List[int],
        target_idx: int
    ) -> torch.Tensor:
        """Predict using specific anchor indices."""
        frames_list = [anchor_frames[i] for i in anchor_indices]
        frames = torch.stack(frames_list, dim=1)  # [1, N, 3, H, W]

        first_idx, last_idx = anchor_indices[0], anchor_indices[-1]
        anchor_times = torch.tensor(
            [(i - first_idx) / (last_idx - first_idx) for i in anchor_indices],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        target_time = (target_idx - first_idx) / (last_idx - first_idx)

        with torch.no_grad():
            output = self.model(
                frames=frames,
                t=target_time,
                timestamps=anchor_times,
                return_intermediates=False,
            )

        return output['pred']


# ==============================================================================
# Vimeo-90K Evaluation
# ==============================================================================

def get_vimeo_test_samples(root: str) -> List[str]:
    """Get Vimeo triplet test samples."""
    list_file = os.path.join(root, "tri_testlist.txt")
    if not os.path.isfile(list_file):
        raise FileNotFoundError(f"Missing list file: {list_file}")

    with open(list_file, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return [ln for ln in lines if not ln.startswith("#")]


def evaluate_vimeo(
    model: GSMamba,
    data_path: str,
    device: torch.device,
    use_lpips: bool = True,
    samples_dir: Optional[Path] = None,
    save_samples: bool = False,
    epoch: int = 0,
) -> Dict[str, float]:
    """
    Evaluate GS-Mamba on Vimeo-90K triplet test set.

    Args:
        model: GS-Mamba model
        data_path: Path to Vimeo triplet data
        device: torch device
        use_lpips: Whether to compute LPIPS
        samples_dir: Directory to save sample images (optional)
        save_samples: Whether to save sample images
        epoch: Current epoch for naming saved samples

    Returns:
        dict with PSNR, SSIM, LPIPS metrics
    """
    samples = get_vimeo_test_samples(data_path)
    print(f"Vimeo test: {len(samples)} samples")

    predictor = GSMambaPredictor(model, device)

    # LPIPS
    lpips_fn = None
    if use_lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net='alex').to(device)
            lpips_fn.eval()
        except ImportError:
            print("Warning: lpips not installed, skipping LPIPS metric")
            use_lpips = False

    all_psnr = []
    all_ssim = []
    all_lpips = []
    samples_saved = False

    for idx, sample in enumerate(tqdm(samples, desc="Vimeo")):
        base = os.path.join(data_path, "sequences", sample)

        # Load frames
        im1 = numpy_to_tensor(load_frame(os.path.join(base, "im1.png")), device)
        im2 = numpy_to_tensor(load_frame(os.path.join(base, "im2.png")), device)  # GT
        im3 = numpy_to_tensor(load_frame(os.path.join(base, "im3.png")), device)

        # Pad
        padder = InputPadder(im1.shape, divisor=32)
        im1_pad, im3_pad = padder.pad(im1, im3)

        # Predict middle frame (t=0.5)
        frames = torch.stack([im1_pad, im3_pad], dim=1)  # [1, 2, 3, H, W]
        anchor_times = torch.tensor([[0.0, 1.0]], device=device)

        with torch.no_grad():
            output = model(
                frames=frames,
                t=0.5,
                timestamps=anchor_times,
                return_intermediates=False,
            )
            pred = padder.unpad(output['pred'])

        # Clamp and quantize prediction (for fair comparison - matches saved image quality)
        pred = pred.clamp(0, 1)
        pred_np = tensor_to_numpy(pred)  # Quantize to uint8
        pred = numpy_to_tensor(pred_np / 255.0, device)  # Back to float [0,1]

        # Compute metrics on quantized prediction
        psnr = compute_psnr(pred, im2)
        ssim = compute_ssim(pred, im2)
        all_psnr.append(psnr)
        all_ssim.append(ssim)

        if use_lpips and lpips_fn is not None:
            lpips_val = compute_lpips(pred, im2, lpips_fn)
            all_lpips.append(lpips_val)

        # Save sample images (first sample only, like SPACE)
        if save_samples and not samples_saved and samples_dir is not None:
            samples_dir.mkdir(parents=True, exist_ok=True)
            # Create grid: [input0, input1, prediction, target]
            grid = torch.cat([im1, im3, pred, im2], dim=0)
            save_image(grid, samples_dir / f"epoch_{epoch:04d}.png", nrow=4)
            samples_saved = True

    results = {
        'psnr': np.mean(all_psnr),
        'ssim': np.mean(all_ssim),
        'psnr_std': np.std(all_psnr),
        'ssim_std': np.std(all_ssim),
    }

    if use_lpips:
        results['lpips'] = np.mean(all_lpips)
        results['lpips_std'] = np.std(all_lpips)

    return results


# ==============================================================================
# X4K Evaluation
# ==============================================================================

def get_x4k_test_sequences(data_path: str) -> List[Tuple[str, str, List[str]]]:
    """
    Get X4K test sequences.

    Returns:
        List of (type_name, seq_name, frame_paths) tuples
    """
    sequences = []

    for type_dir in sorted(glob.glob(os.path.join(data_path, 'Type*'))):
        type_name = os.path.basename(type_dir)
        for seq_dir in sorted(glob.glob(os.path.join(type_dir, '*'))):
            if not os.path.isdir(seq_dir):
                continue
            seq_name = os.path.basename(seq_dir)
            frames = sorted(glob.glob(os.path.join(seq_dir, '*.png')))
            if len(frames) == 33:  # 0-32 frames
                sequences.append((type_name, seq_name, frames))

    return sequences


def evaluate_x4k(
    model: GSMamba,
    data_path: str,
    device: torch.device,
    modes: List[str] = ['XTEST-2k', 'XTEST-4k'],
    use_lpips: bool = False,
    samples_dir: Optional[Path] = None,
    save_samples: bool = False,
    epoch: int = 0,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate GS-Mamba on X4K1000FPS test set using cascaded prediction.

    Args:
        model: GS-Mamba model
        data_path: Path to X4K test data (containing Type1/, Type2/, Type3/)
        device: torch device
        modes: Evaluation modes ('XTEST-2k', 'XTEST-4k')
        use_lpips: Whether to compute LPIPS
        samples_dir: Directory to save sample images (optional)
        save_samples: Whether to save sample images
        epoch: Current epoch for naming saved samples

    Returns:
        dict with results per mode
    """
    sequences = get_x4k_test_sequences(data_path)
    print(f"X4K test: {len(sequences)} sequences")

    predictor = GSMambaPredictor(model, device)
    results = {}

    # LPIPS
    lpips_fn = None
    if use_lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net='alex').to(device)
            lpips_fn.eval()
        except ImportError:
            print("Warning: lpips not installed, skipping LPIPS metric")
            use_lpips = False

    # Target frame indices for 8x interpolation
    target_indices = [4, 8, 12, 16, 20, 24, 28]

    for mode in modes:
        print(f"\nEvaluating {mode}...")

        all_psnr = []
        all_ssim = []
        all_lpips = []
        samples_saved = False

        for type_name, seq_name, frames in tqdm(sequences, desc=mode):
            # Load anchor frames (0 and 32)
            frame_0_np = load_frame(frames[0])
            frame_32_np = load_frame(frames[32])

            # Resize for 2K mode
            if mode == 'XTEST-2k':
                frame_0_np = cv2.resize(frame_0_np, (2048, 1080), interpolation=cv2.INTER_AREA)
                frame_32_np = cv2.resize(frame_32_np, (2048, 1080), interpolation=cv2.INTER_AREA)

            # Convert to tensors
            frame_0 = numpy_to_tensor(frame_0_np, device)
            frame_32 = numpy_to_tensor(frame_32_np, device)

            # Pad for model
            padder = InputPadder(frame_0.shape, divisor=32)
            frame_0_pad, frame_32_pad = padder.pad(frame_0, frame_32)

            # Predict all intermediate frames (cascaded)
            predictions = predictor.predict_8x(frame_0_pad, frame_32_pad, padder)

            # Evaluate each target frame
            seq_psnr = []
            seq_ssim = []
            seq_lpips = []

            for target_idx in target_indices:
                # Load ground truth
                gt_np = load_frame(frames[target_idx])
                if mode == 'XTEST-2k':
                    gt_np = cv2.resize(gt_np, (2048, 1080), interpolation=cv2.INTER_AREA)
                gt = numpy_to_tensor(gt_np, device)

                # Get prediction (convert through numpy for fair comparison)
                pred_np = tensor_to_numpy(predictions[target_idx])
                pred = numpy_to_tensor(pred_np / 255.0, device)

                # Compute metrics
                psnr = compute_psnr(pred, gt)
                ssim = compute_ssim(pred, gt)

                seq_psnr.append(psnr)
                seq_ssim.append(ssim)

                if use_lpips and lpips_fn is not None:
                    lpips_val = compute_lpips(pred, gt, lpips_fn)
                    seq_lpips.append(lpips_val)

            all_psnr.append(np.mean(seq_psnr))
            all_ssim.append(np.mean(seq_ssim))
            if use_lpips:
                all_lpips.append(np.mean(seq_lpips))

            # Save sample images (first sequence only, like SPACE)
            if save_samples and not samples_saved and samples_dir is not None:
                out_dir = samples_dir / f"x4k_{mode.lower()}"
                out_dir.mkdir(parents=True, exist_ok=True)
                # Save grid: [frame0, frame32, pred_16, gt_16]
                pred_16 = predictions[16].clamp(0, 1)
                gt_16_np = load_frame(frames[16])
                if mode == 'XTEST-2k':
                    gt_16_np = cv2.resize(gt_16_np, (2048, 1080), interpolation=cv2.INTER_AREA)
                gt_16 = numpy_to_tensor(gt_16_np, device)
                grid = torch.cat([frame_0, frame_32, pred_16, gt_16], dim=0)
                save_image(grid, out_dir / f"epoch_{epoch:04d}.png", nrow=4)
                samples_saved = True

        results[mode] = {
            'psnr': np.mean(all_psnr),
            'ssim': np.mean(all_ssim),
            'psnr_std': np.std(all_psnr),
            'ssim_std': np.std(all_ssim),
        }
        if use_lpips:
            results[mode]['lpips'] = np.mean(all_lpips)
            results[mode]['lpips_std'] = np.std(all_lpips)

        print(f"{mode}  PSNR: {results[mode]['psnr']:.4f}  SSIM: {results[mode]['ssim']:.4f}")

    return results


# ==============================================================================
# Combined Evaluation (for training validation)
# ==============================================================================

def evaluate_all(
    model: GSMamba,
    vimeo_path: Optional[str],
    x4k_path: Optional[str],
    device: torch.device,
    x4k_mode: str = 'XTEST-2k',
    use_lpips: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate on all available datasets.

    This is the main entry point for validation during training.

    Args:
        model: GS-Mamba model
        vimeo_path: Path to Vimeo triplet (None to skip)
        x4k_path: Path to X4K test (None to skip)
        device: torch device
        x4k_mode: X4K evaluation mode ('XTEST-2k' or 'XTEST-4k')
        use_lpips: Whether to compute LPIPS

    Returns:
        dict with results per dataset
    """
    model.eval()
    results = {}

    if vimeo_path and os.path.exists(vimeo_path):
        print("\n" + "="*60)
        print("Evaluating on Vimeo-90K Triplet")
        print("="*60)
        results['vimeo'] = evaluate_vimeo(model, vimeo_path, device, use_lpips)

    if x4k_path and os.path.exists(x4k_path):
        print("\n" + "="*60)
        print(f"Evaluating on X4K ({x4k_mode})")
        print("="*60)
        x4k_results = evaluate_x4k(model, x4k_path, device, [x4k_mode], use_lpips)
        results['x4k'] = x4k_results.get(x4k_mode, {})

    return results


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate GS-Mamba')

    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='gsmamba',
                        choices=['gsmamba', 'gsmamba_small', 'gsmamba_large'])

    # Dataset
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['vimeo', 'x4k', 'all'])
    parser.add_argument('--vimeo_root', type=str, default='./datasets/vimeo_triplet')
    parser.add_argument('--x4k_root', type=str, default='./datasets/x4k/test')
    parser.add_argument('--x4k_mode', type=str, nargs='+',
                        default=['XTEST-2k'],
                        help='X4K evaluation modes')

    # Options
    parser.add_argument('--use_lpips', action='store_true',
                        help='Compute LPIPS metric')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = build_model(args.model)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"Model: {args.model}")

    # Evaluate
    print("\n" + "="*60)
    print("GS-Mamba Evaluation")
    print("="*60)

    results = {}

    if args.dataset in ['vimeo', 'all']:
        if os.path.exists(args.vimeo_root):
            print("\n" + "="*60)
            print("Vimeo-90K Triplet Test")
            print("="*60)
            results['vimeo'] = evaluate_vimeo(
                model, args.vimeo_root, device, args.use_lpips
            )
        else:
            print(f"Warning: Vimeo path not found: {args.vimeo_root}")

    if args.dataset in ['x4k', 'all']:
        if os.path.exists(args.x4k_root):
            print("\n" + "="*60)
            print("X4K1000FPS Test (Cascaded 8x)")
            print("="*60)
            x4k_results = evaluate_x4k(
                model, args.x4k_root, device, args.x4k_mode, args.use_lpips
            )
            results['x4k'] = x4k_results
        else:
            print(f"Warning: X4K path not found: {args.x4k_root}")

    # Print summary
    print("\n" + "="*60)
    print("Final Results")
    print("="*60)

    if 'vimeo' in results:
        r = results['vimeo']
        print(f"\nVimeo-90K Triplet:")
        print(f"  PSNR: {r['psnr']:.4f} ± {r['psnr_std']:.4f}")
        print(f"  SSIM: {r['ssim']:.4f} ± {r['ssim_std']:.4f}")
        if 'lpips' in r:
            print(f"  LPIPS: {r['lpips']:.4f} ± {r['lpips_std']:.4f}")

    if 'x4k' in results:
        for mode, r in results['x4k'].items() if isinstance(results['x4k'], dict) and 'psnr' not in results['x4k'] else [('default', results['x4k'])]:
            print(f"\nX4K ({mode}):")
            print(f"  PSNR: {r['psnr']:.4f} ± {r['psnr_std']:.4f}")
            print(f"  SSIM: {r['ssim']:.4f} ± {r['ssim_std']:.4f}")
            if 'lpips' in r:
                print(f"  LPIPS: {r['lpips']:.4f} ± {r['lpips_std']:.4f}")


if __name__ == '__main__':
    main()
