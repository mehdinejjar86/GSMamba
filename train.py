#!/usr/bin/env python3
"""
GS-Mamba Training Script

Train GS-Mamba for N-frame video interpolation via 3D Gaussian Splatting + Mamba.

Usage:
    # Single GPU (from project root)
    python -m gsmamba.train --exp_name my_experiment

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 -m gsmamba.train --exp_name my_experiment

    # With specific dataset
    python -m gsmamba.train --vimeo_root /path/to/vimeo --x4k_root /path/to/x4k

    # TEMPO-style X4K training with variable N-frames (like SPACE)
    # --x4k-steps 7 15 31 means anchor spacing of 8, 16, 32 frames
    # --x4k-n-frames 4 3 2 means use 4, 3, 2 anchor frames respectively
    python -m gsmamba.train --mode x4k_only \\
        --x4k_root /path/to/x4k \\
        --x4k-steps 7 15 31 \\
        --x4k-n-frames 4 3 2 \\
        --no_curriculum

    # Or use the convenience script
    python run_train.py --exp_name my_experiment
"""

import os
import sys
import argparse
import random
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from tqdm.auto import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None

from config import get_full_config, update_config_from_args, FullConfig
from models.gs_mamba import GSMamba, build_model
from losses.combined import GSMambaLoss, build_loss
from losses.photometric import SSIMLoss
from data import (
    create_train_loader,
    create_eval_loader,
    get_curriculum_settings,
)


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _silence_non_main_rank(rank: int):
    """
    Reduce log noise from non-main DDP workers.

    Keeps rank 0 as the single source of user-facing logs/progress.
    """
    if rank == 0:
        return

    # Silence Python-level prints from worker ranks.
    import builtins
    builtins.print = lambda *args, **kwargs: None

    # Keep tqdm disabled outside rank 0.
    os.environ.setdefault("TQDM_DISABLE", "1")
    # Reduce C++ distributed warning spam from worker ranks.
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")


def set_seed(seed: int, deterministic: bool = False):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_optimizer(model: nn.Module, config: FullConfig, criterion: nn.Module = None) -> torch.optim.Optimizer:
    """Create optimizer."""
    # Separate parameters for different learning rates
    encoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': encoder_params, 'lr': config.train.learning_rate * 0.1},  # Lower LR for encoder
        {'params': other_params, 'lr': config.train.learning_rate},
    ]

    # Include criterion parameters (e.g. uncertainty log_vars) in the optimizer so
    # they are stepped, zeroed, and clipped alongside model params.
    if criterion is not None:
        criterion_params = [p for p in criterion.parameters() if p.requires_grad]
        if criterion_params:
            param_groups.append({
                'params': criterion_params,
                'lr': config.train.learning_rate,
                'weight_decay': 0.0,  # No weight decay on log_vars
            })

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        betas=(0.9, 0.999),
    )

    return optimizer


def get_scheduler(optimizer, config: FullConfig):
    """
    Create learning rate scheduler.

    Uses epoch-based scheduling to remain stable when curriculum/mixed training
    changes dataloader length across epochs.
    """
    total_epochs = max(int(config.train.epochs), 1)
    warmup_epochs = max(0, min(int(config.train.warmup_epochs), total_epochs))

    if config.train.scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        if warmup_epochs > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=max(total_epochs - warmup_epochs, 1),
                eta_min=config.train.min_lr,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_epochs,
                eta_min=config.train.min_lr,
            )
    elif config.train.scheduler == "step":
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        scheduler = None

    return scheduler


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    config: FullConfig,
    writer: SummaryWriter,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    global_step: int = 0,
):
    """Train for one epoch."""
    model.train()
    criterion.set_epoch(epoch, config.train.epochs)

    total_loss = 0.0
    num_batches = 0
    start_time = time.time()
    total_samples = 0
    component_sums = {}
    last_lr = optimizer.param_groups[0]['lr']
    total_psnr = 0.0
    total_ssim = 0.0

    # Use criterion's SSIM implementation for consistent metric computation.
    ssim_metric = getattr(criterion, 'ssim_loss', None)
    if ssim_metric is None:
        ssim_metric = SSIMLoss().to(device)

    use_tqdm = (rank == 0)
    iterator = dataloader
    if use_tqdm:
        iterator = tqdm(
            dataloader,
            desc=f"Train {epoch}",
            total=len(dataloader),
            dynamic_ncols=True,
            leave=False,
        )

    for batch_idx, batch in enumerate(iterator):
        frames, anchor_times, target_time, target = batch

        # Move to runtime device
        non_blocking = (device.type == 'cuda')
        frames = frames.to(device, non_blocking=non_blocking)
        anchor_times = anchor_times.to(device, non_blocking=non_blocking)
        target_time = target_time.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)

        # Forward pass
        optimizer.zero_grad()

        use_amp = config.train.use_amp and device.type == 'cuda'
        with torch.cuda.amp.autocast(enabled=use_amp):
            # Model forward
            output = model(
                frames=frames,
                t=target_time,
                timestamps=anchor_times,
                return_intermediates=True,
            )

            # Compute loss
            # Pass precomputed flow from model (avoids a second flow_net forward pass)
            losses = criterion(
                pred=output['pred'],
                target=target,
                render=output['render'],
                depth=output['depth'],
                input_frames=frames,
                gaussians_list=output.get('all_gaussians', []),
                gaussians_interp=output.get('gaussians', {}),
                t=target_time,
                use_precomputed_flow=output.get('flow'),
            )

            loss = losses['total']

        # Handle non-finite loss (NaN/Inf from degenerate Gaussians or AMP overflow).
        # IMPORTANT: We must NOT skip backward() because DDP ALLREDUCE happens
        # during backward.  Skipping on one rank while others proceed would
        # permanently desync NCCL collective sequence numbers across ranks.
        # Instead we replace the loss with zero and let backward + scaler run
        # normally (GradScaler detects inf and skips the optimizer step).
        _skip_metrics = False
        if not torch.isfinite(loss):
            if rank == 0:
                print(f"\nWarning: non-finite loss ({loss.item()}) at epoch {epoch} batch {batch_idx}, zeroing loss for DDP sync.")
            _skip_metrics = True
            # Replace with a zero-valued loss that still has a grad path through the model
            # so DDP backward hooks (ALLREDUCE) fire on every rank.
            loss = (output['pred'] * 0).sum()

        if not _skip_metrics:
            with torch.no_grad():
                pred_metrics = output['pred'].detach().float().clamp(0, 1)
                target_metrics = target.detach().float().clamp(0, 1)
                mse = ((pred_metrics - target_metrics) ** 2).mean(dim=[1, 2, 3])
                batch_psnr = (-10 * torch.log10(mse + 1e-8)).mean().item()
                batch_ssim = (1.0 - ssim_metric(pred_metrics, target_metrics)).item()

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping (model + criterion to cover uncertainty log_vars)
        if config.train.grad_clip > 0:
            scaler.unscale_(optimizer)
            all_params = list(model.parameters()) + list(criterion.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, config.train.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if _skip_metrics:
            # Don't pollute running averages with the zeroed-out dummy loss.
            continue

        total_loss += loss.item()
        num_batches += 1
        total_samples += frames.shape[0]
        last_lr = optimizer.param_groups[0]['lr']
        total_psnr += batch_psnr
        total_ssim += batch_ssim

        # Keep detailed TensorBoard curves without per-step CLI printing.
        # Use persistent global_step so the x-axis is monotonic even when
        # the curriculum recreates the dataloader with a different length.
        step = global_step + batch_idx
        if writer is not None:
            writer.add_scalar('train/loss', loss.item(), step)
            writer.add_scalar('train/lr', last_lr, step)
            writer.add_scalar('train/psnr', batch_psnr, step)
            writer.add_scalar('train/ssim', batch_ssim, step)

        for key, value in losses.items():
            if key == 'total':
                continue
            component_sums[key] = component_sums.get(key, 0.0) + value.item()
            if writer is not None:
                writer.add_scalar(f'train/{key}', value.item(), step)

        if use_tqdm:
            iterator.set_postfix({
                'loss': f"{loss.item():.4f}",
                'psnr': f"{batch_psnr:.2f}",
                'ssim': f"{batch_ssim:.4f}",
                'lr': f"{last_lr:.2e}",
            })

    if use_tqdm:
        iterator.close()

    # Reduce metrics across all ranks so rank 0 logs global averages,
    # not just its own 1/world_size slice of the data.
    if world_size > 1 and dist.is_initialized():
        metrics = torch.tensor(
            [total_loss, total_psnr, total_ssim, float(num_batches)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss = metrics[0].item()
        total_psnr = metrics[1].item()
        total_ssim = metrics[2].item()
        num_batches = int(metrics[3].item())

    # Log uncertainty weights at end of epoch
    if rank == 0:
        elapsed = max(time.time() - start_time, 1e-6)
        samples_per_sec = total_samples * world_size / elapsed

        avg_loss = total_loss / max(num_batches, 1)
        avg_psnr = total_psnr / max(num_batches, 1)
        avg_ssim = total_ssim / max(num_batches, 1)
        print(
            f"Epoch {epoch} summary: "
            f"Train Loss={avg_loss:.4f} "
            f"PSNR={avg_psnr:.2f}dB "
            f"SSIM={avg_ssim:.4f} "
            f"LR={last_lr:.2e} "
            f"Speed={samples_per_sec:.1f} samples/s"
        )

        if writer is not None:
            writer.add_scalar('train/loss_epoch', avg_loss, epoch)
            writer.add_scalar('train/psnr_epoch', avg_psnr, epoch)
            writer.add_scalar('train/ssim_epoch', avg_ssim, epoch)
            writer.add_scalar('train/lr_epoch', last_lr, epoch)
            for key, value_sum in component_sums.items():
                writer.add_scalar(f'train/{key}_epoch', value_sum / max(num_batches, 1), epoch)

        uncertainty_stats = criterion.get_uncertainty_stats()
        if uncertainty_stats is not None:
            if writer is not None:
                for name, weight in uncertainty_stats['weights'].items():
                    writer.add_scalar(f'uncertainty/weight_{name}', weight, epoch)
                for name, sigma in uncertainty_stats['sigmas'].items():
                    writer.add_scalar(f'uncertainty/sigma_{name}', sigma, epoch)

            # Print summary
            is_warmup = criterion.is_uncertainty_warmup()
            weight_str = ', '.join([f"{k}={v:.3f}" for k, v in uncertainty_stats['weights'].items()])
            print(f"Epoch {epoch} uncertainty weights ({'warmup' if is_warmup else 'learned'}): {weight_str}")

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_step + num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader,
    epoch: int,
    config: FullConfig,
    writer: SummaryWriter,
    device: torch.device,
    rank: int = 0,
    samples_dir: Path = None,
    save_samples: bool = False,
):
    """Quick evaluation on dataloader (for frequent checks)."""
    model.eval()

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0
    total_samples = 0
    samples_saved = False

    ssim_metric = SSIMLoss().to(device)

    use_tqdm = (rank == 0)
    iterator = dataloader
    if use_tqdm:
        iterator = tqdm(
            dataloader,
            desc=f"Eval {epoch}",
            total=len(dataloader),
            dynamic_ncols=True,
            leave=False,
        )

    for batch in iterator:
        frames, anchor_times, target_time, target = batch

        non_blocking = (device.type == 'cuda')
        frames = frames.to(device, non_blocking=non_blocking)
        anchor_times = anchor_times.to(device, non_blocking=non_blocking)
        target_time = target_time.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)

        # Forward
        output = model(
            frames=frames,
            t=target_time,
            timestamps=anchor_times,
            return_intermediates=False,
        )

        # Compute PSNR
        pred = output['pred'].clamp(0, 1)
        mse = ((pred - target) ** 2).mean(dim=[1, 2, 3])
        psnr = -10 * torch.log10(mse + 1e-8)
        total_psnr += psnr.sum().item()
        ssim_val = 1.0 - ssim_metric(pred.float(), target.float())
        total_ssim += ssim_val.item() * frames.shape[0]
        batch_psnr = psnr.mean().item()
        batch_ssim = ssim_val.item()

        # Simple L1 loss for eval
        loss = nn.functional.l1_loss(pred, target)
        total_loss += loss.item()
        num_batches += 1
        total_samples += frames.shape[0]

        if use_tqdm:
            iterator.set_postfix({
                'loss': f"{loss.item():.4f}",
                'psnr': f"{batch_psnr:.2f}",
                'ssim': f"{batch_ssim:.4f}",
            })

        # Save sample images to disk (like SPACE)
        if save_samples and not samples_saved and rank == 0 and samples_dir is not None:
            out_dir = samples_dir / "vimeo"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Create grid: [input0, input1, prediction, target]
            inp0 = frames[0, 0:1].clamp(0, 1)
            inp1 = frames[0, -1:].clamp(0, 1)
            pred_sample = pred[0:1].clamp(0, 1)
            tgt_sample = target[0:1].clamp(0, 1)
            grid = torch.cat([inp0, inp1, pred_sample, tgt_sample], dim=0)
            save_image(grid, out_dir / f"epoch_{epoch:04d}.png", nrow=4)
            samples_saved = True

    if use_tqdm:
        iterator.close()

    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples

    if rank == 0:
        print(f"Eval Epoch {epoch}: Loss={avg_loss:.4f}, PSNR={avg_psnr:.2f}dB, SSIM={avg_ssim:.4f}")
        if writer is not None:
            writer.add_scalar('eval/loss', avg_loss, epoch)
            writer.add_scalar('eval/psnr', avg_psnr, epoch)
            writer.add_scalar('eval/ssim', avg_ssim, epoch)

        # Log sample images to TensorBoard
        if writer is not None and epoch % 10 == 0:
            writer.add_images('eval/prediction', pred[:4], epoch)
            writer.add_images('eval/target', target[:4], epoch)
            writer.add_images('eval/input_0', frames[:4, 0], epoch)
            writer.add_images('eval/input_1', frames[:4, -1], epoch)

    return avg_loss, avg_psnr


@torch.no_grad()
def evaluate_full(
    model: nn.Module,
    epoch: int,
    config: FullConfig,
    writer: SummaryWriter,
    vimeo_path: str = None,
    x4k_path: str = None,
    rank: int = 0,
    samples_dir: Path = None,
    save_samples: bool = False,
):
    """
    Full evaluation on Vimeo test and X4K cascaded (8x).

    This runs the complete benchmark evaluation including:
    - Vimeo-90K triplet test set
    - X4K1000FPS cascaded 8x interpolation (frames 0, 32 -> predict 4,8,12,16,20,24,28)
    """
    if rank != 0:
        return {}

    # Lazy import to avoid requiring eval-time deps (e.g., cv2) for train startup/help.
    from eval import evaluate_vimeo, evaluate_x4k

    # Get raw model (unwrap DDP if needed)
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.eval()  # Freeze BatchNorm running stats during evaluation
    device = next(raw_model.parameters()).device

    results = {}

    # Vimeo evaluation
    if vimeo_path and os.path.exists(vimeo_path):
        print(f"\n{'='*60}")
        print(f"Full Evaluation - Vimeo-90K Triplet (Epoch {epoch})")
        print(f"{'='*60}")
        try:
            vimeo_results = evaluate_vimeo(
                raw_model, vimeo_path, device, use_lpips=False,
                samples_dir=samples_dir / "vimeo_full" if samples_dir and save_samples else None,
                save_samples=save_samples,
                epoch=epoch,
            )
            results['vimeo'] = vimeo_results

            print(f"Vimeo PSNR: {vimeo_results['psnr']:.4f} dB")
            print(f"Vimeo SSIM: {vimeo_results['ssim']:.4f}")

            if writer is not None:
                writer.add_scalar('eval_full/vimeo_psnr', vimeo_results['psnr'], epoch)
                writer.add_scalar('eval_full/vimeo_ssim', vimeo_results['ssim'], epoch)
        except Exception as e:
            print(f"Vimeo evaluation failed: {e}")

    # X4K cascaded evaluation
    if x4k_path and os.path.exists(x4k_path):
        print(f"\n{'='*60}")
        print(f"Full Evaluation - X4K Cascaded 8x (Epoch {epoch})")
        print(f"{'='*60}")
        try:
            # Validate at configured resolution (default: full-res 4K).
            x4k_scale = getattr(getattr(config, 'data', config), 'x4k_scale', '4k').lower()
            eval_mode = 'XTEST-4k' if x4k_scale == '4k' else 'XTEST-2k'
            x4k_results = evaluate_x4k(
                raw_model, x4k_path, device, modes=[eval_mode], use_lpips=False,
                samples_dir=samples_dir / "x4k_full" if samples_dir and save_samples else None,
                save_samples=save_samples,
                epoch=epoch,
            )
            results['x4k'] = x4k_results.get(eval_mode, {})

            if eval_mode in x4k_results:
                r = x4k_results[eval_mode]
                label = "4K" if eval_mode == 'XTEST-4k' else "2K"
                print(f"X4K ({label}) PSNR: {r['psnr']:.4f} dB")
                print(f"X4K ({label}) SSIM: {r['ssim']:.4f}")

                if writer is not None:
                    writer.add_scalar('eval_full/x4k_psnr', r['psnr'], epoch)
                    writer.add_scalar('eval_full/x4k_ssim', r['ssim'], epoch)
        except Exception as e:
            print(f"X4K evaluation failed: {e}")

    # Restore train mode and free GPU memory reserved during eval
    # (especially important after X4K 4K OOM to avoid CUDA/NCCL corruption).
    raw_model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    config: FullConfig,
    output_dir: Path,
    is_best: bool = False,
):
    """Save checkpoint."""
    # Get model state (handle DDP)
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model': model_state,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'scaler': scaler.state_dict(),
        'config': config,
    }

    # Save latest
    torch.save(checkpoint, output_dir / 'latest.pth')

    # Save periodic
    if epoch % config.train.save_every == 0:
        torch.save(checkpoint, output_dir / f'epoch_{epoch:04d}.pth')

    # Save best
    if is_best:
        torch.save(checkpoint, output_dir / 'best.pth')


def main():
    parser = argparse.ArgumentParser(description='Train GS-Mamba')

    # Model
    parser.add_argument('--model', type=str, default='gsmamba',
                        choices=['gsmamba', 'gsmamba_small', 'gsmamba_large'])
    parser.add_argument('--image_size', type=int, nargs=2, default=None)

    # Data
    parser.add_argument('--vimeo_root', type=str, default=None,
                        help='Path to Vimeo triplet training data')
    parser.add_argument('--x4k_root', type=str, default=None,
                        help='Path to X4K training data')
    parser.add_argument('--x4k_test_root', type=str, default=None,
                        help='Path to X4K test data for cascaded evaluation')
    parser.add_argument('--x4k-steps', type=int, nargs='+', default=None,
                        help='X4K step values for TEMPO-style training. '
                             'E.g., --x4k-steps 7 15 31 means anchor spacing of 8, 16, 32 frames. '
                             'Must be paired with --x4k-n-frames.')
    parser.add_argument('--x4k-n-frames', type=int, nargs='+', default=None,
                        help='X4K n_frames per step for TEMPO-style training. '
                             'E.g., --x4k-n-frames 4 3 2 means use 4, 3, 2 anchor frames respectively. '
                             'Must be paired with --x4k-steps.')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--crop_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--full-coverage-mixed', dest='full_coverage_mixed',
                        action='store_true', default=None,
                        help='Use all available mixed-dataset batches (disable ratio truncation).')
    parser.add_argument('--no-full-coverage-mixed', dest='full_coverage_mixed',
                        action='store_false',
                        help='Use ratio-constrained mixed sampling (may truncate larger datasets).')
    parser.add_argument('--drop-last-mixed', dest='drop_last_mixed',
                        action='store_true', default=None,
                        help='Drop incomplete mixed-mode batches.')
    parser.add_argument('--keep-tail-mixed', dest='drop_last_mixed',
                        action='store_false',
                        help='Keep incomplete mixed-mode batches.')
    parser.add_argument('--mode', type=str, default='vimeo_only',
                        choices=['vimeo_only', 'x4k_only', 'mixed'])
    parser.add_argument('--x4k_fraction', type=float, default=None,
                        help='Fraction of X4K samples to draw per epoch (0.0-1.0). '
                             'E.g., 0.1 draws a random 10%% subset each epoch, '
                             'which is different per epoch. Speeds up x4k_only / mixed training '
                             'when X4K generates millions of samples. Default: 1.0 (all samples).')

    # Evaluation
    parser.add_argument('--eval_full_every', type=int, default=10,
                        help='Run full benchmark evaluation every N epochs')
    parser.add_argument('--skip_x4k_eval', action='store_true',
                        help='Skip X4K evaluation during training')
    parser.add_argument('--x4k_scale', type=str, default=None,
                        choices=['2k', '4k'],
                        help='X4K evaluation resolution: 2k (1920x1080) or 4k (3840x2160). '
                             '2k uses ~4x less GPU memory. Default: from config (4k).')
    parser.add_argument('--save_samples_every', type=int, default=10,
                        help='Save validation sample images every N epochs (0 to disable)')

    # Training
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--use_amp', action='store_true', default=None)
    parser.add_argument('--no_amp', dest='use_amp', action='store_false')
    parser.add_argument('--use_curriculum', action='store_true', default=True)
    parser.add_argument('--no_curriculum', dest='use_curriculum', action='store_false')

    # Flow supervision
    parser.add_argument('--flow_ckpt', type=str, default=None,
                        help='Path to VFIMamba checkpoint for Gaussian Flow Loss')
    parser.add_argument('--flow_model_size', type=str, default='S', choices=['S', 'L'],
                        help='VFIMamba model size: S (F=16) or L (F=32)')

    # Experiment
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda',
                        help='Runtime device (e.g., cuda, cuda:0, cpu)')

    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)
    _silence_non_main_rank(rank)

    # Select runtime device
    requested_device = args.device
    if requested_device.startswith('cuda') and not torch.cuda.is_available():
        if is_main:
            print(f"Warning: requested device '{requested_device}' but CUDA is unavailable, falling back to CPU.")
        requested_device = 'cpu'
    if requested_device == 'cuda' and torch.cuda.is_available() and world_size > 1:
        requested_device = f'cuda:{local_rank}'
    device = torch.device(requested_device)

    # Load config
    config = get_full_config(args.model)
    config = update_config_from_args(config, args)

    # Set experiment name
    if config.exp_name == "gsmamba_default":
        config.exp_name = f"gsmamba_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create output directory
    output_dir = Path(config.output_dir) / config.exp_name
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

    # Set seed
    set_seed(config.seed + rank, config.deterministic)

    # Load VFIMamba flow network before model construction so it can be embedded in the model
    flow_net = None
    if args.flow_ckpt:
        from losses.flow_net_loader import load_vfimamba_flow_net
        flow_net = load_vfimamba_flow_net(
            ckpt_path=args.flow_ckpt,
            model_size=args.flow_model_size,
            device=str(device),
        )
        if is_main:
            print(f"VFIMamba-{args.flow_model_size} loaded for flow-guided Gaussian correspondence + Flow loss")

    # Create model (use updated model config from full config)
    # flow_net is embedded in the model for Phase 1 (frozen) and Phase 2 (unfrozen) joint training
    model = GSMamba(config.model, flow_net=flow_net)
    model = model.to(device)

    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {args.model}")
        print(f"Device: {device}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # Wrap with DDP
    # broadcast_buffers=False: model buffers (grid_x/grid_y/bg_color) are deterministic
    # functions of image_size and get regenerated per-rank via set_image_size().
    # During full eval, rank 0 resizes its buffers to eval resolution while other ranks
    # stay at training size.  With broadcast_buffers=True (default), DDP would try to
    # broadcast rank 0's mismatched-shape buffers at the next forward pass, hanging
    # the collective indefinitely.  Disabling this is safe because every rank already
    # computes the same buffer values from its own input dimensions.
    if world_size > 1:
        if device.type == 'cuda':
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )
        else:
            model = DDP(model, find_unused_parameters=True, broadcast_buffers=False)

    # Create loss function with uncertainty-based weighting
    criterion = build_loss(config.loss, config.model, flow_net=flow_net)
    criterion = criterion.to(device)

    # Guided-flow policy:
    # Enable only for Vimeo midpoint interpolation phase (mode=vimeo_only + vimeo_mode=interp).
    # Disable automatically for X4K/mixed phases.
    flow_policy_enabled = bool(config.loss.use_gflow and flow_net is not None)
    vimeo_mode = getattr(config.data, 'vimeo_mode', 'interp')
    vimeo_midpoint_only = (vimeo_mode == 'interp')
    initial_mode = 'vimeo_only' if args.use_curriculum else args.mode

    if flow_policy_enabled:
        criterion.use_gflow = (initial_mode == 'vimeo_only' and vimeo_midpoint_only)
    else:
        criterion.use_gflow = False

    if is_main:
        print(f"Loss config: uncertainty_weighting={config.loss.use_uncertainty_weighting}, "
              f"warmup_epochs={config.loss.uncertainty_warmup_epochs}")
        if config.loss.use_gflow and flow_net is None:
            print("Warning: Gaussian Flow guidance is disabled because no flow_net was provided.")
            print("         Set loss.use_gflow=False, or pass a pretrained flow network into build_loss().")
        elif flow_policy_enabled:
            print(
                "Gaussian Flow policy: enabled only for Vimeo midpoint interpolation "
                "(mode=vimeo_only, vimeo_mode=interp), disabled otherwise."
            )
            if not vimeo_midpoint_only:
                print(
                    f"Warning: vimeo_mode={vimeo_mode!r}, so Gaussian Flow starts disabled "
                    "because midpoint-only supervision is required."
                )

    prev_use_gflow = criterion.use_gflow

    # Create optimizer and scheduler
    # Pass criterion so log_vars (uncertainty weighting) are stepped, zeroed, and clipped
    optimizer = get_optimizer(model, config, criterion=criterion)

    # Create initial dataloader (will be recreated with curriculum if enabled)
    # When curriculum is disabled, use CLI-provided x4k_steps/x4k_n_frames from config
    train_loader = create_train_loader(
        config,
        rank=rank,
        world_size=world_size,
        mode=args.mode if not args.use_curriculum else 'vimeo_only',
        x4k_steps=config.data.x4k_steps if not args.use_curriculum else None,
        x4k_n_frames=config.data.x4k_n_frames if not args.use_curriculum else None,
        x4k_fraction=config.data.x4k_epoch_fraction,
    )

    scheduler = get_scheduler(optimizer, config)
    scaler = torch.cuda.amp.GradScaler(enabled=config.train.use_amp and device.type == 'cuda')

    # Create eval dataloader
    eval_loader = create_eval_loader(config, dataset_name='vimeo', split='test', batch_size=4)

    # TensorBoard (optional dependency)
    writer = SummaryWriter(output_dir / 'logs') if (is_main and SummaryWriter is not None) else None
    if is_main and SummaryWriter is None:
        print("Warning: tensorboard is not installed. Continuing without TensorBoard logging.")

    # Create samples directory for saving validation visualizations
    samples_dir = output_dir / 'samples'
    if is_main:
        samples_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    start_epoch = 0
    best_psnr = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        _m = model.module if isinstance(model, DDP) else model
        # Drop any keys whose shape doesn't match the current model (e.g. grid_x/grid_y
        # are resolution-dependent buffers recomputed from image_size at build time).
        _current_shapes = {k: v.shape for k, v in _m.state_dict().items()}
        _ckpt_sd = checkpoint['model']
        _shape_skipped = [
            k for k, v in _ckpt_sd.items()
            if k in _current_shapes and v.shape != _current_shapes[k]
        ]
        if _shape_skipped:
            _ckpt_sd = {k: v for k, v in _ckpt_sd.items() if k not in _shape_skipped}
            if is_main:
                print(f"  Resume: skipped {len(_shape_skipped)} shape-mismatched keys (recomputed): {_shape_skipped}")
        _incompatible = _m.load_state_dict(_ckpt_sd, strict=False)
        if is_main and (_incompatible.missing_keys or _incompatible.unexpected_keys):
            if _incompatible.missing_keys:
                print(f"  Resume: missing keys (not in ckpt): {_incompatible.missing_keys}")
            if _incompatible.unexpected_keys:
                print(f"  Resume: unexpected keys (skipped): {_incompatible.unexpected_keys}")
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError as _e:
            if is_main:
                print(f"  Resume: optimizer state skipped ({_e}) — starting with fresh optimizer")
        if scheduler and checkpoint.get('scheduler'):
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
            except Exception as _e:
                if is_main:
                    print(f"  Resume: scheduler state skipped ({_e})")
        try:
            scaler.load_state_dict(checkpoint['scaler'])
        except Exception as _e:
            if is_main:
                print(f"  Resume: scaler state skipped ({_e})")
        start_epoch = checkpoint['epoch'] + 1
        if is_main:
            print(f"Resumed from epoch {start_epoch}")

    # Training loop
    global_step = 0  # Monotonic step counter for TensorBoard (survives dataloader changes)
    if is_main:
        print(f"\nStarting training for {config.train.epochs} epochs")
        print(f"Curriculum: {'enabled' if args.use_curriculum else 'disabled'}")
        print(f"X4K steps: {config.data.x4k_steps}")
        print(f"X4K n_frames: {config.data.x4k_n_frames}")
        print(f"Mixed full coverage: {config.data.full_coverage_mixed}")
        print(f"Mixed drop_last: {config.data.drop_last_mixed}")
        if not args.use_curriculum and args.mode in ['x4k_only', 'mixed']:
            print(f"  -> TEMPO-style: {list(zip(config.data.x4k_steps, config.data.x4k_n_frames))}")
        if config.data.x4k_epoch_fraction < 1.0:
            print(f"X4K epoch fraction: {config.data.x4k_epoch_fraction:.0%} (random subset drawn each epoch)")

    for epoch in range(start_epoch, config.train.epochs):
        # Update dataloader based on curriculum
        if args.use_curriculum:
            curriculum = get_curriculum_settings(config, epoch)
            if is_main and epoch % 10 == 0:
                print(f"Curriculum settings: {curriculum}")

            train_loader = create_train_loader(
                config,
                rank=rank,
                world_size=world_size,
                mode=curriculum.get('mode', 'vimeo_only'),
                x4k_steps=curriculum.get('x4k_steps'),
                x4k_n_frames=curriculum.get('x4k_n_frames'),
                x4k_fraction=config.data.x4k_epoch_fraction,
            )
            current_mode = curriculum.get('mode', 'vimeo_only')
        else:
            current_mode = args.mode

        # Toggle guided-flow supervision by training phase/mode.
        # VFIMamba was trained at t=0.5 only, so flow guidance is valid ONLY for
        # Vimeo midpoint interpolation. Both the loss and the model warping are gated.
        if flow_policy_enabled:
            should_use_gflow = (current_mode == 'vimeo_only' and vimeo_midpoint_only)
            criterion.use_gflow = should_use_gflow
            # Keep model flow_active in sync with loss (same Vimeo-only constraint)
            model_unwrapped = model.module if isinstance(model, DDP) else model
            if hasattr(model_unwrapped, 'flow_active'):
                model_unwrapped.flow_active = should_use_gflow
            if is_main and criterion.use_gflow != prev_use_gflow:
                state = "enabled" if criterion.use_gflow else "disabled"
                print(
                    f"Epoch {epoch}: Gaussian Flow guidance {state} "
                    f"(mode={current_mode}, vimeo_mode={vimeo_mode})."
                )
            prev_use_gflow = criterion.use_gflow

        # Set epoch for distributed sampler
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        elif hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)

        # Train
        train_loss, global_step = train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler, scaler,
            epoch, config, writer, device, rank, world_size,
            global_step=global_step,
        )

        if scheduler is not None:
            scheduler.step()

        run_full_eval_this_epoch = (
            args.eval_full_every > 0 and
            epoch % args.eval_full_every == 0
        )

        # Quick evaluation at the end of every epoch.
        save_samples = (
            args.save_samples_every > 0 and
            epoch % args.save_samples_every == 0
        )

        if run_full_eval_this_epoch:
            # Avoid duplicate validation passes on full-eval epochs.
            eval_loss, eval_psnr = None, None
            is_best = False
            if is_main:
                print(f"Epoch {epoch}: skipping quick eval (full evaluation scheduled).")
        else:
            eval_loss, eval_psnr = evaluate(
                model, criterion, eval_loader, epoch, config, writer, device, rank,
                samples_dir=samples_dir,
                save_samples=save_samples,
            )

            is_best = eval_psnr > best_psnr
            if is_best:
                best_psnr = eval_psnr

        # Save checkpoint every epoch (best is tracked separately).
        if is_main:
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, config, output_dir, is_best
            )

        # Sync all ranks after checkpoint save so no rank races ahead
        # while rank 0 is still writing to disk.
        if world_size > 1:
            dist.barrier()

        # Full benchmark evaluation (less frequent)
        # Includes Vimeo test and X4K cascaded 8x
        # NOTE: ALL ranks enter this block so they all reach the barrier below.
        # evaluate_full() returns {} immediately for rank != 0.
        if run_full_eval_this_epoch:
            # Determine paths (only rank 0 uses them, but compute is cheap)
            vimeo_test_path = args.vimeo_root or config.data.vimeo_root
            x4k_test_path = args.x4k_test_root or config.data.x4k_test_root

            if args.skip_x4k_eval:
                x4k_test_path = None

            # Save samples during full evaluation too
            save_samples_full = (
                args.save_samples_every > 0 and
                epoch % args.save_samples_every == 0
            )

            full_results = evaluate_full(
                model, epoch, config, writer,
                vimeo_path=vimeo_test_path,
                x4k_path=x4k_test_path,
                rank=rank,
                samples_dir=samples_dir,
                save_samples=save_samples_full,
            )

            # Update best based on combined metric (rank 0 only)
            if is_main and full_results:
                combined_psnr = 0.0
                count = 0
                if 'vimeo' in full_results:
                    combined_psnr += full_results['vimeo']['psnr']
                    count += 1
                if 'x4k' in full_results and 'psnr' in full_results['x4k']:
                    combined_psnr += full_results['x4k']['psnr']
                    count += 1
                if count > 0:
                    avg_psnr = combined_psnr / count
                    if avg_psnr > best_psnr:
                        best_psnr = avg_psnr
                        save_checkpoint(
                            model, optimizer, scheduler, scaler,
                            epoch, config, output_dir, is_best=True
                        )

            # Sync all ranks after full eval so non-main ranks don't race ahead
            # into the next epoch's DDP forward pass while rank 0 is still evaluating.
            if world_size > 1:
                dist.barrier()

    # Cleanup
    if writer:
        writer.close()
    cleanup_distributed()

    if is_main:
        print(f"\nTraining completed. Best PSNR: {best_psnr:.2f}dB")
        print(f"Checkpoints saved to: {output_dir}")


if __name__ == '__main__':
    main()
