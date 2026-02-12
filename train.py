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
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from config import get_full_config, update_config_from_args, FullConfig
from models.gs_mamba import GSMamba, build_model
from losses.combined import GSMambaLoss, build_loss
from data import (
    create_train_loader,
    create_eval_loader,
    get_curriculum_settings,
)

# Import evaluation functions for comprehensive validation
from eval import evaluate_vimeo, evaluate_x4k, evaluate_all


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


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


def get_optimizer(model: nn.Module, config: FullConfig) -> torch.optim.Optimizer:
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

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        betas=(0.9, 0.999),
    )

    return optimizer


def get_scheduler(optimizer, config: FullConfig, steps_per_epoch: int):
    """Create learning rate scheduler."""
    total_steps = config.train.epochs * steps_per_epoch
    warmup_steps = config.train.warmup_epochs * steps_per_epoch

    if config.train.scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=config.train.min_lr,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    elif config.train.scheduler == "step":
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=30 * steps_per_epoch, gamma=0.5)
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
    rank: int = 0,
    world_size: int = 1,
):
    """Train for one epoch."""
    model.train()
    criterion.set_epoch(epoch, config.train.epochs)

    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        frames, anchor_times, target_time, target = batch

        # Move to GPU
        frames = frames.cuda(non_blocking=True)
        anchor_times = anchor_times.cuda(non_blocking=True)
        target_time = target_time.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # Forward pass
        optimizer.zero_grad()

        with autocast(enabled=config.train.use_amp):
            # Model forward
            output = model(
                frames=frames,
                t=target_time,
                timestamps=anchor_times,
                return_intermediates=True,
            )

            # Compute loss
            losses = criterion(
                pred=output['pred'],
                target=target,
                render=output['render'],
                depth=output['depth'],
                input_frames=frames,
                gaussians_list=output.get('all_gaussians', []),
                gaussians_interp=output.get('gaussians', {}),
                t=target_time.mean().item(),
            )

            loss = losses['total']

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping
        if config.train.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        # Logging
        global_step = epoch * len(dataloader) + batch_idx
        if batch_idx % config.train.log_every == 0 and rank == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * frames.shape[0] * world_size / elapsed

            lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} "
                f"LR: {lr:.2e} "
                f"Speed: {samples_per_sec:.1f} samples/s"
            )

            # TensorBoard logging
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/lr', lr, global_step)
            for key, value in losses.items():
                if key != 'total':
                    writer.add_scalar(f'train/{key}', value.item(), global_step)

    # Log uncertainty weights at end of epoch
    if rank == 0:
        uncertainty_stats = criterion.get_uncertainty_stats()
        if uncertainty_stats is not None:
            for name, weight in uncertainty_stats['weights'].items():
                writer.add_scalar(f'uncertainty/weight_{name}', weight, epoch)
            for name, sigma in uncertainty_stats['sigmas'].items():
                writer.add_scalar(f'uncertainty/sigma_{name}', sigma, epoch)

            # Print summary
            is_warmup = criterion.is_uncertainty_warmup()
            weight_str = ', '.join([f"{k}={v:.3f}" for k, v in uncertainty_stats['weights'].items()])
            print(f"Epoch {epoch} uncertainty weights ({'warmup' if is_warmup else 'learned'}): {weight_str}")

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader,
    epoch: int,
    config: FullConfig,
    writer: SummaryWriter,
    rank: int = 0,
    samples_dir: Path = None,
    save_samples: bool = False,
):
    """Quick evaluation on dataloader (for frequent checks)."""
    model.eval()

    total_loss = 0.0
    total_psnr = 0.0
    num_batches = 0
    samples_saved = False

    for batch in dataloader:
        frames, anchor_times, target_time, target = batch

        frames = frames.cuda(non_blocking=True)
        anchor_times = anchor_times.cuda(non_blocking=True)
        target_time = target_time.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

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

        # Simple L1 loss for eval
        loss = nn.functional.l1_loss(pred, target)
        total_loss += loss.item()
        num_batches += 1

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

    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / (num_batches * frames.shape[0])

    if rank == 0:
        print(f"Eval Epoch {epoch}: Loss={avg_loss:.4f}, PSNR={avg_psnr:.2f}dB")
        writer.add_scalar('eval/loss', avg_loss, epoch)
        writer.add_scalar('eval/psnr', avg_psnr, epoch)

        # Log sample images to TensorBoard
        if epoch % 10 == 0:
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

    # Get raw model (unwrap DDP if needed)
    raw_model = model.module if hasattr(model, 'module') else model
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
            # Use 2K mode for faster validation during training
            x4k_results = evaluate_x4k(
                raw_model, x4k_path, device, modes=['XTEST-2k'], use_lpips=False,
                samples_dir=samples_dir / "x4k_full" if samples_dir and save_samples else None,
                save_samples=save_samples,
                epoch=epoch,
            )
            results['x4k'] = x4k_results.get('XTEST-2k', {})

            if 'XTEST-2k' in x4k_results:
                r = x4k_results['XTEST-2k']
                print(f"X4K (2K) PSNR: {r['psnr']:.4f} dB")
                print(f"X4K (2K) SSIM: {r['ssim']:.4f}")

                writer.add_scalar('eval_full/x4k_psnr', r['psnr'], epoch)
                writer.add_scalar('eval_full/x4k_ssim', r['ssim'], epoch)
        except Exception as e:
            print(f"X4K evaluation failed: {e}")

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
    parser.add_argument('--mode', type=str, default='vimeo_only',
                        choices=['vimeo_only', 'x4k_only', 'mixed'])

    # Evaluation
    parser.add_argument('--eval_full_every', type=int, default=10,
                        help='Run full benchmark evaluation every N epochs')
    parser.add_argument('--skip_x4k_eval', action='store_true',
                        help='Skip X4K evaluation during training')
    parser.add_argument('--save_samples_every', type=int, default=10,
                        help='Save validation sample images every N epochs (0 to disable)')

    # Training
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--use_amp', action='store_true', default=None)
    parser.add_argument('--no_amp', dest='use_amp', action='store_false')
    parser.add_argument('--use_curriculum', action='store_true', default=True)
    parser.add_argument('--no_curriculum', dest='use_curriculum', action='store_false')

    # Experiment
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--x4k_crop_size', type=int, default=None,
                        help='Crop size for X4K training samples (overrides config.data.x4k_crop_size).')

    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)

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

    # Create model
    model = build_model(args.model)
    model = model.cuda()

    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {args.model}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Create loss function with uncertainty-based weighting
    criterion = build_loss(config.loss, config.model)
    criterion = criterion.cuda()

    if is_main:
        print(f"Loss config: uncertainty_weighting={config.loss.use_uncertainty_weighting}, "
              f"warmup_epochs={config.loss.uncertainty_warmup_epochs}")

    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config)

    # Create initial dataloader (will be recreated with curriculum if enabled)
    # When curriculum is disabled, use CLI-provided x4k_steps/x4k_n_frames from config
    train_loader = create_train_loader(
        config,
        rank=rank,
        world_size=world_size,
        mode=args.mode if not args.use_curriculum else 'vimeo_only',
        x4k_steps=config.data.x4k_steps if not args.use_curriculum else None,
        x4k_n_frames=config.data.x4k_n_frames if not args.use_curriculum else None,
    )

    scheduler = get_scheduler(optimizer, config, len(train_loader))
    scaler = GradScaler(enabled=config.train.use_amp)

    # Create eval dataloader
    eval_loader = create_eval_loader(config, dataset_name='vimeo', split='test', batch_size=4)

    # TensorBoard
    writer = SummaryWriter(output_dir / 'logs') if is_main else None

    # Create samples directory for saving validation visualizations
    samples_dir = output_dir / 'samples'
    if is_main:
        samples_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    start_epoch = 0
    best_psnr = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler and checkpoint.get('scheduler'):
            scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        if is_main:
            print(f"Resumed from epoch {start_epoch}")

    # Training loop
    if is_main:
        print(f"\nStarting training for {config.train.epochs} epochs")
        print(f"Curriculum: {'enabled' if args.use_curriculum else 'disabled'}")
        print(f"X4K steps: {config.data.x4k_steps}")
        print(f"X4K n_frames: {config.data.x4k_n_frames}")
        if not args.use_curriculum and args.mode in ['x4k_only', 'mixed']:
            print(f"  -> TEMPO-style: {list(zip(config.data.x4k_steps, config.data.x4k_n_frames))}")

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
            )

        # Set epoch for distributed sampler
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        elif hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)

        # Train
        train_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler, scaler,
            epoch, config, writer, rank, world_size
        )

        if is_main:
            print(f"Epoch {epoch} completed. Train Loss: {train_loss:.4f}")

        # Quick evaluation (frequent)
        if epoch % config.train.eval_every == 0:
            # Determine if we should save samples this epoch
            save_samples = (
                args.save_samples_every > 0 and
                epoch % args.save_samples_every == 0
            )

            eval_loss, eval_psnr = evaluate(
                model, criterion, eval_loader, epoch, config, writer, rank,
                samples_dir=samples_dir,
                save_samples=save_samples,
            )

            is_best = eval_psnr > best_psnr
            if is_best:
                best_psnr = eval_psnr

            # Save checkpoint
            if is_main:
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch, config, output_dir, is_best
                )

        # Full benchmark evaluation (less frequent)
        # Includes Vimeo test and X4K cascaded 8x
        if epoch % args.eval_full_every == 0 and is_main:
            # Determine paths
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

            # Update best based on combined metric
            if full_results:
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

    # Cleanup
    if writer:
        writer.close()
    cleanup_distributed()

    if is_main:
        print(f"\nTraining completed. Best PSNR: {best_psnr:.2f}dB")
        print(f"Checkpoints saved to: {output_dir}")


if __name__ == '__main__':
    main()
