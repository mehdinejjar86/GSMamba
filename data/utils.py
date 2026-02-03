# gsmamba/data/utils.py
"""
Data Loading Utilities for GS-Mamba

Helper functions to create training and evaluation dataloaders
with proper configuration for different datasets and training modes.
"""

import os
from typing import Optional, List, Union, Literal

import torch
from torch.utils.data import DataLoader, ConcatDataset

from data.vimeo import VimeoTriplet, VimeoSeptuplet, vimeo_collate
from data.x4k import (
    X4K1000Dataset,
    X4KTestDataset,
    X4KSequenceDataset,
    X4KBatchSampler,
    DistributedX4KBatchSampler,
    x4k_collate,
    x4k_test_collate,
    x4k_sequence_collate,
)
from data.samplers import PureBatchSampler, DistributedPureBatchSampler, CurriculumSampler


def get_dataset(
    name: str,
    root: str,
    split: str = "train",
    **kwargs,
):
    """
    Get dataset by name.

    Args:
        name: Dataset name ('vimeo_triplet', 'vimeo_septuplet', 'x4k', 'x4k_test')
        root: Dataset root directory
        split: 'train' or 'test'
        **kwargs: Additional dataset arguments

    Returns:
        Dataset instance
    """
    name = name.lower()

    if name == "vimeo_triplet" or name == "vimeo":
        return VimeoTriplet(root=root, split=split, **kwargs)
    elif name == "vimeo_septuplet":
        return VimeoSeptuplet(root=root, split=split, **kwargs)
    elif name == "x4k" or name == "x4k1000":
        return X4K1000Dataset(root=root, split=split, **kwargs)
    elif name == "x4k_test":
        return X4KTestDataset(root=root, **kwargs)
    elif name == "x4k_sequence":
        return X4KSequenceDataset(root=root, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def create_train_loader(
    config,
    rank: int = 0,
    world_size: int = 1,
    mode: Literal["vimeo_only", "x4k_only", "mixed"] = "vimeo_only",
    x4k_steps: Optional[List[int]] = None,
    x4k_n_frames: Optional[List[int]] = None,
) -> DataLoader:
    """
    Create training dataloader with proper configuration.

    Args:
        config: Configuration object with data settings
        rank: Process rank for DDP
        world_size: Number of processes for DDP
        mode: Training mode
            - 'vimeo_only': Only Vimeo triplet (N=2)
            - 'x4k_only': Only X4K with variable N
            - 'mixed': Both datasets with ratio-based sampling
        x4k_steps: X4K step values (default from config)
        x4k_n_frames: X4K N values (default from config)

    Returns:
        DataLoader instance
    """
    # Get data config
    data_cfg = getattr(config, 'data', config)

    batch_size = getattr(data_cfg, 'batch_size', 4)
    num_workers = getattr(data_cfg, 'num_workers', 4)

    # Default X4K settings
    if x4k_steps is None:
        x4k_steps = getattr(data_cfg, 'x4k_steps', [5, 31, 31])
    if x4k_n_frames is None:
        x4k_n_frames = getattr(data_cfg, 'x4k_n_frames', [4, 3, 2])

    if mode == "vimeo_only":
        # Vimeo triplet (N=2)
        dataset = VimeoTriplet(
            root=getattr(data_cfg, 'vimeo_root', './datasets/vimeo_triplet'),
            split="train",
            mode=getattr(data_cfg, 'vimeo_mode', 'interp'),
            crop_size=getattr(data_cfg, 'crop_size', 256),
            aug_flip=True,
            aug_reverse=True,
        )

        if world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=vimeo_collate,
                pin_memory=True,
                drop_last=True,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=vimeo_collate,
                pin_memory=True,
                drop_last=True,
            )

    elif mode == "x4k_only":
        # X4K with variable N
        dataset = X4K1000Dataset(
            root=getattr(data_cfg, 'x4k_root', './datasets/x4k'),
            split="train",
            steps=x4k_steps,
            n_frames=x4k_n_frames,
            crop_size=getattr(data_cfg, 'x4k_crop_size', 512),
            aug_flip=True,
            aug_reverse=True,
        )

        if world_size > 1:
            sampler = DistributedX4KBatchSampler(
                dataset,
                batch_size=batch_size,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
        else:
            sampler = X4KBatchSampler(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )

        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=x4k_collate,
            pin_memory=True,
        )

    elif mode == "mixed":
        # Mixed training with both datasets
        vimeo = VimeoTriplet(
            root=getattr(data_cfg, 'vimeo_root', './datasets/vimeo_triplet'),
            split="train",
            mode=getattr(data_cfg, 'vimeo_mode', 'interp'),
            crop_size=getattr(data_cfg, 'crop_size', 256),
            aug_flip=True,
            aug_reverse=True,
        )

        x4k = X4K1000Dataset(
            root=getattr(data_cfg, 'x4k_root', './datasets/x4k'),
            split="train",
            steps=x4k_steps,
            n_frames=x4k_n_frames,
            crop_size=getattr(data_cfg, 'x4k_crop_size', 512),
            aug_flip=True,
            aug_reverse=True,
        )

        concat = ConcatDataset([vimeo, x4k])
        ratios = getattr(data_cfg, 'dataset_ratios', [0.7, 0.3])

        if world_size > 1:
            sampler = DistributedPureBatchSampler(
                dataset_sizes=[len(vimeo), len(x4k)],
                batch_size=batch_size,
                ratios=ratios,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
        else:
            sampler = PureBatchSampler(
                dataset_sizes=[len(vimeo), len(x4k)],
                batch_size=batch_size,
                ratios=ratios,
                shuffle=True,
                drop_last=True,
            )

        # Mixed collate function that handles both formats
        def mixed_collate(batch):
            # Check if batch is from Vimeo (N=2) or X4K (variable N)
            # Both return (frames, anchor_times, target_time, target)
            return vimeo_collate(batch)  # Same format, so this works

        loader = DataLoader(
            concat,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=mixed_collate,
            pin_memory=True,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return loader


def create_eval_loader(
    config,
    dataset_name: str = "vimeo",
    split: str = "test",
    batch_size: int = 1,
) -> DataLoader:
    """
    Create evaluation dataloader.

    Args:
        config: Configuration object with data settings
        dataset_name: Dataset name ('vimeo', 'x4k_test', 'x4k_sequence')
        split: 'test' or 'val'
        batch_size: Batch size (usually 1 for evaluation)

    Returns:
        DataLoader instance
    """
    data_cfg = getattr(config, 'data', config)
    num_workers = getattr(data_cfg, 'num_workers', 4)

    if dataset_name == "vimeo" or dataset_name == "vimeo_triplet":
        dataset = VimeoTriplet(
            root=getattr(data_cfg, 'vimeo_root', './datasets/vimeo_triplet'),
            split=split,
            mode="interp",  # Always interpolation for eval
            crop_size=None,  # No crop for eval
            aug_flip=False,
            aug_reverse=False,
        )
        collate = vimeo_collate

    elif dataset_name == "vimeo_septuplet":
        dataset = VimeoSeptuplet(
            root=getattr(data_cfg, 'vimeo_septuplet_root', './datasets/vimeo_septuplet'),
            split=split,
            crop_size=None,
            aug_flip=False,
            aug_reverse=False,
        )
        collate = vimeo_collate

    elif dataset_name == "x4k_test":
        dataset = X4KTestDataset(
            root=getattr(data_cfg, 'x4k_test_root', './datasets/x4k/test'),
            target_indices=getattr(data_cfg, 'x4k_target_indices', [16]),
        )
        collate = x4k_test_collate

    elif dataset_name == "x4k_sequence":
        dataset = X4KSequenceDataset(
            root=getattr(data_cfg, 'x4k_test_root', './datasets/x4k/test'),
            scale=getattr(data_cfg, 'x4k_scale', '4k'),
            target_indices=getattr(data_cfg, 'x4k_target_indices', [4, 8, 12, 16, 20, 24, 28]),
        )
        collate = x4k_sequence_collate

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    return loader


def get_curriculum_settings(config, epoch: int) -> dict:
    """
    Get curriculum learning settings for the current epoch.

    Curriculum schedule:
        Stage 0 (0-10%): Vimeo only (N=2, simple)
        Stage 1 (10-25%): X4K N=4 only (small motion)
        Stage 2 (25-40%): X4K N=3 (medium motion)
        Stage 3 (40-55%): X4K N=2 (large motion)
        Stage 4 (55%+): Full mixed training

    Args:
        config: Configuration with train.epochs
        epoch: Current epoch

    Returns:
        Dict with 'mode', 'x4k_steps', 'x4k_n_frames' settings
    """
    train_cfg = getattr(config, 'train', config)
    total_epochs = getattr(train_cfg, 'epochs', 100)

    # Curriculum phases (percentage of training)
    curriculum_fraction = getattr(train_cfg, 'curriculum_fraction', 0.55)
    curriculum_epochs = int(total_epochs * curriculum_fraction)

    # Stage boundaries
    stage_len = curriculum_epochs // 5

    if epoch >= curriculum_epochs:
        # Full mixed training
        return {
            'mode': 'mixed',
            'x4k_steps': [5, 31, 31],
            'x4k_n_frames': [4, 3, 2],
        }

    stage = epoch // stage_len

    if stage == 0:
        # Stage 0: Vimeo only (simple N=2)
        return {'mode': 'vimeo_only'}
    elif stage == 1:
        # Stage 1: X4K N=4 (small motion, more context)
        return {
            'mode': 'x4k_only',
            'x4k_steps': [5],
            'x4k_n_frames': [4],
        }
    elif stage == 2:
        # Stage 2: X4K N=3 (medium motion)
        return {
            'mode': 'x4k_only',
            'x4k_steps': [31],
            'x4k_n_frames': [3],
        }
    elif stage == 3:
        # Stage 3: X4K N=2 (large motion)
        return {
            'mode': 'x4k_only',
            'x4k_steps': [31],
            'x4k_n_frames': [2],
        }
    else:
        # Stage 4: Mixed X4K
        return {
            'mode': 'x4k_only',
            'x4k_steps': [5, 31, 31],
            'x4k_n_frames': [4, 3, 2],
        }


def resolve_dataset_root(path: str, fallback_env: str = None) -> str:
    """
    Resolve dataset root path, checking environment variables.

    Args:
        path: Path to check
        fallback_env: Environment variable to check if path doesn't exist

    Returns:
        Resolved path
    """
    if os.path.exists(path):
        return path

    if fallback_env and fallback_env in os.environ:
        env_path = os.environ[fallback_env]
        if os.path.exists(env_path):
            return env_path

    # Common paths to check
    common_paths = [
        os.path.expanduser(f"~/datasets/{os.path.basename(path)}"),
        f"/data/{os.path.basename(path)}",
        f"/datasets/{os.path.basename(path)}",
    ]

    for p in common_paths:
        if os.path.exists(p):
            return p

    return path  # Return original even if not found
