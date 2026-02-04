# gsmamba/data/__init__.py
"""
GS-Mamba Data Module

Dataloaders for video frame interpolation training with variable N frames.
Supports Vimeo-90K (triplet/septuplet) and X4K1000 (variable step/N).

Output format for all dataloaders:
    frames: [B, N, 3, H, W] - N input anchor frames
    anchor_times: [B, N] - timestamps for each anchor [0.0, ..., 1.0]
    target_time: [B] - target timestep to predict
    target: [B, 3, H, W] - ground truth frame at target_time
"""

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
from data.samplers import PureBatchSampler, DistributedPureBatchSampler
from data.utils import create_train_loader, create_eval_loader, get_dataset, get_curriculum_settings

__all__ = [
    # Vimeo
    "VimeoTriplet",
    "VimeoSeptuplet",
    "vimeo_collate",
    # X4K
    "X4K1000Dataset",
    "X4KTestDataset",
    "X4KSequenceDataset",
    "X4KBatchSampler",
    "DistributedX4KBatchSampler",
    "x4k_collate",
    "x4k_test_collate",
    "x4k_sequence_collate",
    # Samplers
    "PureBatchSampler",
    "DistributedPureBatchSampler",
    # Utilities
    "create_train_loader",
    "create_eval_loader",
    "get_dataset",
    "get_curriculum_settings",
]
