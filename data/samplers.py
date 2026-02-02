# gsmamba/data/samplers.py
"""
Custom Samplers for GS-Mamba Mixed-Dataset Training

Provides pure batch sampling for training with multiple datasets
that have different frame counts (e.g., Vimeo N=2 vs X4K N=4).
"""

import math
import random
from typing import List, Iterator

import torch
from torch.utils.data import Sampler


class PureBatchSampler(Sampler):
    """
    Samples pure batches from concatenated datasets.

    Each batch contains samples from only ONE dataset (no mixing).
    This is critical when datasets have different frame counts (e.g., N=2 vs N=4)
    because the model expects consistent tensor shapes within a batch.

    Example:
        vimeo = VimeoTriplet(...)   # N=2 frames
        x4k = X4K1000Dataset(...)   # N=4 frames
        concat = ConcatDataset([vimeo, x4k])

        sampler = PureBatchSampler(
            dataset_sizes=[len(vimeo), len(x4k)],
            batch_size=8,
            ratios=[0.7, 0.3],  # 70% Vimeo batches, 30% X4K batches
        )

        loader = DataLoader(concat, batch_sampler=sampler)
    """

    def __init__(
        self,
        dataset_sizes: List[int],
        batch_size: int,
        ratios: List[float],
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        """
        Args:
            dataset_sizes: Number of samples in each dataset [N1, N2, ...]
            batch_size: Samples per batch
            ratios: Proportion of batches from each dataset (must sum to 1.0)
            shuffle: Shuffle samples within each dataset
            drop_last: Drop incomplete batches
        """
        assert len(dataset_sizes) == len(ratios), "Sizes and ratios must match"
        assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {sum(ratios)}"

        self.dataset_sizes = dataset_sizes
        self.batch_size = batch_size
        self.ratios = ratios
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Cumulative offsets for ConcatDataset indexing
        self.offsets = [0]
        for size in dataset_sizes:
            self.offsets.append(self.offsets[-1] + size)

        # Number of batches per dataset
        self.batches_per_dataset = []
        for size in dataset_sizes:
            num_batches = size // batch_size if drop_last else math.ceil(size / batch_size)
            self.batches_per_dataset.append(num_batches)

        # Total batches based on ratios
        total_batches = min(
            int(num_batches / ratio) if ratio > 0 else float('inf')
            for num_batches, ratio in zip(self.batches_per_dataset, ratios)
        )
        self.total_batches = total_batches

        # Actual batches per dataset
        self.actual_batches = [int(total_batches * ratio) for ratio in ratios]

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches with deterministic dataset assignment."""

        # Generate shuffled indices for each dataset
        dataset_indices = []
        for dataset_idx, size in enumerate(self.dataset_sizes):
            indices = list(range(size))
            if self.shuffle:
                random.shuffle(indices)

            # Convert to global ConcatDataset indices
            offset = self.offsets[dataset_idx]
            global_indices = [idx + offset for idx in indices]
            dataset_indices.append(global_indices)

        # Create batches for each dataset
        all_batches = []
        for dataset_idx, (indices, num_batches) in enumerate(
            zip(dataset_indices, self.actual_batches)
        ):
            for batch_idx in range(num_batches):
                start = batch_idx * self.batch_size
                end = start + self.batch_size

                if end <= len(indices):
                    batch = indices[start:end]
                    all_batches.append((dataset_idx, batch))
                elif not self.drop_last and start < len(indices):
                    batch = indices[start:]
                    all_batches.append((dataset_idx, batch))

        # Shuffle batches (but keep each batch pure)
        if self.shuffle:
            random.shuffle(all_batches)

        # Yield batches
        for _, batch in all_batches:
            yield batch

    def __len__(self) -> int:
        return sum(self.actual_batches)


class DistributedPureBatchSampler(Sampler):
    """
    DDP-compatible version of PureBatchSampler.

    Splits data across GPUs while maintaining pure batches.
    Each GPU gets a subset of batches, sampled according to ratios.

    Example:
        sampler = DistributedPureBatchSampler(
            dataset_sizes=[len(vimeo), len(x4k)],
            batch_size=8,
            ratios=[0.7, 0.3],
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
        )

        loader = DataLoader(concat, batch_sampler=sampler)

        # Call set_epoch() before each epoch for proper shuffling
        for epoch in range(num_epochs):
            sampler.set_epoch(epoch)
            for batch in loader:
                ...
    """

    def __init__(
        self,
        dataset_sizes: List[int],
        batch_size: int,
        ratios: List[float],
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        """
        Args:
            dataset_sizes: Number of samples in each dataset
            batch_size: Samples per batch
            ratios: Proportion of batches from each dataset (must sum to 1.0)
            num_replicas: Number of processes (GPUs)
            rank: Rank of current process
            shuffle: Shuffle samples
            drop_last: Drop incomplete batches
            seed: Random seed for reproducibility
        """
        import torch.distributed as dist

        assert len(dataset_sizes) == len(ratios), "Sizes and ratios must match"
        assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {sum(ratios)}"

        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        self.dataset_sizes = dataset_sizes
        self.batch_size = batch_size
        self.ratios = ratios
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Cumulative offsets for ConcatDataset
        self.offsets = [0]
        for size in dataset_sizes:
            self.offsets.append(self.offsets[-1] + size)

        # Batches per dataset (global, before splitting across GPUs)
        self.global_batches = []
        for size in dataset_sizes:
            num_batches = size // batch_size if drop_last else math.ceil(size / batch_size)
            self.global_batches.append(num_batches)

        # Total global batches based on ratios
        total_global_batches = min(
            int(num_batches / ratio) if ratio > 0 else float('inf')
            for num_batches, ratio in zip(self.global_batches, ratios)
        )

        # Batches per replica (per GPU)
        self.batches_per_replica = total_global_batches // num_replicas
        if not drop_last and total_global_batches % num_replicas != 0:
            self.batches_per_replica += 1

        # Actual batches per dataset per replica
        self.actual_batches = [int(self.batches_per_replica * ratio) for ratio in ratios]

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches for this replica."""

        # Set random seed for reproducibility
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Generate shuffled indices for each dataset
        dataset_indices = []
        for dataset_idx, size in enumerate(self.dataset_sizes):
            indices = torch.randperm(size, generator=g).tolist() if self.shuffle else list(range(size))

            # Convert to global ConcatDataset indices
            offset = self.offsets[dataset_idx]
            global_indices = [idx + offset for idx in indices]
            dataset_indices.append(global_indices)

        # Create all batches for each dataset
        all_batches = []
        for dataset_idx, (indices, num_batches) in enumerate(
            zip(dataset_indices, self.actual_batches)
        ):
            # Create batches (total across all replicas)
            for batch_idx in range(num_batches * self.num_replicas):
                start = batch_idx * self.batch_size
                end = start + self.batch_size

                if end <= len(indices):
                    batch = indices[start:end]
                    all_batches.append((dataset_idx, batch))
                elif not self.drop_last and start < len(indices):
                    batch = indices[start:]
                    all_batches.append((dataset_idx, batch))

        # Shuffle batches
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(all_batches)

        # Select batches for this replica
        batches_for_rank = all_batches[self.rank::self.num_replicas]

        # Yield batches
        for _, batch in batches_for_rank:
            yield batch

    def __len__(self) -> int:
        return sum(self.actual_batches)

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling. Call at start of each epoch."""
        self.epoch = epoch


class CurriculumSampler(Sampler):
    """
    Curriculum sampler for progressive N-frame training.

    Starts with simpler tasks (N=2) and gradually introduces
    more complex tasks (N=3, N=4) as training progresses.

    Example:
        sampler = CurriculumSampler(
            dataset=x4k,
            batch_size=8,
            curriculum={
                0: [2],           # Epochs 0-4: N=2 only
                5: [2, 3],        # Epochs 5-9: N=2 and N=3
                10: [2, 3, 4],    # Epochs 10+: All N values
            }
        )
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        curriculum: dict,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        """
        Args:
            dataset: X4K1000Dataset with samples_by_n attribute
            batch_size: Samples per batch
            curriculum: Dict mapping epoch -> list of N values to use
            shuffle: Shuffle samples
            drop_last: Drop incomplete batches
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.curriculum = curriculum
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0

        # Get sorted curriculum milestones
        self.milestones = sorted(curriculum.keys())

    def set_epoch(self, epoch: int):
        """Set current epoch for curriculum."""
        self.epoch = epoch

    def _get_active_n_values(self) -> List[int]:
        """Get N values to use at current epoch."""
        active_milestone = 0
        for milestone in self.milestones:
            if self.epoch >= milestone:
                active_milestone = milestone
            else:
                break
        return self.curriculum[active_milestone]

    def __iter__(self):
        active_n = self._get_active_n_values()
        batches = []

        for n_frames in active_n:
            if n_frames not in self.dataset.samples_by_n:
                continue

            indices = self.dataset.samples_by_n[n_frames].copy()
            if self.shuffle:
                random.shuffle(indices)

            # Create batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        # Shuffle batches
        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self):
        active_n = self._get_active_n_values()
        total = 0
        for n_frames in active_n:
            if n_frames in self.dataset.samples_by_n:
                indices = self.dataset.samples_by_n[n_frames]
                n_batches = len(indices) // self.batch_size
                if not self.drop_last and len(indices) % self.batch_size > 0:
                    n_batches += 1
                total += n_batches
        return total
