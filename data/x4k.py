# gsmamba/data/x4k.py
"""
X4K1000 Dataset Loaders for GS-Mamba

Supports variable N frames with STEP-based motion magnitude control.
Adapted from SPACE with GS-Mamba specific optimizations.

STEP controls motion magnitude, N controls number of anchor frames:
  - step=5, n_frames=4: small motion, more temporal context
  - step=31, n_frames=3: large motion, medium context
  - step=31, n_frames=2: large motion, minimal context
"""

import os
import random
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


def _is_main_process() -> bool:
    """Check if this is the main process (for logging)."""
    return os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0"


class X4K1000Dataset(data.Dataset):
    """
    X4K1000 dataset loader with STEP-based variable N frame sampling.

    Dataset structure:
        root/train/parent_dir/seq_name/0000.png ... 0064.png (65 frames)

    STEP controls motion magnitude, N controls number of anchor frames.
    Like TEMPO: --n_frames 4 3 2 with --x4k_step 5 31 31
    This pairs (step=5, N=4), (step=31, N=3), (step=31, N=2)

    Each sequence (65 frames) generates multiple samples per (step, N) pair.

    Returns:
        frames: [N, 3, H, W] - N anchor frames
        anchor_times: [N] - normalized timestamps [0.0, ..., 1.0]
        target_time: scalar - normalized position of target
        target: [3, H, W] - target frame
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        steps: Union[int, List[int]] = 1,
        crop_size: Optional[int] = 512,
        aug_flip: bool = True,
        aug_reverse: bool = True,
        n_frames: Union[int, List[int]] = 4,
    ):
        """
        Args:
            root: Dataset root (containing train/ and test/ directories)
            split: 'train' or 'test'
            steps: Frame spacing parameter(s). Can be:
                - Single int: steps=5 (only that motion magnitude)
                - List of ints: steps=[5, 31, 31] (multiple motion magnitudes)
            crop_size: Spatial crop for 4K images (None for no crop)
            aug_flip: Enable horizontal flip augmentation
            aug_reverse: Enable temporal reversal augmentation
            n_frames: Number of anchor frames. Can be:
                - Single int: n_frames=4 (same N for all steps)
                - List of ints: n_frames=[4, 3, 2] (paired with steps)
                  Must have same length as steps when both are lists.
        """
        self.root = Path(root)
        self.split = split
        self.is_train = (split == "train")

        # Convert single int to list for uniform handling
        self.steps = [steps] if isinstance(steps, int) else list(steps)
        self.n_frames_list = [n_frames] if isinstance(n_frames, int) else list(n_frames)

        # If single n_frames, expand to match all steps
        if len(self.n_frames_list) == 1 and len(self.steps) > 1:
            self.n_frames_list = self.n_frames_list * len(self.steps)

        # Validate matching lengths
        if len(self.n_frames_list) != len(self.steps):
            raise ValueError(
                f"n_frames ({len(self.n_frames_list)}) must match steps ({len(self.steps)}) "
                f"or be a single value. Got n_frames={n_frames}, steps={steps}"
            )

        self.crop_size = crop_size
        self.aug_flip = aug_flip
        self.aug_reverse = aug_reverse

        # For backward compatibility: single n_frames for datasets with uniform N
        self.n_frames = self.n_frames_list[0] if len(set(self.n_frames_list)) == 1 else None

        # Scan all sequences
        self.sequences = self._scan_sequences()
        if _is_main_process():
            print(f"[X4K {split}] Found {len(self.sequences)} sequences")

        # Generate (sequence_idx, target_idx, anchors, n_frames) tuples
        self.samples = self._generate_samples()

        # Group samples by n_frames for efficient batch sampling
        self._index_by_n_frames()

        if _is_main_process():
            print(f"[X4K {split}] Generated {len(self.samples)} samples with "
                  f"(steps, n_frames)={list(zip(self.steps, self.n_frames_list))}")

    def _scan_sequences(self) -> List[str]:
        """Scan nested directory structure for 65-frame sequences."""
        sequences = []
        split_dir = self.root / self.split

        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        # Iterate through parent dirs (002, 003, ...)
        for parent_dir in sorted(split_dir.iterdir()):
            if not parent_dir.is_dir():
                continue

            # Iterate through sequence dirs (occ*.*)
            for seq_dir in sorted(parent_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue

                # Verify 65 frames
                frames = sorted(seq_dir.glob("*.png"))
                if len(frames) == 65:
                    # Store relative path from split_dir
                    rel_path = seq_dir.relative_to(split_dir)
                    sequences.append(str(rel_path))

        return sequences

    def _generate_samples(self) -> List[Tuple[int, int, List[int], int]]:
        """
        Generate all (sequence, target, anchors, n_frames) tuples using sliding window.

        Sliding window approach:
        - For step=7, N=4: window has anchors spaced by 8 frames
        - Window slides through the 65-frame sequence
        - Each window position generates samples for all valid target frames

        Returns:
            List of (seq_idx, target_frame, anchors, n_frames) tuples
        """
        all_samples = []

        for step, n_frames in zip(self.steps, self.n_frames_list):
            # Anchor spacing = step + 1 (step=7 means 0→8, spacing of 8)
            spacing = step + 1

            # Window span: from first to last anchor
            window_span = (n_frames - 1) * spacing

            # Check if window fits in sequence
            if window_span >= 65:
                max_step = 64 // (n_frames - 1) - 1
                raise ValueError(
                    f"step={step} with n_frames={n_frames} (spacing={spacing}) "
                    f"produces window_span={window_span}, but sequence only has 65 frames (0-64). "
                    f"Max step for N={n_frames} is {max_step}"
                )

            # Valid window starting positions
            max_start = 64 - window_span
            num_windows = max_start + 1

            step_samples = 0

            for seq_idx in range(len(self.sequences)):
                for window_start in range(num_windows):
                    # Compute anchor positions for this window
                    anchors = [window_start + i * spacing for i in range(n_frames)]

                    # Valid targets: all frames between first and last anchor, excluding anchors
                    anchor_set = set(anchors)
                    valid_targets = [
                        i for i in range(anchors[0] + 1, anchors[-1])
                        if i not in anchor_set
                    ]

                    # Generate sample for each target
                    for target_frame in valid_targets:
                        all_samples.append((seq_idx, target_frame, anchors, n_frames))
                        step_samples += 1

            targets_per_window = window_span - 1 - (n_frames - 2)  # span - 1 minus interior anchors
            if _is_main_process():
                print(f"  STEP={step}, N={n_frames}: {num_windows} windows × "
                      f"{targets_per_window} targets × {len(self.sequences)} seqs = {step_samples} samples")

        return all_samples

    def _index_by_n_frames(self):
        """Index samples by n_frames for efficient batch sampling with same N."""
        self.samples_by_n = {}
        for idx, sample in enumerate(self.samples):
            n_frames = sample[3]
            if n_frames not in self.samples_by_n:
                self.samples_by_n[n_frames] = []
            self.samples_by_n[n_frames].append(idx)

        # Print distribution
        if _is_main_process():
            for n, indices in sorted(self.samples_by_n.items()):
                print(f"    N={n}: {len(indices)} samples")

    def get_n_frames_values(self) -> List[int]:
        """Get unique n_frames values in this dataset."""
        return sorted(self.samples_by_n.keys())

    def get_samples_with_n(self, n_frames: int) -> List[int]:
        """Get sample indices that have the given n_frames."""
        return self.samples_by_n.get(n_frames, [])

    def _load_frames(self, seq_path: str, frame_indices: List[int]) -> List[torch.Tensor]:
        """Load frames at specified indices."""
        frames = []
        seq_dir = self.root / self.split / seq_path

        for idx in frame_indices:
            frame_path = seq_dir / f"{idx:04d}.png"
            with Image.open(frame_path) as img:
                img = img.convert("RGB")
                arr = np.array(img)
                tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                frames.append(tensor)

        return frames

    def _random_crop_all(self, frames: List[torch.Tensor], size: int) -> List[torch.Tensor]:
        """Apply same random crop to all frames."""
        if size is None or size == 0:
            return frames

        _, H, W = frames[0].shape
        if H <= size and W <= size:
            return frames

        # Random crop parameters
        top = random.randint(0, max(0, H - size))
        left = random.randint(0, max(0, W - size))

        return [f[:, top:top + size, left:left + size] for f in frames]

    def _hflip_all(self, frames: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply same horizontal flip to all frames."""
        return [torch.flip(f, dims=[2]) for f in frames]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            frames: [N, 3, H, W] anchor frames (N varies per sample)
            anchor_times: [N] normalized times [0.0, ..., 1.0]
            target_time: scalar, normalized position of target
            target: [3, H, W] target frame
        """
        seq_idx, target_frame, anchors, n_frames = self.samples[idx]
        seq_path = self.sequences[seq_idx]

        # Load anchors + target
        all_indices = anchors + [target_frame]
        all_frames = self._load_frames(seq_path, all_indices)

        # Augmentations (consistent across all frames)
        if self.is_train:
            if self.crop_size:
                all_frames = self._random_crop_all(all_frames, self.crop_size)
            if self.aug_flip and random.random() < 0.5:
                all_frames = self._hflip_all(all_frames)
            if self.aug_reverse and random.random() < 0.5:
                # Temporal reversal
                all_frames = all_frames[::-1]
                # Recompute target position (reverse the anchor order)
                anchors = anchors[::-1]
                # Target position relative to reversed anchors
                target_frame = anchors[0] + (anchors[-1] - target_frame)

        # Split anchors and target
        anchor_frames = all_frames[:n_frames]
        target_data = all_frames[-1]

        # Stack anchors into [N, 3, H, W]
        frames = torch.stack(anchor_frames, dim=0)

        # Compute normalized times
        anchor_times = torch.linspace(0.0, 1.0, n_frames)

        # Target time (normalized position between first and last anchor)
        original_anchors = self.samples[idx][2]
        target_time = (target_frame - original_anchors[0]) / (original_anchors[-1] - original_anchors[0])
        target_time = torch.tensor(target_time, dtype=torch.float32).clamp(0.0, 1.0)

        return frames, anchor_times, target_time, target_data


class X4KBatchSampler(data.Sampler):
    """
    Batch sampler that groups samples by n_frames.

    Ensures all samples in a batch have the same number of anchor frames,
    required for tensor stacking. Shuffles within each N group.
    """

    def __init__(
        self,
        dataset: X4K1000Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Get samples grouped by N
        self.samples_by_n = dataset.samples_by_n

    def __iter__(self):
        batches = []

        for n_frames, indices in self.samples_by_n.items():
            # Shuffle indices for this N
            if self.shuffle:
                indices = indices.copy()
                random.shuffle(indices)

            # Create batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        # Shuffle batches across all N values
        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self):
        total = 0
        for indices in self.samples_by_n.values():
            n_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size > 0:
                n_batches += 1
            total += n_batches
        return total


class DistributedX4KBatchSampler(data.Sampler):
    """
    Distributed batch sampler that groups samples by n_frames.

    For multi-GPU training with DDP. Each rank gets different batches
    but all samples in each batch have the same N.
    """

    def __init__(
        self,
        dataset: X4K1000Dataset,
        batch_size: int,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        import torch.distributed as dist

        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        self.samples_by_n = dataset.samples_by_n

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling across ranks."""
        self.epoch = epoch

    def __iter__(self):
        # Create generator with seed for reproducibility
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        all_batches = []

        for n_frames, indices in self.samples_by_n.items():
            # Shuffle indices for this N
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in perm]

            # Create batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)

        # Shuffle all batches
        if self.shuffle:
            perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in perm]

        # Distribute batches across ranks
        # Pad to make divisible by num_replicas
        num_batches = len(all_batches)
        total_batches = ((num_batches + self.num_replicas - 1) // self.num_replicas) * self.num_replicas

        # Repeat batches to fill
        if total_batches > num_batches:
            all_batches = all_batches + all_batches[:total_batches - num_batches]

        # Subsample for this rank
        batches_per_rank = total_batches // self.num_replicas
        start_idx = self.rank * batches_per_rank
        end_idx = start_idx + batches_per_rank

        yield from all_batches[start_idx:end_idx]

    def __len__(self):
        total = 0
        for indices in self.samples_by_n.values():
            n_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size > 0:
                n_batches += 1
            total += n_batches
        return (total + self.num_replicas - 1) // self.num_replicas


class X4KTestDataset(data.Dataset):
    """
    X4K1000 TEST dataset loader for validation/evaluation.

    Different structure from training:
        root/Type1/TEST01_xxx_fxxxx/0000.png ... 0032.png (33 frames)
        root/Type2/...
        root/Type3/...

    Type1/2/3 represent different motion difficulty levels.
    15 total sequences (5 per type).

    Standard X4K evaluation protocol:
    - Use frames 0 and 32 as anchors (N=2)
    - Predict frame 16 (middle) for 2x interpolation
    - Or predict multiple intermediate frames for Nx interpolation
    """

    def __init__(
        self,
        root: str,
        multi_target: bool = False,
        target_indices: List[int] = None,
    ):
        """
        Args:
            root: Dataset root (containing Type1/, Type2/, Type3/)
            multi_target: If True, generate multiple targets per sequence
            target_indices: Which frames to predict (default: [16] for 2x)
                           For 8x: [4, 8, 12, 16, 20, 24, 28]
        """
        self.root = Path(root)
        self.multi_target = multi_target
        self.target_indices = target_indices or [16]  # Default: predict middle frame
        self.num_frames = 33  # 0 to 32

        # Scan sequences
        self.sequences = self._scan_sequences()
        if _is_main_process():
            print(f"[X4K Test] Found {len(self.sequences)} sequences")

        # Generate samples
        self.samples = self._generate_samples()
        if _is_main_process():
            print(f"[X4K Test] Generated {len(self.samples)} samples (targets: {self.target_indices})")

    def _scan_sequences(self) -> List[Tuple[str, str, str]]:
        """Scan Type1/2/3 directories for test sequences."""
        sequences = []

        for type_dir in sorted(self.root.iterdir()):
            if not type_dir.is_dir() or not type_dir.name.startswith("Type"):
                continue

            for seq_dir in sorted(type_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue

                # Verify frame count
                frames = sorted(seq_dir.glob("*.png"))
                if len(frames) == self.num_frames:
                    sequences.append((type_dir.name, seq_dir.name, str(seq_dir)))

        return sequences

    def _generate_samples(self) -> List[Tuple[int, int]]:
        """Generate (seq_idx, target_frame) pairs."""
        samples = []

        for seq_idx in range(len(self.sequences)):
            for target_idx in self.target_indices:
                samples.append((seq_idx, target_idx))

        return samples

    def _load_frame(self, seq_path: str, frame_idx: int) -> torch.Tensor:
        """Load a single frame."""
        frame_path = Path(seq_path) / f"{frame_idx:04d}.png"
        with Image.open(frame_path) as img:
            img = img.convert("RGB")
            arr = np.array(img)
            return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Returns:
            frames: [2, 3, H, W] - first and last frame
            anchor_times: [2] - [0.0, 1.0]
            target_time: scalar - normalized position
            target: [3, H, W] - target frame
        """
        seq_idx, target_frame = self.samples[idx]
        type_name, seq_name, seq_path = self.sequences[seq_idx]

        # Load anchor frames (first and last)
        frame_0 = self._load_frame(seq_path, 0)
        frame_32 = self._load_frame(seq_path, 32)
        frames = torch.stack([frame_0, frame_32], dim=0)

        # Load target frame
        target = self._load_frame(seq_path, target_frame)

        # Normalized times
        anchor_times = torch.tensor([0.0, 1.0], dtype=torch.float32)
        target_time = torch.tensor(target_frame / 32.0, dtype=torch.float32)

        return frames, anchor_times, target_time, target


class X4KSequenceDataset(data.Dataset):
    """
    X4K1000 TEST dataset for cascaded 8x evaluation.

    Returns anchor frames (0, 32) and ground truth for 7 target frames
    for cascaded prediction: N=2 → N=3 → N=4.
    """

    def __init__(
        self,
        root: str,
        scale: str = "4k",  # "4k" or "2k"
        target_indices: List[int] = None,
    ):
        """
        Args:
            root: Dataset root (containing Type1/, Type2/, Type3/)
            scale: "4k" for full resolution, "2k" for downscaled (2048x1080)
            target_indices: Which frames to predict (default: [4, 8, 12, 16, 20, 24, 28])
        """
        self.root = Path(root)
        self.scale = scale
        self.target_indices = target_indices or [4, 8, 12, 16, 20, 24, 28]  # 8x
        self.num_frames = 33  # 0 to 32

        # Scan sequences
        self.sequences = self._scan_sequences()
        if _is_main_process():
            print(f"[X4K Sequence] Found {len(self.sequences)} sequences (scale={scale})")

    def _scan_sequences(self) -> List[Tuple[str, str, str]]:
        """Scan Type1/2/3 directories for test sequences."""
        sequences = []

        for type_dir in sorted(self.root.iterdir()):
            if not type_dir.is_dir() or not type_dir.name.startswith("Type"):
                continue

            for seq_dir in sorted(type_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue

                # Verify frame count
                frames = sorted(seq_dir.glob("*.png"))
                if len(frames) == self.num_frames:
                    sequences.append((type_dir.name, seq_dir.name, str(seq_dir)))

        return sequences

    def _load_frame(self, seq_path: str, frame_idx: int) -> torch.Tensor:
        """Load a single frame, optionally resized for 2k mode."""
        try:
            import cv2
            use_cv2 = True
        except ImportError:
            use_cv2 = False

        frame_path = Path(seq_path) / f"{frame_idx:04d}.png"

        if use_cv2:
            img = cv2.imread(str(frame_path))
            if img is None:
                raise FileNotFoundError(f"Could not load: {frame_path}")

            # Resize for 2K mode
            if self.scale == "2k":
                img = cv2.resize(img, (2048, 1080), interpolation=cv2.INTER_AREA)

            # Convert BGR to RGB, normalize to [0, 1]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            return torch.from_numpy(img).permute(2, 0, 1)
        else:
            # Fallback to PIL
            with Image.open(frame_path) as img:
                img = img.convert("RGB")
                if self.scale == "2k":
                    img = img.resize((2048, 1080), Image.LANCZOS)
                arr = np.array(img).astype(np.float32) / 255.0
                return torch.from_numpy(arr).permute(2, 0, 1)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        """
        Returns:
            frame_0: [3, H, W] - first frame
            frame_32: [3, H, W] - last frame
            gt_frames: dict {idx: [3, H, W]} - ground truth for target indices
            metadata: dict with 'type', 'sequence', 'scale'
        """
        type_name, seq_name, seq_path = self.sequences[idx]

        # Load anchor frames (first and last)
        frame_0 = self._load_frame(seq_path, 0)
        frame_32 = self._load_frame(seq_path, 32)

        # Load all target ground truth frames
        gt_frames = {}
        for target_idx in self.target_indices:
            gt_frames[target_idx] = self._load_frame(seq_path, target_idx)

        metadata = {
            'type': type_name,
            'sequence': seq_name,
            'scale': self.scale,
        }

        return frame_0, frame_32, gt_frames, metadata


def x4k_collate(batch):
    """
    Collate function for X4K batches.

    IMPORTANT: All samples in batch must have same N (number of anchor frames).
    Use X4KBatchSampler or DistributedX4KBatchSampler to ensure this.

    Returns:
        frames: [B, N, 3, H, W]
        anchor_times: [B, N]
        target_time: [B]
        target: [B, 3, H, W]
    """
    frames = torch.stack([b[0] for b in batch], dim=0)
    anchor_times = torch.stack([b[1] for b in batch], dim=0)
    target_time = torch.stack([b[2] for b in batch], dim=0)
    target = torch.stack([b[3] for b in batch], dim=0)
    return frames, anchor_times, target_time, target


def x4k_test_collate(batch):
    """Collate function for X4K test batches."""
    frames = torch.stack([b[0] for b in batch], dim=0)
    anchor_times = torch.stack([b[1] for b in batch], dim=0)
    target_time = torch.stack([b[2] for b in batch], dim=0)
    target = torch.stack([b[3] for b in batch], dim=0)
    return frames, anchor_times, target_time, target


def x4k_sequence_collate(batch):
    """
    Collate function for X4KSequenceDataset.

    Returns:
        frame_0: [B, 3, H, W]
        frame_32: [B, 3, H, W]
        gt_frames: list of dicts
        metadata: list of dicts
    """
    frame_0 = torch.stack([b[0] for b in batch], dim=0)
    frame_32 = torch.stack([b[1] for b in batch], dim=0)
    gt_frames = [b[2] for b in batch]
    metadata = [b[3] for b in batch]
    return frame_0, frame_32, gt_frames, metadata
