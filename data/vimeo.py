# gsmamba/data/vimeo.py
"""
Vimeo-90K Dataset Loaders for GS-Mamba

Supports both triplet (N=2) and septuplet (N=4+) formats.
Adapted from SPACE with GS-Mamba specific optimizations.
"""

import os
import random
from typing import Tuple, List, Literal, Optional

import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image


def _is_main_process() -> bool:
    """Check if this is the main process (for logging)."""
    return os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0"


Mode = Literal["interp", "extrap_fwd", "extrap_bwd", "mix"]


class VimeoTriplet(data.Dataset):
    """
    Vimeo-90K Triplet dataset loader for GS-Mamba.

    Dataset structure:
        root/sequences/<scene>/<clip>/im1.png, im2.png, im3.png
        root/tri_trainlist.txt
        root/tri_testlist.txt

    Modes:
        - 'interp': (im1, im3) -> im2, times [0,1], target 0.5
        - 'extrap_fwd': (im1, im2) -> im3, times [0,1], target 2.0
        - 'extrap_bwd': (im2, im3) -> im1, times [0,1], target -1.0
        - 'mix': randomly sample one of the above each __getitem__

    Returns:
        frames: [2, 3, H, W] - two anchor frames
        anchor_times: [2] - timestamps [0.0, 1.0]
        target_time: scalar - target timestamp
        target: [3, H, W] - target frame
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "test"] = "train",
        mode: Mode = "interp",
        crop_size: Optional[int] = 256,
        aug_flip: bool = True,
        aug_reverse: bool = True,
    ):
        """
        Args:
            root: Dataset root (containing sequences/ and list files)
            split: 'train' or 'test'
            mode: 'interp', 'extrap_fwd', 'extrap_bwd', or 'mix'
            crop_size: Random crop size (None for no crop)
            aug_flip: Enable horizontal flip augmentation
            aug_reverse: Enable temporal reversal augmentation
        """
        super().__init__()
        self.root = root
        self.mode = mode
        self.crop_size = crop_size
        self.aug_flip = aug_flip
        self.aug_reverse = aug_reverse
        self.is_train = (split == "train")

        # Load sequence list
        list_file = os.path.join(root, f"tri_{'train' if split == 'train' else 'test'}list.txt")
        seq_root = os.path.join(root, "sequences")

        if not os.path.isfile(list_file):
            raise FileNotFoundError(f"Missing list file: {list_file}")
        if not os.path.isdir(seq_root):
            raise FileNotFoundError(f"Missing sequences directory: {seq_root}")

        with open(list_file, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        self.items = [ln for ln in lines if not ln.startswith("#")]

        self.to_tensor = T.ToTensor()

        if _is_main_process():
            print(f"[Vimeo Triplet {split}] Loaded {len(self.items)} sequences, mode={mode}")

    def __len__(self) -> int:
        return len(self.items)

    def _load_triplet(self, rel_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load im1, im2, im3 as [3, H, W] tensors in [0, 1]."""
        base = os.path.join(self.root, "sequences", rel_path)

        def load_png(name):
            p = os.path.join(base, f"{name}.png")
            with Image.open(p) as im:
                return self.to_tensor(im.convert("RGB"))

        return load_png("im1"), load_png("im2"), load_png("im3")

    @staticmethod
    def _random_crop_all(frames: List[torch.Tensor], size: int) -> List[torch.Tensor]:
        """Apply same random crop to all frames."""
        if size is None or size == 0:
            return frames
        _, H, W = frames[0].shape
        if H < size or W < size:
            return frames
        top = random.randint(0, H - size)
        left = random.randint(0, W - size)
        return [f[:, top:top + size, left:left + size] for f in frames]

    @staticmethod
    def _hflip_all(frames: List[torch.Tensor]) -> List[torch.Tensor]:
        """Horizontal flip all frames."""
        return [torch.flip(f, dims=[2]) for f in frames]

    def _pack(self, A: torch.Tensor, B: torch.Tensor, tgt: torch.Tensor, tgt_time: float):
        """Pack into output format."""
        frames = torch.stack([A, B], dim=0)  # [2, 3, H, W]
        anchor_times = torch.tensor([0.0, 1.0], dtype=torch.float32)
        target_time = torch.tensor(tgt_time, dtype=torch.float32)
        return frames, anchor_times, target_time, tgt

    def __getitem__(self, idx: int):
        rel = self.items[idx]
        im1, im2, im3 = self._load_triplet(rel)

        # Augmentation
        if self.is_train:
            frames = [im1, im2, im3]
            if self.crop_size:
                frames = self._random_crop_all(frames, self.crop_size)
            if self.aug_flip and random.random() < 0.5:
                frames = self._hflip_all(frames)
            if self.aug_reverse and random.random() < 0.5:
                frames = frames[::-1]  # Temporal reversal
            im1, im2, im3 = frames

        # Mode selection
        if self.mode == "interp":
            return self._pack(im1, im3, im2, 0.5)
        elif self.mode == "extrap_fwd":
            return self._pack(im1, im2, im3, 2.0)
        elif self.mode == "extrap_bwd":
            return self._pack(im2, im3, im1, -1.0)
        else:  # mix
            r = random.random()
            if r < 0.5:
                return self._pack(im1, im3, im2, 0.5)      # 50% interp
            elif r < 0.75:
                return self._pack(im1, im2, im3, 2.0)      # 25% forward extrap
            else:
                return self._pack(im2, im3, im1, -1.0)     # 25% backward extrap


class VimeoSeptuplet(data.Dataset):
    """
    Vimeo-90K Septuplet dataset loader (7 frames) for GS-Mamba.

    Uses configurable anchor frames with intermediate frames as targets.
    Better for GS-Mamba as it provides more temporal context (N > 2).

    Dataset structure:
        root/sequences/<scene>/<clip>/im1.png ... im7.png
        root/sep_trainlist.txt
        root/sep_testlist.txt

    Default: 4 evenly-spaced anchors [0, 2, 4, 6], predict [1, 3, 5]

    Returns:
        frames: [N, 3, H, W] - N anchor frames
        anchor_times: [N] - timestamps [0.0, ..., 1.0]
        target_time: scalar - target timestamp
        target: [3, H, W] - target frame
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "test"] = "train",
        crop_size: Optional[int] = 256,
        aug_flip: bool = True,
        aug_reverse: bool = True,
        input_indices: List[int] = None,
        target_indices: List[int] = None,
    ):
        """
        Args:
            root: Dataset root
            split: 'train' or 'test'
            crop_size: Random crop size
            aug_flip: Enable horizontal flip
            aug_reverse: Enable temporal reversal
            input_indices: Which frames to use as input (0-indexed, default [0, 2, 4, 6])
            target_indices: Which frames can be targets (default [1, 3, 5])
        """
        super().__init__()
        self.root = root
        self.crop_size = crop_size
        self.aug_flip = aug_flip
        self.aug_reverse = aug_reverse
        self.is_train = (split == "train")

        self.input_indices = input_indices or [0, 2, 4, 6]
        self.target_indices = target_indices or [1, 3, 5]
        self.num_frames = 7

        # Validate indices
        all_indices = set(self.input_indices) | set(self.target_indices)
        if max(all_indices) >= self.num_frames or min(all_indices) < 0:
            raise ValueError(f"Indices must be in [0, {self.num_frames - 1}]")

        # Load sequence list
        list_file = os.path.join(root, f"sep_{'train' if split == 'train' else 'test'}list.txt")
        seq_root = os.path.join(root, "sequences")

        if not os.path.isfile(list_file):
            raise FileNotFoundError(f"Missing list file: {list_file}")

        with open(list_file, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        self.items = [ln for ln in lines if not ln.startswith("#")]

        self.to_tensor = T.ToTensor()

        if _is_main_process():
            print(f"[Vimeo Septuplet {split}] Loaded {len(self.items)} sequences, "
                  f"N={len(self.input_indices)} anchors")

    def __len__(self) -> int:
        return len(self.items)

    def _load_frames(self, rel_path: str) -> List[torch.Tensor]:
        """Load all 7 frames."""
        base = os.path.join(self.root, "sequences", rel_path)
        frames = []
        for i in range(1, 8):  # im1.png to im7.png
            p = os.path.join(base, f"im{i}.png")
            with Image.open(p) as im:
                frames.append(self.to_tensor(im.convert("RGB")))
        return frames

    def _random_crop_all(self, frames: List[torch.Tensor], size: int) -> List[torch.Tensor]:
        """Apply same random crop to all frames."""
        if size is None or size == 0:
            return frames
        _, H, W = frames[0].shape
        if H < size or W < size:
            return frames
        top = random.randint(0, H - size)
        left = random.randint(0, W - size)
        return [f[:, top:top + size, left:left + size] for f in frames]

    def _hflip_all(self, frames: List[torch.Tensor]) -> List[torch.Tensor]:
        """Horizontal flip all frames."""
        return [torch.flip(f, dims=[2]) for f in frames]

    def __getitem__(self, idx: int):
        rel = self.items[idx]
        all_frames = self._load_frames(rel)

        # Random target selection
        target_idx = random.choice(self.target_indices)

        # Get input and target indices
        input_indices = self.input_indices.copy()

        # Augmentation (before selecting frames)
        if self.is_train:
            if self.crop_size:
                all_frames = self._random_crop_all(all_frames, self.crop_size)
            if self.aug_flip and random.random() < 0.5:
                all_frames = self._hflip_all(all_frames)
            if self.aug_reverse and random.random() < 0.5:
                # Temporal reversal: reverse frame order and indices
                all_frames = all_frames[::-1]
                max_idx = self.num_frames - 1
                input_indices = [max_idx - i for i in input_indices][::-1]
                target_idx = max_idx - target_idx

        # Get target
        target = all_frames[target_idx]

        # Get input frames
        input_frames = [all_frames[i] for i in input_indices]

        # Stack and compute times
        frames = torch.stack(input_frames, dim=0)  # [N, 3, H, W]

        # Compute normalized times for anchor frames
        anchor_times = torch.tensor(
            [i / (self.num_frames - 1) for i in input_indices],
            dtype=torch.float32
        )
        target_time = torch.tensor(target_idx / (self.num_frames - 1), dtype=torch.float32)

        return frames, anchor_times, target_time, target


def vimeo_collate(batch):
    """
    Collate function for Vimeo datasets.

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
