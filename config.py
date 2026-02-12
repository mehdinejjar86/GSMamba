"""
GS-Mamba Configuration

Defines model architectures, data, and training configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from functools import partial
import torch.nn as nn


# ==============================================================================
# Data Configuration
# ==============================================================================

@dataclass
class DataConfig:
    """Configuration for data loading."""

    # Vimeo configuration
    vimeo_root: str = "./datasets/vimeo_triplet"
    vimeo_septuplet_root: str = "./datasets/vimeo_septuplet"
    vimeo_mode: str = "interp"  # "interp", "extrap_fwd", "extrap_bwd", "mix"

    # X4K configuration - VARIABLE N FRAMES
    x4k_root: str = "./datasets/x4k"
    x4k_test_root: str = "./datasets/x4k/test"
    x4k_steps: List[int] = field(default_factory=lambda: [5, 31, 31])  # Motion magnitudes
    x4k_n_frames: List[int] = field(default_factory=lambda: [4, 3, 2])  # PAIRED - Variable N per step
    x4k_crop_size: Optional[int] = 256  # X4K train is already 768x768 (no crop needed), test is 2K/4K
    x4k_target_indices: List[int] = field(default_factory=lambda: [16])  # For test (2x)
    x4k_scale: str = "4k"  # "4k" or "2k" for test

    # Training data settings
    crop_size: Optional[int] = None  # Vimeo crop size (None = full resolution 448x256)
    batch_size: int = 4
    num_workers: int = 4

    # Mixed training
    use_mixed_training: bool = False
    dataset_ratios: List[float] = field(default_factory=lambda: [0.7, 0.3])  # [Vimeo, X4K]

    # Augmentation
    aug_flip: bool = True
    aug_reverse: bool = True


# ==============================================================================
# Training Configuration
# ==============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimization
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    epochs: int = 100

    # Batch size
    batch_size: int = 4
    accumulate_grad_batches: int = 1

    # Scheduler
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    min_lr: float = 1e-6

    # Gradient clipping
    grad_clip: float = 1.0

    # Mixed precision
    use_amp: bool = True

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_fraction: float = 0.55  # Fraction of training for curriculum

    # Gaussian Flow curriculum
    gflow_warmup_epochs: int = 0  # Epochs before starting Gaussian Flow
    gflow_decay_start: float = 0.0  # Fraction of training to start decay
    gflow_decay_end: float = 0.5  # Fraction of training when weight reaches 0

    # Checkpointing
    save_every: int = 5
    eval_every: int = 5
    keep_last_n: int = 5

    # Logging
    log_every: int = 100
    wandb_project: str = "gs-mamba"
    wandb_run_name: Optional[str] = None


# ==============================================================================
# Loss Configuration
# ==============================================================================

@dataclass
class LossConfig:
    """Configuration for loss functions.

    Optimized for PSNR/SSIM maximization with optional uncertainty-based weighting.
    """

    # =====================================================================
    # Uncertainty-Based Weighting (Kendall et al.)
    # When enabled, learns optimal weights automatically during training
    # =====================================================================
    use_uncertainty_weighting: bool = True  # Enable learnable loss weights
    uncertainty_warmup_epochs: int = 5  # Epochs before transitioning to learned weights

    # Initial log-variance values for uncertainty weighting
    # Lower values → higher initial weight (log_var=0 → weight=1.0)
    initial_log_vars: Dict[str, float] = field(default_factory=lambda: {
        'l1': 0.0,       # High initial weight (PSNR)
        'ssim': 0.0,     # High initial weight (SSIM metric!)
        'lap': 1.0,      # Medium
        'recon': 1.5,    # Lower
        'depth': 2.0,    # Regularization
        'temporal': 2.0, # Regularization
        'gflow': 1.5,    # Auxiliary
        'opacity_reg': 2.5,
        'scale_reg': 2.5,
    })

    # =====================================================================
    # Fixed Weights (used during warmup or when uncertainty disabled)
    # Optimized for PSNR/SSIM maximization
    # =====================================================================

    # Photometric (primary losses for PSNR/SSIM)
    w_photo: float = 1.0   # L1 loss - good for PSNR
    w_ssim: float = 1.0    # SSIM loss - INCREASED from 0.2 (target metric!)
    w_lap: float = 0.1     # Laplacian pyramid - DECREASED from 0.5 (reduce over-smoothing)

    # Perceptual (DISABLED for pure PSNR/SSIM - conflicts with mathematical metrics)
    w_lpips: float = 0.0   # DISABLED - perceptual loss fights against PSNR/SSIM
    use_lpips: bool = False

    # Reconstruction (reduced weight - focus on interpolation task)
    w_recon: float = 0.1   # DECREASED from 0.5

    # Geometric (minimal regularization)
    w_depth: float = 0.001     # DECREASED from 0.01
    w_temporal: float = 0.01   # DECREASED from 0.1 (allow non-linear motion)

    # Gaussian Flow (faster decay for pure 3D learning)
    w_gflow_max: float = 0.05  # DECREASED from 0.1
    gflow_decay_fraction: float = 0.3  # Decay to 0 over this fraction of training
    use_gflow: bool = False

    # Regularization (minimal)
    w_opacity_reg: float = 0.001  # DECREASED from 0.01
    w_scale_reg: float = 0.001    # DECREASED from 0.01


# ==============================================================================
# Model Configuration
# ==============================================================================

@dataclass
class GSMambaConfig:
    """Configuration for GS-Mamba model."""

    # Model name
    name: str = "gsmamba"

    # Input configuration
    max_n_frames: int = 7  # Maximum number of input frames
    image_size: Tuple[int, int] = (256, 256)

    # Feature encoder (SS2D backbone)
    embed_dims: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    depths: List[int] = field(default_factory=lambda: [2, 2, 2, 3, 3])
    conv_stages: int = 2  # First N stages use conv, rest use Mamba
    d_state: int = 16  # SSM state dimension

    # Temporal fusion
    temporal_hidden_dim: int = 256
    temporal_num_layers: int = 4
    bidirectional: bool = True

    # Gaussian prediction head
    gaussian_channels: int = 11  # depth, depth_scale, xy_offset(2), scale_xy(2), rotation, color(3), opacity

    # Renderer
    sh_degree: int = 0  # Start simple, can increase
    default_fov: float = 0.8  # ~45 degrees

    # UNet refinement
    refine_channels: int = 32
    use_refinement: bool = True

    # Training
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4

    # Loss weights
    w_photo: float = 1.0
    w_lap: float = 0.5
    w_lpips: float = 0.1
    w_recon: float = 0.5
    w_depth: float = 0.01
    w_temporal: float = 0.1
    w_gflow_max: float = 0.1  # Max Gaussian flow weight (decays over training)
    gflow_decay_epochs: float = 0.5  # Decay to 0 over this fraction of training


@dataclass
class GSMambaSmallConfig(GSMambaConfig):
    """Smaller model for faster experimentation."""
    name: str = "gsmamba_small"
    embed_dims: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    temporal_hidden_dim: int = 128
    temporal_num_layers: int = 2
    refine_channels: int = 16


@dataclass
class GSMambaLargeConfig(GSMambaConfig):
    """Larger model for best quality."""
    name: str = "gsmamba_large"
    embed_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    depths: List[int] = field(default_factory=lambda: [2, 2, 2, 4, 4])
    temporal_hidden_dim: int = 512
    temporal_num_layers: int = 6
    refine_channels: int = 64


# Model registry
MODEL_CONFIGS = {
    "gsmamba": GSMambaConfig,
    "gsmamba_small": GSMambaSmallConfig,
    "gsmamba_large": GSMambaLargeConfig,
}


# ==============================================================================
# Full Configuration
# ==============================================================================

@dataclass
class FullConfig:
    """Complete configuration for GS-Mamba training."""

    # Sub-configurations
    model: GSMambaConfig = field(default_factory=GSMambaConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    # Experiment
    exp_name: str = "gsmamba_default"
    output_dir: str = "./outputs"
    seed: int = 42
    deterministic: bool = False

    @classmethod
    def from_model_name(cls, model_name: str = "gsmamba") -> "FullConfig":
        """Create full config with specified model configuration."""
        model_config = MODEL_CONFIGS.get(model_name, GSMambaConfig)()
        return cls(model=model_config)


def get_config(name: str = "gsmamba") -> GSMambaConfig:
    """Get model configuration by name."""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[name]()


def get_full_config(model_name: str = "gsmamba") -> FullConfig:
    """Get full configuration with specified model."""
    return FullConfig.from_model_name(model_name)


def update_config_from_args(config: FullConfig, args) -> FullConfig:
    """
    Update configuration from command line arguments.

    Args:
        config: Base configuration
        args: Namespace with command line arguments

    Returns:
        Updated configuration
    """
    # Update model config
    if hasattr(args, 'image_size') and args.image_size:
        config.model.image_size = tuple(args.image_size)

    # Update data config
    if hasattr(args, 'vimeo_root') and args.vimeo_root:
        config.data.vimeo_root = args.vimeo_root
    if hasattr(args, 'x4k_root') and args.x4k_root:
        config.data.x4k_root = args.x4k_root
    if hasattr(args, 'batch_size') and args.batch_size:
        config.data.batch_size = args.batch_size
        config.train.batch_size = args.batch_size
    if hasattr(args, 'crop_size') and args.crop_size:
        config.data.crop_size = args.crop_size
    if hasattr(args, 'x4k_crop_size') and args.x4k_crop_size:
        config.data.x4k_crop_size = args.x4k_crop_size
    if hasattr(args, 'num_workers') and args.num_workers:
        config.data.num_workers = args.num_workers

    # Update X4K TEMPO-style step/n_frames configuration
    # Note: argparse converts dashes to underscores (x4k-steps -> x4k_steps)
    x4k_steps = getattr(args, 'x4k_steps', None)
    x4k_n_frames = getattr(args, 'x4k_n_frames', None)

    if x4k_steps is not None and x4k_n_frames is not None:
        # Validate that steps and n_frames are paired
        if len(x4k_steps) != len(x4k_n_frames):
            raise ValueError(
                f"--x4k-steps and --x4k-n-frames must have the same length. "
                f"Got steps={x4k_steps} (len={len(x4k_steps)}) and "
                f"n_frames={x4k_n_frames} (len={len(x4k_n_frames)})"
            )
        config.data.x4k_steps = list(x4k_steps)
        config.data.x4k_n_frames = list(x4k_n_frames)
    elif x4k_steps is not None or x4k_n_frames is not None:
        raise ValueError(
            "--x4k-steps and --x4k-n-frames must be specified together. "
            "E.g., --x4k-steps 7 15 31 --x4k-n-frames 4 3 2"
        )

    # Update training config
    if hasattr(args, 'lr') and args.lr:
        config.train.learning_rate = args.lr
    if hasattr(args, 'epochs') and args.epochs:
        config.train.epochs = args.epochs
    if hasattr(args, 'use_amp') and args.use_amp is not None:
        config.train.use_amp = args.use_amp

    # Update experiment info
    if hasattr(args, 'exp_name') and args.exp_name:
        config.exp_name = args.exp_name
    if hasattr(args, 'output_dir') and args.output_dir:
        config.output_dir = args.output_dir
    if hasattr(args, 'seed') and args.seed:
        config.seed = args.seed

    return config
