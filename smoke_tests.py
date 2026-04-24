#!/usr/bin/env python3
"""
GS-Mamba smoke tests.

Runs a compact suite of sanity checks for:
1) Environment/device visibility
2) Core model components (fusion/interpolator/renderer)
3) Full model forward + backward
4) Optional real-data samples from Vimeo/X4K

Usage examples:
  python smoke_tests.py
  python smoke_tests.py --device cuda --amp
  python smoke_tests.py --datasets none
  python smoke_tests.py --vimeo_root /path/to/vimeo_triplet --x4k_root /path/to/X4K
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import GSMambaSmallConfig
from data.utils import create_eval_loader
from data.vimeo import VimeoTriplet, vimeo_collate
from data.x4k import X4K1000Dataset
from models.gaussian_interpolator import GaussianInterpolator, NFrameGaussianMamba
from models.gs_mamba import GSMamba
from models.renderer import GaussianRenderer, RASTERIZER_AVAILABLE
from models.temporal_fusion import HybridTemporalFusion
from modules.ss2d import SS2D
from train import evaluate


DEFAULT_VIMEO = "/home/nightstalker/Projects/datasets/video/vimeo_triplet"
DEFAULT_X4K = "/home/nightstalker/Projects/datasets/video/X4K"


@dataclass
class SmokeContext:
    args: argparse.Namespace
    device: torch.device
    model: GSMamba | None = None


def _banner(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _make_tiny_model(device: torch.device, image_size: int) -> GSMamba:
    cfg = GSMambaSmallConfig()
    cfg.image_size = (image_size, image_size)
    cfg.depths = [1, 1, 1, 1, 1]
    cfg.embed_dims = [16, 24, 32, 48, 64]
    cfg.temporal_num_layers = 1
    cfg.refine_channels = 8
    cfg.use_refinement = True

    model = GSMamba(cfg).to(device)
    return model


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains non-finite values")


def case_env(ctx: SmokeContext) -> None:
    _banner("Environment")
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"cuda_device_0={torch.cuda.get_device_name(0)}")
    print(f"selected_device={ctx.device}")
    print(f"rasterizer_imported={RASTERIZER_AVAILABLE}")


def case_interpolator_shapes(ctx: SmokeContext) -> None:
    _banner("NFrameGaussianMamba Shapes")
    from models.gaussian_interpolator import NFrameGaussianMamba
    B, N, HW = 2, 3, 8 * 8
    device = ctx.device

    gaussians_list = [
        {
            "xyz":      torch.randn(B, HW, 3, device=device),
            "scale":    torch.rand(B, HW, 3, device=device).add(0.01),
            "rotation": F.normalize(torch.randn(B, HW, 4, device=device), dim=-1),
            "opacity":  torch.sigmoid(torch.randn(B, HW, 1, device=device)),
            "color":    torch.sigmoid(torch.randn(B, HW, 3, device=device)),
        }
        for _ in range(N)
    ]
    timestamps = torch.linspace(0, 1, N, device=device).unsqueeze(0).expand(B, -1)

    model = NFrameGaussianMamba(d_state=8, expand=2, num_layers=2).to(device).eval()

    # scalar t
    out1 = model(gaussians_list, t=0.5, timestamps=timestamps)
    # batched t
    out2 = model(gaussians_list, t=torch.tensor([0.25, 0.75], device=device), timestamps=timestamps)

    for name, out in [("out1", out1), ("out2", out2)]:
        if tuple(out["xyz"].shape) != (B, HW, 3):
            raise AssertionError(f"{name}.xyz shape mismatch: {tuple(out['xyz'].shape)}")
        if tuple(out["rotation"].shape) != (B, HW, 4):
            raise AssertionError(f"{name}.rotation shape mismatch: {tuple(out['rotation'].shape)}")
        norms = out["rotation"].norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
            raise AssertionError(f"{name}.rotation not unit quaternions")
        _assert_finite(f"{name}.xyz", out["xyz"])
        _assert_finite(f"{name}.opacity", out["opacity"])

    print(f"NFrameGaussianMamba outputs correct shapes and unit quaternions for B={B}, N={N}, HW={HW}")


def case_hybrid_temporal_fusion(ctx: SmokeContext) -> None:
    _banner("Hybrid Temporal Fusion")
    B, N, C, H, W = 2, 3, 16, 8, 8
    device = ctx.device

    module = HybridTemporalFusion(
        dim=C,
        num_ssm_layers=2,
        num_attn_layers=1,
        d_state=8,
        num_heads=4,
    ).to(device).eval()

    x = torch.randn(B, N, C, H, W, device=device)
    timestamps = torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.25, 1.0]], device=device)
    y = module(x, timestamps)
    if tuple(y.shape) != tuple(x.shape):
        raise AssertionError(f"shape mismatch: got {tuple(y.shape)}, expected {tuple(x.shape)}")
    _assert_finite("hybrid_temporal_fusion_output", y)
    print(f"fusion output shape={tuple(y.shape)}")


def case_ss2d(ctx: SmokeContext) -> None:
    _banner("SS2D Forward")
    device = ctx.device
    x = torch.randn(1, 8, 8, 16, device=device)
    module = SS2D(d_model=16, d_state=8).to(device).eval()
    y = module(x)
    if tuple(y.shape) != tuple(x.shape):
        raise AssertionError(f"shape mismatch: got {tuple(y.shape)}, expected {tuple(x.shape)}")
    _assert_finite("ss2d_output", y)
    print(f"ss2d output shape={tuple(y.shape)}")


def case_renderer(ctx: SmokeContext) -> None:
    _banner("Renderer Forward")
    device = ctx.device
    B, G, S = 1, 64, ctx.args.size

    gaussians = {
        "xyz": torch.randn(B, G, 3, device=device),
        "scale": torch.rand(B, G, 3, device=device) * 0.05 + 0.01,
        "rotation": F.normalize(torch.randn(B, G, 4, device=device), dim=-1),
        "opacity": torch.sigmoid(torch.randn(B, G, 1, device=device)),
        "color": torch.rand(B, G, 3, device=device),
    }
    gaussians["xyz"][..., 2] = gaussians["xyz"][..., 2].abs() + 1.0

    renderer = GaussianRenderer(image_size=(S, S))
    out = renderer(gaussians)

    if tuple(out["render"].shape) != (B, 3, S, S):
        raise AssertionError(f"render shape mismatch: {tuple(out['render'].shape)}")
    if tuple(out["depth"].shape) != (B, 1, S, S):
        raise AssertionError(f"depth shape mismatch: {tuple(out['depth'].shape)}")
    _assert_finite("renderer_render", out["render"])
    _assert_finite("renderer_depth", out["depth"])
    print(f"renderer output render={tuple(out['render'].shape)} depth={tuple(out['depth'].shape)}")


def case_model_forward(ctx: SmokeContext) -> None:
    _banner("GSMamba Forward")
    device = ctx.device
    B, N, S = ctx.args.batch, ctx.args.frames, ctx.args.size

    if ctx.model is None:
        ctx.model = _make_tiny_model(device, S)
    ctx.model.eval()

    frames = torch.rand(B, N, 3, S, S, device=device)
    timestamps = torch.linspace(0.0, 1.0, N, device=device).unsqueeze(0).expand(B, -1)

    with torch.no_grad():
        out_scalar = ctx.model(frames=frames, t=0.5, timestamps=timestamps, return_intermediates=True)
        out_batched = ctx.model(frames=frames, t=torch.rand(B, device=device), timestamps=timestamps)

    if tuple(out_scalar["pred"].shape) != (B, 3, S, S):
        raise AssertionError(f"pred shape mismatch: {tuple(out_scalar['pred'].shape)}")
    if tuple(out_batched["pred"].shape) != (B, 3, S, S):
        raise AssertionError(f"batched pred shape mismatch: {tuple(out_batched['pred'].shape)}")
    _assert_finite("model_pred_scalar_t", out_scalar["pred"])
    _assert_finite("model_pred_batched_t", out_batched["pred"])
    print(f"pred shape={tuple(out_scalar['pred'].shape)}")


def case_backward_step(ctx: SmokeContext) -> None:
    _banner("Single Optimizer Step")
    device = ctx.device
    B, N, S = 1, max(2, ctx.args.frames), ctx.args.size

    model = _make_tiny_model(device, S).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    frames = torch.rand(B, N, 3, S, S, device=device)
    timestamps = torch.linspace(0.0, 1.0, N, device=device).unsqueeze(0).expand(B, -1)
    target = torch.rand(B, 3, S, S, device=device)

    if ctx.args.amp and device.type == "cuda":
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=True):
            out = model(frames=frames, t=0.5, timestamps=timestamps)
            loss = torch.nn.functional.l1_loss(out["pred"], target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.zero_grad(set_to_none=True)
        out = model(frames=frames, t=0.5, timestamps=timestamps)
        loss = torch.nn.functional.l1_loss(out["pred"], target)
        loss.backward()
        optimizer.step()

    loss_value = float(loss.detach())
    if not torch.isfinite(loss.detach()):
        raise AssertionError(f"loss is non-finite: {loss_value}")
    print(f"train step loss={loss_value:.6f}")


def case_train_help(_: SmokeContext) -> None:
    _banner("Train CLI Help")
    result = subprocess.run(
        [sys.executable, "train.py", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"train.py --help failed:\n{result.stderr}")
    first_line = result.stdout.splitlines()[0] if result.stdout else "<empty>"
    print(first_line)


class _SyntheticVFIDataset(Dataset):
    """Tiny synthetic dataset with train/eval-compatible sample format."""

    def __init__(self, num_samples: int, n_frames: int, size: int):
        self.num_samples = int(num_samples)
        self.n_frames = int(n_frames)
        self.size = int(size)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, _: int):
        S = self.size
        N = self.n_frames
        frames = torch.rand(N, 3, S, S)
        anchor_times = torch.linspace(0.0, 1.0, N)
        target_time = torch.tensor(0.5, dtype=torch.float32)
        target = torch.rand(3, S, S)
        return frames, anchor_times, target_time, target


def case_validation_synthetic(ctx: SmokeContext) -> None:
    _banner("Validation Loop (Synthetic)")
    device = ctx.device
    S = ctx.args.size

    model = _make_tiny_model(device, S).eval()
    dataset = _SyntheticVFIDataset(
        num_samples=max(2, ctx.args.val_batches),
        n_frames=max(2, ctx.args.frames),
        size=S,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(ctx.args.batch, len(dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=vimeo_collate,
        pin_memory=(device.type == "cuda"),
    )

    avg_loss, avg_psnr = evaluate(
        model=model,
        criterion=None,
        dataloader=loader,
        epoch=0,
        config=None,
        writer=None,
        device=device,
        rank=0,
        samples_dir=None,
        save_samples=False,
    )
    if not (torch.isfinite(torch.tensor(avg_loss)) and torch.isfinite(torch.tensor(avg_psnr))):
        raise AssertionError(f"non-finite validation metrics: loss={avg_loss}, psnr={avg_psnr}")
    print(f"synthetic validation metrics: loss={avg_loss:.6f}, psnr={avg_psnr:.3f}dB")


def case_validation_vimeo(ctx: SmokeContext) -> None:
    _banner("Validation Loop (Vimeo Test Split)")
    root = ctx.args.vimeo_root
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Vimeo root does not exist: {root}")

    # Mirror train.py validation path: create_eval_loader(..., dataset_name='vimeo', split='test')
    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.data = _Cfg()
    cfg.data.vimeo_root = root
    cfg.data.num_workers = 0

    loader = create_eval_loader(cfg, dataset_name="vimeo", split="test", batch_size=1)
    batches = []
    for i, batch in enumerate(loader):
        batches.append(batch)
        if i + 1 >= ctx.args.val_batches:
            break
    if not batches:
        raise RuntimeError("No batches produced by Vimeo test validation loader")

    model = _make_tiny_model(ctx.device, ctx.args.size).eval()
    avg_loss, avg_psnr = evaluate(
        model=model,
        criterion=None,
        dataloader=batches,  # bounded subset for quick smoke run
        epoch=0,
        config=None,
        writer=None,
        device=ctx.device,
        rank=0,
        samples_dir=None,
        save_samples=False,
    )
    if not (torch.isfinite(torch.tensor(avg_loss)) and torch.isfinite(torch.tensor(avg_psnr))):
        raise AssertionError(f"non-finite Vimeo validation metrics: loss={avg_loss}, psnr={avg_psnr}")
    print(f"vimeo validation metrics ({len(batches)} batch): loss={avg_loss:.6f}, psnr={avg_psnr:.3f}dB")


def case_vimeo_sample(ctx: SmokeContext) -> None:
    _banner("Vimeo Real Data Sample")
    root = ctx.args.vimeo_root
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Vimeo root does not exist: {root}")

    dataset = VimeoTriplet(
        root=root,
        split="train",
        mode="interp",
        crop_size=ctx.args.crop_size,
        aug_flip=False,
        aug_reverse=False,
    )
    frames, anchor_times, target_time, target = dataset[0]

    print(f"sample frames={tuple(frames.shape)} target={tuple(target.shape)} target_time={float(target_time):.3f}")
    if frames.shape[0] < 2:
        raise AssertionError("Vimeo sample returned fewer than 2 frames")

    if ctx.model is None:
        ctx.model = _make_tiny_model(ctx.device, ctx.args.size)
    ctx.model.eval()

    # Resize sample to configured test size for a fast forward sanity pass.
    frames_b = torch.nn.functional.interpolate(frames, size=(ctx.args.size, ctx.args.size), mode="bilinear", align_corners=False).unsqueeze(0).to(ctx.device)
    target_time_b = target_time.unsqueeze(0).to(ctx.device)
    anchor_times_b = anchor_times.unsqueeze(0).to(ctx.device)
    with torch.no_grad():
        out = ctx.model(frames=frames_b, t=target_time_b, timestamps=anchor_times_b)
    _assert_finite("vimeo_forward_pred", out["pred"])
    print(f"vimeo forward pred={tuple(out['pred'].shape)}")


def case_x4k_sample(ctx: SmokeContext) -> None:
    _banner("X4K Real Data Sample")
    root = ctx.args.x4k_root
    if not os.path.isdir(root):
        raise FileNotFoundError(f"X4K root does not exist: {root}")

    dataset = X4K1000Dataset(
        root=root,
        split="train",
        steps=[ctx.args.x4k_step],
        n_frames=[ctx.args.x4k_n_frames],
        crop_size=ctx.args.crop_size,
        aug_flip=False,
        aug_reverse=False,
    )
    frames, anchor_times, target_time, target = dataset[0]
    print(f"sample frames={tuple(frames.shape)} target={tuple(target.shape)} target_time={float(target_time):.3f}")
    if frames.shape[0] != ctx.args.x4k_n_frames:
        raise AssertionError(f"expected {ctx.args.x4k_n_frames} frames, got {frames.shape[0]}")


def case_slerp_correctness(ctx: SmokeContext) -> None:
    _banner("SLERP Correctness")
    from models.gaussian_interpolator import GaussianInterpolator

    device = ctx.device
    B, P = 4, 16
    q0 = F.normalize(torch.randn(B, P, 4, device=device), dim=-1)
    q1 = F.normalize(torch.randn(B, P, 4, device=device), dim=-1)

    alpha0    = torch.zeros(B, 1, 1, device=device)
    alpha1    = torch.ones(B, 1, 1, device=device)
    alpha_mid = torch.full((B, 1, 1), 0.5, device=device)

    out0    = GaussianInterpolator._slerp(q0, q1, alpha0)
    out1    = GaussianInterpolator._slerp(q0, q1, alpha1)
    out_mid = GaussianInterpolator._slerp(q0, q1, alpha_mid)

    # t=0 → q0
    if not torch.allclose(out0, q0, atol=1e-5):
        raise AssertionError("slerp(q0,q1,0) should equal q0")
    # t=1 → ±q1 element-wise (each position may independently flip sign for shortest arc)
    elem_ok = (torch.isclose(out1, q1, atol=1e-4) | torch.isclose(out1, -q1, atol=1e-4)).all()
    if not elem_ok:
        raise AssertionError("slerp(q0,q1,1) should equal ±q1 element-wise")
    # Unit norm at midpoint
    norms = out_mid.norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
        raise AssertionError(f"slerp output not unit norm: max_err={float((norms - 1).abs().max()):.2e}")
    # Shortest arc: slerp(q, -q, 0.5) → unit result (not zero vector)
    out_neg = GaussianInterpolator._slerp(q0, -q0, alpha_mid)
    norms_neg = out_neg.norm(dim=-1)
    if not torch.allclose(norms_neg, torch.ones_like(norms_neg), atol=1e-4):
        raise AssertionError("slerp shortest-arc failed: output not unit norm")
    _assert_finite("slerp_output", out_mid)
    print("slerp: endpoints correct, unit-norm, shortest-arc OK")


def case_flow_guided_warping(ctx: SmokeContext) -> None:
    _banner("Flow-Guided Gaussian Warping")
    from models.gaussian_interpolator import GaussianInterpolator

    device = ctx.device
    B, H, W = 2, 16, 16
    N = H * W

    gaussians = []
    for _ in range(2):
        gaussians.append({
            "xyz":      torch.randn(B, N, 3, device=device),
            "scale":    torch.rand(B, N, 3, device=device) * 0.1 + 1e-3,
            "rotation": F.normalize(torch.randn(B, N, 4, device=device), dim=-1),
            "opacity":  torch.sigmoid(torch.randn(B, N, 1, device=device)),
            "color":    torch.sigmoid(torch.randn(B, N, 3, device=device)),
        })

    interp = GaussianInterpolator(hidden_dim=32, num_layers=1).to(device).eval()
    timestamps = torch.tensor([0.0, 1.0], device=device)

    with torch.no_grad():
        out_base = interp(gaussians, t=0.5, timestamps=timestamps)
        zero_flow = torch.zeros(B, 4, H, W, device=device)
        out_zero  = interp(gaussians, t=0.5, timestamps=timestamps, flow=zero_flow)
        rand_flow = torch.randn(B, 4, H, W, device=device) * 2.0
        out_rand  = interp(gaussians, t=0.5, timestamps=timestamps, flow=rand_flow)

    if not torch.allclose(out_base['xyz'], out_zero['xyz'], atol=1e-4):
        raise AssertionError("Zero flow warp should match no-flow baseline")
    if torch.allclose(out_rand['xyz'], out_base['xyz'], atol=1e-4):
        raise AssertionError("Non-zero flow should change interpolation output")
    _assert_finite("flow_warped_xyz", out_rand['xyz'])
    print("flow warping: zero-flow=identity, non-zero-flow changes output")


def case_gaussian_head_channels(ctx: SmokeContext) -> None:
    _banner("GaussianHead Output Channels")
    from models.gaussian_head import GaussianHead

    device = ctx.device
    B, C, H, W = 2, 64, 16, 16
    head = GaussianHead(in_channels=C).to(device).eval()

    x = torch.randn(B, C, H, W, device=device)
    with torch.no_grad():
        out = head(x)

    expected_keys = {'depth', 'depth_scale', 'xy_offset', 'scale_xy', 'rotation', 'color', 'opacity'}
    if set(out.keys()) != expected_keys:
        raise AssertionError(f"GaussianHead output keys mismatch: {set(out.keys())} vs {expected_keys}")
    if out['rotation'].shape != (B, 4, H, W):
        raise AssertionError(f"rotation shape: expected (B,4,H,W), got {tuple(out['rotation'].shape)}")
    opacity = out['opacity']
    if not (opacity.min() >= 0 and opacity.max() <= 1):
        raise AssertionError(f"Opacity out of [0,1]: min={float(opacity.min()):.4f}, max={float(opacity.max()):.4f}")
    color = out['color']
    if not (color.min() >= 0 and color.max() <= 1):
        raise AssertionError(f"Color out of [0,1]: min={float(color.min()):.4f}, max={float(color.max()):.4f}")
    rot_norms = out['rotation'].norm(dim=1)  # (B, H, W)
    if not torch.allclose(rot_norms, torch.ones_like(rot_norms), atol=1e-5):
        raise AssertionError(f"Rotation not unit quaternion: max_err={float((rot_norms - 1).abs().max()):.2e}")
    for k, v in out.items():
        _assert_finite(f"gaussian_head_{k}", v)
    print(f"gaussian head: keys={sorted(out.keys())}, rotation_shape={tuple(out['rotation'].shape)}, "
          f"opacity=[{float(opacity.min()):.3f},{float(opacity.max()):.3f}]")


def case_curriculum_schedule(ctx: SmokeContext) -> None:
    _banner("Curriculum Schedule")
    from data.utils import get_curriculum_settings

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.train = _Cfg()
    cfg.train.epochs = 100
    cfg.train.curriculum_fraction = 0.55  # default

    # With 100 epochs: curriculum_epochs=55, stage_len=11
    expected = {
        0:  ('vimeo_only', None, None),
        5:  ('vimeo_only', None, None),
        11: ('x4k_only',  [5],  [4]),
        22: ('x4k_only',  [31], [3]),
        33: ('x4k_only',  [31], [2]),
        44: ('mixed',     [5, 31, 31], [4, 3, 2]),
        55: ('mixed',     [5, 31, 31], [4, 3, 2]),
        99: ('mixed',     [5, 31, 31], [4, 3, 2]),
    }

    for epoch, (exp_mode, exp_steps, exp_nf) in expected.items():
        s = get_curriculum_settings(cfg, epoch)
        if s['mode'] != exp_mode:
            raise AssertionError(f"epoch={epoch}: expected mode={exp_mode!r}, got {s['mode']!r}")
        if exp_steps is not None and s.get('x4k_steps') != exp_steps:
            raise AssertionError(f"epoch={epoch}: expected x4k_steps={exp_steps}, got {s.get('x4k_steps')}")
        if exp_nf is not None and s.get('x4k_n_frames') != exp_nf:
            raise AssertionError(f"epoch={epoch}: expected x4k_n_frames={exp_nf}, got {s.get('x4k_n_frames')}")

    print("curriculum schedule: all stage transitions correct for 100-epoch run")
    for ep in [0, 11, 22, 33, 44, 55, 99]:
        s = get_curriculum_settings(cfg, ep)
        print(f"  epoch={ep:3d}: mode={s['mode']!r:12s} steps={s.get('x4k_steps')} n_frames={s.get('x4k_n_frames')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GS-Mamba smoke tests")
    parser.add_argument("--device", type=str, default="auto", help="auto|cuda|cuda:0|cpu")
    parser.add_argument("--size", type=int, default=32, help="Synthetic spatial size (HxW)")
    parser.add_argument("--batch", type=int, default=2, help="Synthetic batch size")
    parser.add_argument("--frames", type=int, default=3, help="Synthetic N frames")
    parser.add_argument("--amp", action="store_true", help="Run AMP train-step check (CUDA only)")
    parser.add_argument("--datasets", type=str, default="all", choices=["none", "vimeo", "x4k", "all"])
    parser.add_argument("--val_batches", type=int, default=2, help="Number of validation batches to run in validation-loop smoke tests")
    parser.add_argument("--crop_size", type=int, default=128, help="Crop size for real-data sample checks")
    parser.add_argument("--vimeo_root", type=str, default=DEFAULT_VIMEO)
    parser.add_argument("--x4k_root", type=str, default=DEFAULT_X4K)
    parser.add_argument("--x4k_step", type=int, default=7, help="X4K step for sample test")
    parser.add_argument("--x4k_n_frames", type=int, default=4, help="X4K n_frames for sample test")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(requested)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{requested}' but CUDA is unavailable in this shell.")
    return dev


def run_case(name: str, fn: Callable[[SmokeContext], None], ctx: SmokeContext) -> Tuple[str, bool, float]:
    start = time.time()
    try:
        print(f"\n[RUN ] {name}", flush=True)
        fn(ctx)
        elapsed = time.time() - start
        print(f"[PASS] {name} ({elapsed:.2f}s)", flush=True)
        return name, True, elapsed
    except Exception:
        elapsed = time.time() - start
        print(f"[FAIL] {name} ({elapsed:.2f}s)", flush=True)
        traceback.print_exc()
        return name, False, elapsed


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    ctx = SmokeContext(args=args, device=device)

    cases: List[Tuple[str, Callable[[SmokeContext], None]]] = [
        ("environment", case_env),
        ("interpolator_shapes", case_interpolator_shapes),
        ("hybrid_temporal_fusion", case_hybrid_temporal_fusion),
        ("ss2d_forward", case_ss2d),
        ("renderer_forward", case_renderer),
        ("model_forward", case_model_forward),
        ("single_optimizer_step", case_backward_step),
        ("train_help", case_train_help),
        ("validation_synthetic_loop", case_validation_synthetic),
        ("slerp_correctness", case_slerp_correctness),
        ("flow_guided_warping", case_flow_guided_warping),
        ("gaussian_head_channels", case_gaussian_head_channels),
        ("curriculum_schedule", case_curriculum_schedule),
    ]

    if args.datasets in ("vimeo", "all"):
        cases.append(("validation_vimeo_loop", case_validation_vimeo))
        cases.append(("vimeo_real_sample", case_vimeo_sample))
    if args.datasets in ("x4k", "all"):
        cases.append(("x4k_real_sample", case_x4k_sample))

    print("GS-Mamba Smoke Tests", flush=True)
    print(f"device={ctx.device}, datasets={args.datasets}, amp={args.amp}", flush=True)

    results = [run_case(name, fn, ctx) for name, fn in cases]
    failed = [name for name, ok, _ in results if not ok]

    print("\n=== Summary ===", flush=True)
    print(f"total={len(results)} passed={len(results) - len(failed)} failed={len(failed)}", flush=True)
    if failed:
        print("failed_cases=" + ", ".join(failed), flush=True)
        return 1
    print("all smoke tests passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
