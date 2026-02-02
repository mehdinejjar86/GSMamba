"""
Gaussian Renderer

Differentiable Gaussian splatting renderer for GS-Mamba.
Wraps the diff-gaussian-rasterization CUDA extension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, NamedTuple
import math
import numpy as np

# Try to import the CUDA rasterizer
try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer
    )
    RASTERIZER_AVAILABLE = True
except ImportError:
    RASTERIZER_AVAILABLE = False
    print("Warning: diff_gaussian_rasterization not available. Using fallback renderer.")


class CameraParams(NamedTuple):
    """Camera parameters for rendering."""
    width: int
    height: int
    fov_x: float
    fov_y: float
    world_view_transform: torch.Tensor
    projection_matrix: torch.Tensor
    full_proj_transform: torch.Tensor
    camera_center: torch.Tensor


def create_canonical_camera(
        width: int,
        height: int,
        fov: float = 0.8,  # ~45 degrees
        z_near: float = 0.01,
        z_far: float = 100.0,
        device: torch.device = torch.device('cuda'),
) -> CameraParams:
    """
    Create a canonical frontal camera.

    Camera is positioned at z=2.0 looking at the origin.

    Args:
        width: Image width
        height: Image height
        fov: Field of view in radians
        z_near: Near clipping plane
        z_far: Far clipping plane
        device: Torch device

    Returns:
        CameraParams with all matrices
    """
    # Camera position: at z=2 looking at origin
    R = np.eye(3)  # Identity rotation (frontal view)
    T = np.array([0, 0, 2.0])  # Translation

    # World to view transform
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.T
    Rt[:3, 3] = -R.T @ T
    Rt[3, 3] = 1.0
    world_view_transform = torch.tensor(Rt, dtype=torch.float32, device=device).T

    # Projection matrix (OpenGL style)
    tan_half_fov = math.tan(fov / 2)
    aspect = width / height

    proj = torch.zeros((4, 4), dtype=torch.float32, device=device)
    proj[0, 0] = 1.0 / (aspect * tan_half_fov)
    proj[1, 1] = 1.0 / tan_half_fov
    proj[2, 2] = -(z_far + z_near) / (z_far - z_near)
    proj[2, 3] = -2.0 * z_far * z_near / (z_far - z_near)
    proj[3, 2] = -1.0
    projection_matrix = proj.T

    # Full projection
    full_proj_transform = world_view_transform @ projection_matrix

    # Camera center
    camera_center = torch.tensor([0, 0, 2.0], dtype=torch.float32, device=device)

    return CameraParams(
        width=width,
        height=height,
        fov_x=fov,
        fov_y=fov,
        world_view_transform=world_view_transform,
        projection_matrix=projection_matrix,
        full_proj_transform=full_proj_transform,
        camera_center=camera_center,
    )


def rotation_to_quaternion(rotation_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert 2D rotation angle to quaternion (rotation around z-axis).

    Args:
        rotation_angle: Rotation angle in radians (B, N, 1) or (N, 1)

    Returns:
        Quaternion (w, x, y, z) with shape (..., 4)
    """
    # Rotation around z-axis
    half_angle = rotation_angle / 2
    w = torch.cos(half_angle)
    x = torch.zeros_like(half_angle)
    y = torch.zeros_like(half_angle)
    z = torch.sin(half_angle)

    return torch.cat([w, x, y, z], dim=-1)


class GaussianRenderer(nn.Module):
    """
    Differentiable Gaussian splatting renderer.

    Renders 3D Gaussians to 2D images using the diff-gaussian-rasterization
    CUDA extension.

    Args:
        image_size: Output image size (H, W)
        fov: Field of view in radians (default: 0.8, ~45 degrees)
        sh_degree: Spherical harmonics degree (default: 0, just RGB)
        bg_color: Background color (default: black)
    """

    def __init__(
            self,
            image_size: Tuple[int, int] = (256, 256),
            fov: float = 0.8,
            sh_degree: int = 0,
            bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        super().__init__()

        self.height, self.width = image_size
        self.fov = fov
        self.sh_degree = sh_degree

        self.register_buffer(
            'bg_color',
            torch.tensor(bg_color, dtype=torch.float32)
        )

        # Camera will be created lazily on first forward pass
        self._camera = None

    def _get_camera(self, device: torch.device) -> CameraParams:
        """Get or create camera parameters."""
        if self._camera is None or self._camera.world_view_transform.device != device:
            self._camera = create_canonical_camera(
                self.width, self.height, self.fov, device=device
            )
        return self._camera

    def _render_single(
            self,
            gaussians: Dict[str, torch.Tensor],
            camera: CameraParams,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render a single batch of Gaussians.

        Args:
            gaussians: Dict with xyz, scale, rotation, opacity, color
                      Each has shape (num_gaussians, C)
            camera: Camera parameters

        Returns:
            Tuple of (rendered_image, depth_image)
        """
        if not RASTERIZER_AVAILABLE:
            return self._fallback_render(gaussians, camera)

        num_gaussians = gaussians['xyz'].shape[0]
        device = gaussians['xyz'].device

        # Prepare Gaussian parameters
        means3D = gaussians['xyz']  # (N, 3)

        # Screenspace points for gradient flow
        screenspace_points = torch.zeros_like(means3D, requires_grad=True) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Scale: apply exp activation
        scales = torch.exp(gaussians['scale'])  # (N, 3)

        # Rotation: convert angle to quaternion
        rotations = rotation_to_quaternion(gaussians['rotation'])  # (N, 4)
        rotations = F.normalize(rotations, dim=-1)

        # Opacity
        opacities = gaussians['opacity']  # (N, 1)

        # Color to SH (degree 0)
        colors = gaussians['color']  # (N, 3)
        # Convert RGB to SH DC coefficient
        # For SH degree 0: color = SH_C0 * dc, where SH_C0 = 0.28209479177387814
        SH_C0 = 0.28209479177387814
        features_dc = (colors - 0.5) / SH_C0  # (N, 3)
        features_dc = features_dc.unsqueeze(1)  # (N, 1, 3)

        # No higher order SH
        features_rest = torch.zeros(
            num_gaussians, 0, 3, device=device, dtype=features_dc.dtype
        )
        shs = torch.cat([features_dc, features_rest], dim=1)  # (N, 1, 3)

        # Setup rasterization settings
        raster_settings = GaussianRasterizationSettings(
            image_height=camera.height,
            image_width=camera.width,
            tanfovx=math.tan(camera.fov_x * 0.5),
            tanfovy=math.tan(camera.fov_y * 0.5),
            bg=self.bg_color.to(device),
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=self.sh_degree,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
            antialiasing=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Render
        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=screenspace_points,
            shs=shs,
            colors_precomp=None,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )

        rendered_image = rendered_image.clamp(0, 1)

        return rendered_image, depth_image

    def _fallback_render(
            self,
            gaussians: Dict[str, torch.Tensor],
            camera: CameraParams,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fallback software renderer (much slower, for testing without CUDA).

        This is a simplified alpha-compositing renderer without proper
        Gaussian splatting. Only use for debugging.
        """
        device = gaussians['xyz'].device
        H, W = camera.height, camera.width

        # Initialize output
        rendered = torch.zeros(3, H, W, device=device)
        depth = torch.zeros(1, H, W, device=device)
        alpha_accum = torch.zeros(1, H, W, device=device)

        # Project Gaussians to screen
        xyz = gaussians['xyz']  # (N, 3)
        colors = gaussians['color']  # (N, 3)
        opacity = gaussians['opacity']  # (N, 1)

        # Simple projection (ignore proper camera transform for now)
        # This is just a placeholder - real rendering requires CUDA rasterizer
        focal = (H + W) / 2
        x_proj = (xyz[:, 0] / (xyz[:, 2] + 1e-6) * focal + W / 2).long()
        y_proj = (xyz[:, 1] / (xyz[:, 2] + 1e-6) * focal + H / 2).long()

        # Sort by depth (far to near for back-to-front compositing)
        depth_order = xyz[:, 2].argsort(descending=True)

        for idx in depth_order:
            x, y = x_proj[idx], y_proj[idx]
            if 0 <= x < W and 0 <= y < H:
                a = opacity[idx, 0] * (1 - alpha_accum[0, y, x])
                rendered[:, y, x] += a * colors[idx]
                alpha_accum[0, y, x] += a
                depth[0, y, x] = xyz[idx, 2]

        return rendered, depth

    def forward(
            self,
            gaussians: Dict[str, torch.Tensor],
            camera: Optional[CameraParams] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Render Gaussians to images.

        Args:
            gaussians: Dict with Gaussian parameters
                      xyz: (B, N, 3) or (N, 3) positions
                      scale: (B, N, 3) or (N, 3) log-scales
                      rotation: (B, N, 1) or (N, 1) rotation angles
                      opacity: (B, N, 1) or (N, 1) opacities
                      color: (B, N, 3) or (N, 3) RGB colors
            camera: Optional camera parameters (uses canonical if None)

        Returns:
            Dict with:
                render: (B, 3, H, W) or (3, H, W) rendered images
                depth: (B, 1, H, W) or (1, H, W) depth maps
        """
        # Handle batched vs unbatched input
        xyz = gaussians['xyz']
        is_batched = xyz.dim() == 3

        if camera is None:
            camera = self._get_camera(xyz.device)

        if is_batched:
            B = xyz.shape[0]
            renders = []
            depths = []

            for b in range(B):
                single_gaussians = {
                    k: v[b] for k, v in gaussians.items()
                }
                render, depth = self._render_single(single_gaussians, camera)
                renders.append(render)
                depths.append(depth)

            return {
                'render': torch.stack(renders, dim=0),
                'depth': torch.stack(depths, dim=0),
            }
        else:
            render, depth = self._render_single(gaussians, camera)
            return {
                'render': render,
                'depth': depth,
            }


class DifferentiableRenderer(nn.Module):
    """
    Wrapper around GaussianRenderer with additional differentiable operations.

    Includes learnable camera parameters and exposure compensation.

    Args:
        image_size: Output image size
        learnable_fov: Whether to learn the field of view
        learnable_exposure: Whether to learn exposure compensation
    """

    def __init__(
            self,
            image_size: Tuple[int, int] = (256, 256),
            fov: float = 0.8,
            learnable_fov: bool = False,
            learnable_exposure: bool = False,
    ):
        super().__init__()

        self.renderer = GaussianRenderer(image_size, fov)

        # Learnable FOV
        if learnable_fov:
            self.fov_param = nn.Parameter(torch.tensor([fov]))
        else:
            self.register_buffer('fov_param', torch.tensor([fov]))

        # Learnable exposure
        if learnable_exposure:
            self.exposure_matrix = nn.Parameter(torch.eye(3))
            self.exposure_bias = nn.Parameter(torch.zeros(3))
        else:
            self.exposure_matrix = None
            self.exposure_bias = None

    def forward(
            self,
            gaussians: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Render with optional exposure compensation.

        Args:
            gaussians: Gaussian parameters

        Returns:
            Rendered outputs
        """
        # Update FOV if learnable
        if isinstance(self.fov_param, nn.Parameter):
            self.renderer.fov = self.fov_param.item()

        outputs = self.renderer(gaussians)

        # Apply exposure compensation
        if self.exposure_matrix is not None:
            render = outputs['render']
            # render: (B, 3, H, W) or (3, H, W)
            if render.dim() == 4:
                render = torch.einsum('bchw,cd->bdhw', render, self.exposure_matrix)
                render = render + self.exposure_bias.view(1, 3, 1, 1)
            else:
                render = torch.einsum('chw,cd->dhw', render, self.exposure_matrix)
                render = render + self.exposure_bias.view(3, 1, 1)
            outputs['render'] = render.clamp(0, 1)

        return outputs
