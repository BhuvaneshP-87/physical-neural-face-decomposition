"""Geometry utilities for the differentiable face renderer."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(slots=True)
class ViewTransform:
    """Small-angle view transform used for limited novel-view synthesis."""

    yaw_degrees: float = 0.0
    pitch_degrees: float = 0.0
    roll_degrees: float = 0.0
    depth_parallax: float = 0.12
    translation: tuple[float, float] = (0.0, 0.0)
    scale: float = 1.0


def canonical_grid(
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return a normalized [H, W, 2] grid in the range [-1, 1]."""

    y_coords = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x_coords = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1)


def create_face_mask(
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    axis_ratio: tuple[float, float] = (0.82, 0.95),
) -> torch.Tensor:
    """Create an ellipse-like facial support mask."""

    grid = canonical_grid(height, width, device=device, dtype=dtype)
    x = grid[..., 0] / axis_ratio[0]
    y = grid[..., 1] / axis_ratio[1]
    radius = x.square() + y.square()
    mask = (radius <= 1.0).to(dtype=dtype or torch.float32)
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask


def _ensure_batched_depth(depth: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if depth.ndim == 2:
        return depth.unsqueeze(0).unsqueeze(0), True
    if depth.ndim == 3:
        return depth.unsqueeze(0), True
    if depth.ndim != 4:
        raise ValueError(f"Expected depth map with 2, 3, or 4 dimensions, got {depth.ndim}.")
    return depth, False


def depth_to_normals(depth: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Convert a depth map to unit normals using Sobel filters.

    The depth tensor can be [H, W], [1, H, W], or [B, 1, H, W].
    """

    depth_batched, squeeze_output = _ensure_batched_depth(depth)
    if depth_batched.shape[1] != 1:
        raise ValueError("Depth tensor must have a single channel.")

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=depth_batched.device,
        dtype=depth_batched.dtype,
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=depth_batched.device,
        dtype=depth_batched.dtype,
    ).view(1, 1, 3, 3) / 8.0

    grad_x = F.conv2d(depth_batched, sobel_x, padding=1)
    grad_y = F.conv2d(depth_batched, sobel_y, padding=1)
    ones = torch.ones_like(depth_batched)
    normals = torch.cat((-grad_x, -grad_y, ones), dim=1)
    normals = F.normalize(normals, dim=1, eps=epsilon)

    if squeeze_output:
        return normals.squeeze(0)
    return normals


def depth_to_points(depth: torch.Tensor) -> torch.Tensor:
    """Lift a depth map into a canonical 3D point cloud."""

    depth_batched, squeeze_output = _ensure_batched_depth(depth)
    batch, _, height, width = depth_batched.shape
    grid = canonical_grid(height, width, device=depth_batched.device, dtype=depth_batched.dtype)
    grid = grid.permute(2, 0, 1).unsqueeze(0).expand(batch, -1, -1, -1)
    points = torch.cat((grid[:, :1], grid[:, 1:2], depth_batched), dim=1)
    if squeeze_output:
        return points.squeeze(0)
    return points


def view_rotation_matrix(view: ViewTransform, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a 3x3 rotation matrix for the requested view."""

    yaw = math.radians(view.yaw_degrees)
    pitch = math.radians(view.pitch_degrees)
    roll = math.radians(view.roll_degrees)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    rotation_y = torch.tensor(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        device=device,
        dtype=dtype,
    )
    rotation_x = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]],
        device=device,
        dtype=dtype,
    )
    rotation_z = torch.tensor(
        [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    return rotation_z @ rotation_x @ rotation_y


def rotate_normals(normals: torch.Tensor, view: ViewTransform) -> torch.Tensor:
    """Rotate normals by the requested view transform."""

    squeeze_output = normals.ndim == 3
    if squeeze_output:
        normals = normals.unsqueeze(0)
    if normals.ndim != 4 or normals.shape[1] != 3:
        raise ValueError("Normals must be [B, 3, H, W] or [3, H, W].")

    rotation = view_rotation_matrix(view, device=normals.device, dtype=normals.dtype)
    rotated = torch.einsum("ij,bjhw->bihw", rotation, normals)
    rotated = F.normalize(rotated, dim=1, eps=1e-6)
    if squeeze_output:
        return rotated.squeeze(0)
    return rotated


def warp_grid_for_view(
    depth: torch.Tensor,
    view: ViewTransform,
) -> torch.Tensor:
    """Create a sampling grid for a limited view transform.

    This uses a depth-aware small-angle approximation suitable for short yaw/pitch sweeps.
    """

    depth_batched, squeeze_output = _ensure_batched_depth(depth)
    batch, _, height, width = depth_batched.shape
    grid = canonical_grid(height, width, device=depth.device, dtype=depth.dtype)
    grid = grid.unsqueeze(0).expand(batch, -1, -1, -1).clone()

    yaw = math.radians(view.yaw_degrees)
    pitch = math.radians(view.pitch_degrees)
    roll = math.radians(view.roll_degrees)
    translation_x, translation_y = view.translation

    mean_depth = depth_batched.mean(dim=(-1, -2), keepdim=True)
    centered_depth = depth_batched - mean_depth
    depth_offset = centered_depth.squeeze(1) * view.depth_parallax

    grid[..., 0] = grid[..., 0] + torch.tan(torch.tensor(yaw, device=depth.device, dtype=depth.dtype)) * depth_offset
    grid[..., 1] = grid[..., 1] + torch.tan(torch.tensor(pitch, device=depth.device, dtype=depth.dtype)) * depth_offset

    if abs(view.scale - 1.0) > 1e-6:
        grid = grid * view.scale

    if abs(roll) > 1e-6:
        cos_r = math.cos(roll)
        sin_r = math.sin(roll)
        x_coord = grid[..., 0]
        y_coord = grid[..., 1]
        grid[..., 0] = cos_r * x_coord - sin_r * y_coord
        grid[..., 1] = sin_r * x_coord + cos_r * y_coord

    grid[..., 0] = grid[..., 0] + translation_x
    grid[..., 1] = grid[..., 1] + translation_y

    if squeeze_output:
        return grid.squeeze(0)
    return grid

