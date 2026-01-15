"""Differentiable PyTorch fallback renderer for face inverse rendering."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import ViewTransform, create_face_mask, depth_to_normals, warp_grid_for_view
from .lighting import spherical_harmonics_shading


@dataclass(slots=True)
class RenderResult:
    """Container returned by the renderer."""

    image: torch.Tensor
    shading: torch.Tensor
    normals: torch.Tensor
    depth: torch.Tensor
    mask: torch.Tensor | None = None
    warped_albedo: torch.Tensor | None = None


def _ensure_batched_image(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if image.ndim == 3:
        return image.unsqueeze(0), True
    if image.ndim != 4:
        raise ValueError(f"Expected image with 3 or 4 dimensions, got {image.ndim}.")
    return image, False


def _broadcast_mask(mask: torch.Tensor, batch: int) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.ndim != 4:
        raise ValueError("Mask must be [H, W], [1, H, W], or [B, 1, H, W].")
    if mask.shape[0] == 1 and batch > 1:
        mask = mask.expand(batch, -1, -1, -1)
    return mask


def _warp_image(image: torch.Tensor, depth: torch.Tensor, view: ViewTransform) -> torch.Tensor:
    grid = warp_grid_for_view(depth, view)
    if grid.ndim == 3:
        grid = grid.unsqueeze(0)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)


class TorchFaceRenderer(nn.Module):
    """A compact differentiable renderer that uses Lambertian SH shading."""

    def __init__(
        self,
        *,
        background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        clamp_output: bool = True,
        use_soft_mask: bool = True,
    ) -> None:
        super().__init__()
        background = torch.tensor(background_color, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("background_color", background)
        self.clamp_output = clamp_output
        self.use_soft_mask = use_soft_mask

    def forward(
        self,
        depth: torch.Tensor,
        albedo: torch.Tensor,
        lighting_coefficients: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        view: ViewTransform | None = None,
        residual: torch.Tensor | None = None,
    ) -> RenderResult:
        depth_batched, depth_squeezed = _ensure_batched_image(depth)
        albedo_batched, albedo_squeezed = _ensure_batched_image(albedo)

        if depth_batched.shape[1] != 1:
            raise ValueError("Depth must have a single channel.")
        if albedo_batched.shape[1] != 3:
            raise ValueError("Albedo must have three channels.")
        if depth_batched.shape[0] != albedo_batched.shape[0]:
            if depth_batched.shape[0] == 1:
                depth_batched = depth_batched.expand(albedo_batched.shape[0], -1, -1, -1)
            elif albedo_batched.shape[0] == 1:
                albedo_batched = albedo_batched.expand(depth_batched.shape[0], -1, -1, -1)
            else:
                raise ValueError("Depth and albedo batch sizes are incompatible.")

        batch = depth_batched.shape[0]
        if mask is None:
            mask = create_face_mask(depth_batched.shape[-2], depth_batched.shape[-1], device=depth.device, dtype=depth.dtype)
        mask_batched = _broadcast_mask(mask, batch)

        if view is not None:
            warped_depth = _warp_image(depth_batched, depth_batched, view)
            warped_albedo = _warp_image(albedo_batched, depth_batched, view)
        else:
            warped_depth = depth_batched
            warped_albedo = albedo_batched

        normals = depth_to_normals(warped_depth)
        shading = spherical_harmonics_shading(normals, lighting_coefficients)
        if shading.ndim == 3:
            shading = shading.unsqueeze(0)
        if warped_albedo.shape != shading.shape:
            shading = shading.expand_as(warped_albedo)

        image = warped_albedo * shading
        if residual is not None:
            residual_batched, _ = _ensure_batched_image(residual)
            if residual_batched.shape != image.shape:
                residual_batched = residual_batched.expand_as(image)
            image = image + residual_batched

        if self.use_soft_mask:
            image = image * mask_batched + self.background_color.to(device=image.device, dtype=image.dtype) * (1.0 - mask_batched)

        if self.clamp_output:
            image = image.clamp(0.0, 1.0)
            shading = shading.clamp_min(0.0)

        result = RenderResult(
            image=image.squeeze(0) if albedo_squeezed and depth_squeezed else image,
            shading=shading.squeeze(0) if albedo_squeezed and depth_squeezed else shading,
            normals=normals.squeeze(0) if depth_squeezed else normals,
            depth=warped_depth.squeeze(0) if depth_squeezed else warped_depth,
            mask=mask_batched.squeeze(0) if mask_batched.shape[0] == 1 else mask_batched,
            warped_albedo=warped_albedo.squeeze(0) if albedo_squeezed else warped_albedo,
        )
        return result

    @torch.no_grad()
    def render_preset(
        self,
        depth: torch.Tensor,
        albedo: torch.Tensor,
        lighting_coefficients: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        view: ViewTransform | None = None,
    ) -> RenderResult:
        return self.forward(depth, albedo, lighting_coefficients, mask=mask, view=view)

