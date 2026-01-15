"""Lightweight priors and state containers for facial inverse rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..renderer.lighting import make_lighting_preset


@dataclass(slots=True)
class FaceState:
    """Current estimate of geometry, albedo, lighting, and optional residual."""

    depth: torch.Tensor
    albedo: torch.Tensor
    lighting: torch.Tensor
    residual: torch.Tensor | None = None

    def detach(self) -> "FaceState":
        return FaceState(
            depth=self.depth.detach(),
            albedo=self.albedo.detach(),
            lighting=self.lighting.detach(),
            residual=None if self.residual is None else self.residual.detach(),
        )


def create_initial_face_state(
    image: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    device: torch.device | None = None,
    prior: Any | None = None,
) -> FaceState:
    """Create a stable initialization from a target face image."""

    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected image tensor shaped [B, 3, H, W] or [3, H, W].")

    device = device or image.device
    batch, _, height, width = image.shape
    if prior is not None:
        prior_depth = getattr(prior, "depth", None)
        prior_albedo = getattr(prior, "albedo", None)
        prior_lighting = getattr(prior, "lighting", None)
    else:
        prior_depth = prior_albedo = prior_lighting = None

    lighting = prior_lighting
    if lighting is None:
        lighting = make_lighting_preset("front", device=device, dtype=image.dtype).coefficients
        lighting = lighting.unsqueeze(0).expand(batch, -1, -1).contiguous()
    elif lighting.ndim == 2:
        lighting = lighting.unsqueeze(0)
    lighting = lighting.to(device=device, dtype=image.dtype)

    if mask is None:
        mask = torch.ones(batch, 1, height, width, device=device, dtype=image.dtype)
    elif mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)

    smoothed = image * mask + 0.5 * (1.0 - mask)
    depth = prior_depth
    if depth is None:
        depth = torch.zeros(batch, 1, height, width, device=device, dtype=image.dtype)
        depth = depth + 0.02 * torch.randn_like(depth)
    elif depth.ndim == 3:
        depth = depth.unsqueeze(0)
    depth = depth.to(device=device, dtype=image.dtype)
    if depth.shape[0] == 1 and batch > 1:
        depth = depth.expand(batch, -1, -1, -1).contiguous()

    albedo = prior_albedo
    if albedo is None:
        albedo = smoothed.clamp(0.0, 1.0)
    elif albedo.ndim == 3:
        albedo = albedo.unsqueeze(0)
    albedo = albedo.to(device=device, dtype=image.dtype).clamp(0.0, 1.0)
    if albedo.shape[0] == 1 and batch > 1:
        albedo = albedo.expand(batch, -1, -1, -1).contiguous()

    if lighting.shape[0] == 1 and batch > 1:
        lighting = lighting.expand(batch, -1, -1).contiguous()

    return FaceState(depth=depth, albedo=albedo, lighting=lighting)
