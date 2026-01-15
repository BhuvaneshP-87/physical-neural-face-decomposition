"""Spherical-harmonics lighting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(slots=True)
class LightingPreset:
    """A named set of SH lighting coefficients."""

    name: str
    coefficients: torch.Tensor
    description: str = ""


def spherical_harmonics_basis(normals: torch.Tensor) -> torch.Tensor:
    """Evaluate the 2nd-order SH basis for a normal map.

    Returns a tensor shaped [B, 9, H, W] or [9, H, W] for unbatched input.
    """

    squeeze_output = normals.ndim == 3
    if squeeze_output:
        normals = normals.unsqueeze(0)
    if normals.ndim != 4 or normals.shape[1] != 3:
        raise ValueError("Normals must be [B, 3, H, W] or [3, H, W].")

    normals = F.normalize(normals, dim=1, eps=1e-6)
    x, y, z = normals[:, 0:1], normals[:, 1:2], normals[:, 2:3]

    basis = torch.cat(
        [
            torch.full_like(x, 0.282095),
            0.488603 * y,
            0.488603 * z,
            0.488603 * x,
            1.092548 * x * y,
            1.092548 * y * z,
            0.315392 * (3.0 * z.square() - 1.0),
            1.092548 * x * z,
            0.546274 * (x.square() - y.square()),
        ],
        dim=1,
    )

    if squeeze_output:
        return basis.squeeze(0)
    return basis


def _broadcast_coefficients(coefficients: torch.Tensor, batch: int) -> torch.Tensor:
    if coefficients.ndim == 2:
        return coefficients.unsqueeze(0).expand(batch, -1, -1)
    if coefficients.ndim == 3:
        if coefficients.shape[0] != batch:
            raise ValueError("Per-sample lighting coefficients must match the batch size.")
        return coefficients
    raise ValueError("Lighting coefficients must have shape [3, 9] or [B, 3, 9].")


def spherical_harmonics_shading(
    normals: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    clamp: bool = True,
) -> torch.Tensor:
    """Compute RGB shading from normals and SH coefficients."""

    squeeze_output = normals.ndim == 3
    if squeeze_output:
        normals = normals.unsqueeze(0)

    basis = spherical_harmonics_basis(normals)
    if basis.ndim != 4:
        raise RuntimeError("Unexpected basis shape.")

    coeff = _broadcast_coefficients(coefficients, normals.shape[0])
    shading = torch.einsum("bcn,bnhw->bchw", coeff, basis)
    if clamp:
        shading = shading.clamp_min(0.0)

    if squeeze_output:
        return shading.squeeze(0)
    return shading


def make_lighting_preset(
    name: str,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> LightingPreset:
    """Create a convenient relighting preset."""

    dtype = dtype or torch.float32
    coefficients = torch.zeros(3, 9, device=device, dtype=dtype)

    # Shared ambient term.
    coefficients[:, 0] = torch.tensor([0.65, 0.62, 0.60], device=device, dtype=dtype)
    coefficients[:, 2] = 0.18
    coefficients[:, 6] = 0.05

    preset_key = name.lower().strip()
    description = "Default front-lit appearance."

    if preset_key in {"front", "studio", "neutral"}:
        coefficients[:, 1] = 0.02
        coefficients[:, 3] = 0.04
        description = "Front-facing soft studio lighting."
    elif preset_key in {"side", "sidelight", "side-light"}:
        coefficients[:, 1] = 0.06
        coefficients[:, 3] = -0.16
        coefficients[:, 7] = 0.08
        description = "Directional side light for stronger facial contouring."
    elif preset_key in {"sunset", "warm", "warm-sunset"}:
        coefficients[:, 0] = torch.tensor([0.78, 0.58, 0.40], device=device, dtype=dtype)
        coefficients[:, 1] = 0.05
        coefficients[:, 3] = 0.11
        coefficients[:, 7] = 0.03
        description = "Warm sunset-style illumination."
    elif preset_key in {"colored", "color", "rgb"}:
        coefficients[:, 0] = torch.tensor([0.48, 0.62, 0.80], device=device, dtype=dtype)
        coefficients[:, 1] = 0.03
        coefficients[:, 3] = 0.06
        coefficients[:, 4] = 0.04
        description = "Tinted illumination with a color cast."
    else:
        description = f"Generic lighting preset '{name}'."

    return LightingPreset(name=name, coefficients=coefficients, description=description)

