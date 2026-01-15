"""Visualization and export helpers for reconstructions and relighting demos."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.nn import functional as F


def _ensure_image_tensor(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        return image
    if image.ndim == 4 and image.shape[0] == 1:
        return image.squeeze(0)
    raise ValueError("Expected an image tensor shaped [C, H, W] or [1, C, H, W].")


def make_image_grid(images: Sequence[torch.Tensor], *, nrow: int = 4, padding: int = 2, pad_value: float = 0.0) -> torch.Tensor:
    """Tile a sequence of image tensors into a single grid tensor."""

    images = [_ensure_image_tensor(image) for image in images]
    if not images:
        raise ValueError("At least one image is required.")

    channels, height, width = images[0].shape
    for image in images:
        if image.shape != (channels, height, width):
            raise ValueError("All images in the grid must share the same shape.")

    ncol = min(nrow, len(images))
    nrow = (len(images) + ncol - 1) // ncol

    grid_height = nrow * height + padding * (nrow - 1)
    grid_width = ncol * width + padding * (ncol - 1)
    grid = torch.full((channels, grid_height, grid_width), pad_value, device=images[0].device, dtype=images[0].dtype)

    for index, image in enumerate(images):
        row = index // ncol
        col = index % ncol
        top = row * (height + padding)
        left = col * (width + padding)
        grid[:, top : top + height, left : left + width] = image

    return grid.clamp(0.0, 1.0)


def _to_uint8(image: torch.Tensor) -> "numpy.ndarray":  # type: ignore[name-defined]
    import numpy as np

    image = _ensure_image_tensor(image).detach().cpu().clamp(0.0, 1.0)
    array = image.permute(1, 2, 0).mul(255.0).round().to(torch.uint8).numpy()
    return array if array.ndim == 3 else np.repeat(array[..., None], 3, axis=-1)


def save_image_grid(path: str | Path, images: Sequence[torch.Tensor], *, nrow: int = 4) -> Path:
    """Save a comparison grid to disk using OpenCV, imageio, or Pillow as available."""

    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_image_grid(images, nrow=nrow)
    array = _to_uint8(grid)

    try:  # pragma: no cover - optional dependency
        import cv2

        bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)
        return path
    except Exception:
        pass

    try:  # pragma: no cover - optional dependency
        from PIL import Image

        Image.fromarray(array).save(path)
        return path
    except Exception:
        pass

    try:  # pragma: no cover - optional dependency
        import imageio.v2 as imageio

        imageio.imwrite(path, array)
        return path
    except Exception:
        pass

    np.save(path.with_suffix(".npy"), array)
    return path.with_suffix(".npy")


def save_gif(path: str | Path, frames: Iterable[torch.Tensor], *, fps: int = 10) -> Path:
    """Save a sequence of frames as a GIF."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [_to_uint8(frame) for frame in frames]

    try:  # pragma: no cover - optional dependency
        import imageio.v2 as imageio

        imageio.mimsave(path, arrays, duration=1.0 / max(1, fps))
        return path
    except Exception:
        import numpy as np

        np.save(path.with_suffix(".npy"), np.stack(arrays, axis=0))
        return path.with_suffix(".npy")

