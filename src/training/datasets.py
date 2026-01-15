"""Face datasets for training and synthetic self-supervision."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from ..data.preprocessing import (
    FacePreprocessConfig,
    FacePreprocessor,
)
from ..models.face_priors import build_face_prior_backend
from ..renderer.geometry import create_face_mask
from ..renderer.lighting import make_lighting_preset
from ..renderer.torch_renderer import TorchFaceRenderer


class FaceImageFolderDataset(Dataset):
    """Dataset that reads a directory of face images."""

    def __init__(
        self,
        image_dir: str | Path,
        *,
        recursive: bool = True,
        preprocessor: FacePreprocessor | None = None,
        preprocess_config: FacePreprocessConfig | None = None,
        face_prior_backend: Any | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        patterns = ["**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.bmp"] if recursive else ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
        self.paths = sorted({path for pattern in patterns for path in self.image_dir.glob(pattern)})
        if not self.paths:
            raise FileNotFoundError(f"No image files found in {self.image_dir}.")
        self.preprocessor = preprocessor or FacePreprocessor(preprocess_config)
        self.face_prior_backend = face_prior_backend or build_face_prior_backend("synthetic")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        preprocessed = self.preprocessor(self.paths[index])
        prior = self.face_prior_backend.estimate(preprocessed.image, mask=preprocessed.mask)
        state = prior.to_face_state(preprocessed.image, mask=preprocessed.mask)
        sample = {
            "image": preprocessed.image,
            "mask": preprocessed.mask.squeeze(0),
            "depth": state.depth.squeeze(0),
            "albedo": state.albedo.squeeze(0),
            "lighting": state.lighting.squeeze(0),
            "bbox": None if preprocessed.bbox is None else preprocessed.bbox.as_tuple(),
            "original_size": preprocessed.original_size,
            "crop_size": preprocessed.crop_size,
            "source_path": str(self.paths[index]),
            "metadata": {
                **preprocessed.metadata,
                "prior": prior.to_dict(),
            },
        }
        return sample


class SyntheticFaceDataset(Dataset):
    """Torch-only synthetic face dataset for quick experiments and demos."""

    def __init__(
        self,
        *,
        length: int = 64,
        image_size: tuple[int, int] = (256, 256),
        seed: int = 1234,
    ) -> None:
        self.length = length
        self.image_size = image_size
        self.seed = seed
        self.renderer = TorchFaceRenderer()

    def __len__(self) -> int:
        return self.length

    def _generator(self, index: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(self.seed + index)
        return generator

    def _smooth_noise(self, shape: tuple[int, ...], generator: torch.Generator, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coarse = torch.rand(shape, generator=generator, device=device, dtype=dtype)
        noise = F.interpolate(coarse, size=self.image_size, mode="bilinear", align_corners=False)
        return noise

    def _make_depth(self, generator: torch.Generator, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        height, width = self.image_size
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        face = torch.exp(-((grid_x / 0.82).square() + (grid_y / 0.96).square()) * 1.75)
        nose = 0.12 * torch.exp(-((grid_x / 0.18).square() + ((grid_y + 0.05) / 0.24).square()) * 0.75)
        cheeks = 0.03 * torch.exp(-(((grid_x - 0.32) / 0.18).square() + ((grid_y + 0.03) / 0.18).square()) * 0.9)
        cheeks = cheeks + 0.03 * torch.exp(-(((grid_x + 0.32) / 0.18).square() + ((grid_y + 0.03) / 0.18).square()) * 0.9)
        noise = 0.02 * self._smooth_noise((1, 1, 24, 24), generator, device, dtype).squeeze()
        depth = 0.18 * face + nose + cheeks + noise
        return depth.unsqueeze(0)

    def _make_albedo(self, generator: torch.Generator, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        height, width = self.image_size
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        base_color = torch.tensor([0.78, 0.60, 0.50], device=device, dtype=dtype).view(3, 1, 1)
        cheek_blush = 0.08 * torch.exp(-(((grid_x - 0.34) / 0.25).square() + ((grid_y + 0.08) / 0.18).square()) * 1.2)
        cheek_blush = cheek_blush + 0.08 * torch.exp(-(((grid_x + 0.34) / 0.25).square() + ((grid_y + 0.08) / 0.18).square()) * 1.2)
        lip_color = 0.06 * torch.exp(-((grid_x / 0.22).square() + ((grid_y + 0.47) / 0.09).square()) * 1.5)
        eye_shadow = 0.05 * torch.exp(-((grid_x / 0.62).square() + ((grid_y - 0.2) / 0.16).square()) * 2.5)
        detail_noise = 0.04 * self._smooth_noise((1, 3, 32, 32), generator, device, dtype).squeeze(0)
        albedo = base_color + cheek_blush.unsqueeze(0) * torch.tensor([1.0, 0.15, 0.1], device=device, dtype=dtype).view(3, 1, 1)
        albedo = albedo + lip_color.unsqueeze(0) * torch.tensor([0.25, 0.05, 0.05], device=device, dtype=dtype).view(3, 1, 1)
        albedo = albedo - eye_shadow.unsqueeze(0) * 0.04
        albedo = (albedo + detail_noise).clamp(0.0, 1.0)
        return albedo

    def _make_residual(self, generator: torch.Generator, device: torch.device, dtype: torch.dtype, mask: torch.Tensor) -> torch.Tensor:
        high_freq = torch.rand((1, 3, *self.image_size), generator=generator, device=device, dtype=dtype)
        high_freq = high_freq - F.avg_pool2d(high_freq, kernel_size=7, stride=1, padding=3)
        tint = torch.tensor([0.02, 0.015, 0.012], device=device, dtype=dtype).view(1, 3, 1, 1)
        return 0.06 * high_freq * mask + tint * mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        device = torch.device("cpu")
        dtype = torch.float32
        generator = self._generator(index)
        depth = self._make_depth(generator, device, dtype)
        albedo = self._make_albedo(generator, device, dtype)
        mask = create_face_mask(self.image_size[0], self.image_size[1], device=device, dtype=dtype)
        lighting_name = ["front", "side", "sunset", "colored"][index % 4]
        lighting = make_lighting_preset(lighting_name, device=device, dtype=dtype).coefficients.unsqueeze(0)
        physical = self.renderer(depth, albedo, lighting, mask=mask)
        residual = self._make_residual(generator, device, dtype, mask)
        image = (physical.image + residual).clamp(0.0, 1.0)
        return {
            "image": image.squeeze(0),
            "physical": physical.image.squeeze(0),
            "depth": depth.squeeze(0),
            "albedo": albedo.squeeze(0),
            "lighting": lighting.squeeze(0),
            "mask": mask.squeeze(0),
            "residual": residual.squeeze(0),
            "preset": lighting_name,
        }
