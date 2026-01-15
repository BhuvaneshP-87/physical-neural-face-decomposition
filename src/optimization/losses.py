"""Loss functions for inverse rendering and residual appearance learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class LossBreakdown:
    reconstruction: torch.Tensor
    perceptual: torch.Tensor
    lighting_regularization: torch.Tensor
    albedo_smoothness: torch.Tensor
    depth_smoothness: torch.Tensor
    residual_penalty: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return (
            self.reconstruction
            + self.perceptual
            + self.lighting_regularization
            + self.albedo_smoothness
            + self.depth_smoothness
            + self.residual_penalty
        )


def _ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected a 3D or 4D tensor, got {tensor.ndim}D.")
    return tensor


def masked_l1_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    prediction = _ensure_4d(prediction)
    target = _ensure_4d(target)
    diff = (prediction - target).abs()
    if mask is not None:
        mask = _ensure_4d(mask)
        if mask.shape[1] == 1 and diff.shape[1] != 1:
            mask = mask.expand(-1, diff.shape[1], -1, -1)
        diff = diff * mask
        denominator = mask.sum().clamp_min(1.0)
    else:
        denominator = torch.tensor(diff.numel() / diff.shape[1], device=diff.device, dtype=diff.dtype)
    return diff.sum() / denominator


def total_variation_loss(tensor: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    tensor = _ensure_4d(tensor)
    horizontal = tensor[..., :, 1:] - tensor[..., :, :-1]
    vertical = tensor[..., 1:, :] - tensor[..., :-1, :]
    loss = horizontal.abs().mean() + vertical.abs().mean()
    if mask is not None:
        mask = _ensure_4d(mask)
        loss = loss * mask.mean().clamp_min(1e-6)
    return loss


def lighting_regularization_loss(lighting: torch.Tensor) -> torch.Tensor:
    """Encourage low-energy higher-order lighting coefficients."""

    if lighting.ndim == 2:
        lighting = lighting.unsqueeze(0)
    if lighting.ndim != 3 or lighting.shape[-1] != 9:
        raise ValueError("Lighting tensor must be [B, 3, 9] or [3, 9].")
    ambient = lighting[..., :1]
    higher_order = lighting[..., 1:]
    return 1e-3 * higher_order.square().mean() + 5e-4 * ambient.square().mean()


def albedo_smoothness_loss(albedo: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return total_variation_loss(albedo, mask=mask)


def depth_smoothness_loss(depth: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return total_variation_loss(depth, mask=mask)


class _FixedPerceptualFeatures(nn.Module):
    """Fallback feature extractor used when torchvision VGG is unavailable."""

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        blur = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 16.0
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("blur", blur)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        image = _ensure_4d(image)
        gray = image.mean(dim=1, keepdim=True)
        smoothed = F.conv2d(gray, self.blur, padding=1)
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        pooled = F.avg_pool2d(gray, kernel_size=2, stride=2)
        return [gray, smoothed, grad_x, grad_y, pooled]


class PerceptualLoss(nn.Module):
    """Perceptual loss with a VGG backend when available and a fixed fallback otherwise."""

    def __init__(self, *, use_vgg: bool = True) -> None:
        super().__init__()
        self.use_vgg = use_vgg
        self.fallback = _FixedPerceptualFeatures()
        self.vgg = None

        if use_vgg:
            try:  # pragma: no cover - optional dependency
                from torchvision.models import VGG16_Weights, vgg16

                backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES)
                features = list(backbone.features.children())[:16]
                self.vgg = nn.Sequential(*features).eval()
                for parameter in self.vgg.parameters():
                    parameter.requires_grad_(False)
            except Exception:
                self.vgg = None

    def _extract_features(self, image: torch.Tensor) -> list[torch.Tensor]:
        if self.vgg is not None:
            x = image
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            x = (x - mean) / std
            features = []
            current = x
            for layer in self.vgg:
                current = layer(current)
                if isinstance(layer, nn.ReLU):
                    features.append(current)
                if len(features) >= 4:
                    break
            return features
        return self.fallback(image)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = _ensure_4d(prediction)
        target = _ensure_4d(target)
        pred_features = self._extract_features(prediction)
        target_features = self._extract_features(target)
        loss = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
        for pred_feature, target_feature in zip(pred_features, target_features):
            loss = loss + (pred_feature - target_feature).abs().mean()
        return loss


def compute_inverse_rendering_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    lighting: torch.Tensor | None = None,
    albedo: torch.Tensor | None = None,
    depth: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    perceptual_loss: PerceptualLoss | None = None,
) -> LossBreakdown:
    """Bundle the common losses used by the inverse rendering pipeline."""

    reconstruction = masked_l1_loss(prediction, target, mask=mask)
    perceptual = torch.zeros_like(reconstruction)
    if perceptual_loss is not None:
        perceptual = perceptual_loss(prediction, target)
    lighting_reg = lighting_regularization_loss(lighting) if lighting is not None else torch.zeros_like(reconstruction)
    albedo_reg = albedo_smoothness_loss(albedo, mask=mask) if albedo is not None else torch.zeros_like(reconstruction)
    depth_reg = depth_smoothness_loss(depth, mask=mask) if depth is not None else torch.zeros_like(reconstruction)
    residual_reg = residual.abs().mean() if residual is not None else torch.zeros_like(reconstruction)
    return LossBreakdown(
        reconstruction=reconstruction,
        perceptual=perceptual,
        lighting_regularization=lighting_reg,
        albedo_smoothness=albedo_reg,
        depth_smoothness=depth_reg,
        residual_penalty=residual_reg,
    )

