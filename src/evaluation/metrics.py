"""Quantitative metrics used for reconstruction and relighting evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from ..optimization.losses import PerceptualLoss


@dataclass(slots=True)
class EvaluationMetrics:
    """Standard image-quality metrics."""

    psnr: float
    ssim: float
    lpips: float | None = None


def _ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError("Expected a 3D or 4D tensor.")
    return tensor


def psnr(prediction: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0) -> torch.Tensor:
    prediction = _ensure_4d(prediction)
    target = _ensure_4d(target)
    mse = F.mse_loss(prediction, target)
    return 20.0 * torch.log10(torch.tensor(data_range, device=prediction.device, dtype=prediction.dtype)) - 10.0 * torch.log10(mse.clamp_min(1e-12))


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    prediction = _ensure_4d(prediction)
    target = _ensure_4d(target)

    channel = prediction.shape[1]
    coords = torch.arange(window_size, device=prediction.device, dtype=prediction.dtype) - window_size // 2
    gaussian = torch.exp(-(coords.square()) / (2 * sigma * sigma))
    gaussian = gaussian / gaussian.sum()
    window_1d = gaussian.view(1, 1, 1, -1)
    window_2d = gaussian.view(1, 1, -1, 1) * window_1d
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()

    mu_x = F.conv2d(prediction, window, padding=window_size // 2, groups=channel)
    mu_y = F.conv2d(target, window, padding=window_size // 2, groups=channel)
    sigma_x = F.conv2d(prediction * prediction, window, padding=window_size // 2, groups=channel) - mu_x.square()
    sigma_y = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu_y.square()
    sigma_xy = F.conv2d(prediction * target, window, padding=window_size // 2, groups=channel) - mu_x * mu_y

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    score = numerator / denominator.clamp_min(1e-12)
    return score.mean()


def lpips_proxy(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """A perceptual distance proxy used when the LPIPS package is unavailable."""

    perceptual_loss = PerceptualLoss(use_vgg=True)
    return perceptual_loss(prediction, target)


def compute_evaluation_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    use_lpips: bool = False,
) -> EvaluationMetrics:
    psnr_value = float(psnr(prediction, target).detach().cpu())
    ssim_value = float(ssim(prediction, target).detach().cpu())
    lpips_value = float(lpips_proxy(prediction, target).detach().cpu()) if use_lpips else None
    return EvaluationMetrics(psnr=psnr_value, ssim=ssim_value, lpips=lpips_value)

