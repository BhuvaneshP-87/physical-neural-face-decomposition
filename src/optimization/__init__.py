"""Inverse rendering losses and optimization loops."""

from .inverse_rendering import InverseRenderingResult, InverseRenderer, OptimizationHistory
from .mitsuba_inverse import (
    MitsubaInverseConfig,
    MitsubaInverseSummary,
    mitsuba_drjit_available,
    run_mitsuba_inverse_experiment,
)
from .losses import (
    PerceptualLoss,
    albedo_smoothness_loss,
    compute_inverse_rendering_losses,
    lighting_regularization_loss,
    masked_l1_loss,
    total_variation_loss,
)

__all__ = [
    "InverseRenderingResult",
    "InverseRenderer",
    "OptimizationHistory",
    "MitsubaInverseConfig",
    "MitsubaInverseSummary",
    "mitsuba_drjit_available",
    "run_mitsuba_inverse_experiment",
    "PerceptualLoss",
    "albedo_smoothness_loss",
    "compute_inverse_rendering_losses",
    "lighting_regularization_loss",
    "masked_l1_loss",
    "total_variation_loss",
]
