"""Optimization loop for inverse rendering and residual appearance recovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.optim import Adam

from ..config import OptimizationConfig
from ..models.face_prior import FaceState, create_initial_face_state
from ..models.residual_net import ResidualAppearanceNet, ResidualNetOutput
from ..renderer.geometry import ViewTransform, create_face_mask
from ..renderer.lighting import make_lighting_preset
from ..renderer.torch_renderer import RenderResult, TorchFaceRenderer
from .losses import PerceptualLoss, compute_inverse_rendering_losses


@dataclass(slots=True)
class OptimizationHistory:
    """Scalar losses and occasional image snapshots from optimization."""

    total: list[float] = field(default_factory=list)
    reconstruction: list[float] = field(default_factory=list)
    perceptual: list[float] = field(default_factory=list)
    lighting_regularization: list[float] = field(default_factory=list)
    albedo_smoothness: list[float] = field(default_factory=list)
    depth_smoothness: list[float] = field(default_factory=list)
    residual_penalty: list[float] = field(default_factory=list)
    snapshots: list[dict[str, torch.Tensor]] = field(default_factory=list)


@dataclass(slots=True)
class InverseRenderingResult:
    """Final output from the inverse renderer."""

    state: FaceState
    physical: RenderResult
    refined: RenderResult | None
    history: OptimizationHistory


class InverseRenderer:
    """Jointly optimize geometry, albedo, and lighting from a single face image."""

    def __init__(
        self,
        renderer: TorchFaceRenderer | None = None,
        config: OptimizationConfig | None = None,
        perceptual_loss: PerceptualLoss | None = None,
    ) -> None:
        self.renderer = renderer or TorchFaceRenderer()
        self.config = config or OptimizationConfig()
        self.perceptual_loss = perceptual_loss or PerceptualLoss(use_vgg=False)

    @staticmethod
    def _ensure_batched_image(image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            return image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError("Target image must be [3, H, W] or [B, 3, H, W].")
        return image

    def _initial_state(self, target: torch.Tensor, mask: torch.Tensor | None = None) -> FaceState:
        return create_initial_face_state(target, mask=mask, device=target.device)

    def _prepare_optimizer(
        self,
        depth: torch.nn.Parameter,
        albedo: torch.nn.Parameter,
        lighting: torch.nn.Parameter,
        residual_model: nn.Module | None,
    ) -> Adam:
        parameter_groups: list[dict[str, Any]] = [
            {"params": [depth], "lr": self.config.lr_geometry},
            {"params": [albedo], "lr": self.config.lr_albedo},
            {"params": [lighting], "lr": self.config.lr_lighting},
        ]
        if residual_model is not None:
            parameter_groups.append({"params": residual_model.parameters(), "lr": self.config.lr_residual})
        return Adam(parameter_groups)

    def optimize(
        self,
        target_image: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        num_iterations: int | None = None,
        residual_model: ResidualAppearanceNet | None = None,
        view: ViewTransform | None = None,
        snapshot_every: int | None = None,
        initial_state: FaceState | None = None,
    ) -> InverseRenderingResult:
        """Run inverse rendering against a target face image."""

        target = self._ensure_batched_image(target_image)
        if target.shape[1] != 3:
            raise ValueError("Target image must have three channels.")

        batch, _, height, width = target.shape
        mask = mask.to(target.device) if mask is not None else create_face_mask(height, width, device=target.device, dtype=target.dtype)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and batch > 1:
            mask = mask.expand(batch, -1, -1, -1)

        state = initial_state.detach() if initial_state is not None else self._initial_state(target, mask=mask)
        state = FaceState(
            depth=state.depth.to(target.device, dtype=target.dtype),
            albedo=state.albedo.to(target.device, dtype=target.dtype),
            lighting=state.lighting.to(target.device, dtype=target.dtype),
            residual=None if state.residual is None else state.residual.to(target.device, dtype=target.dtype),
        )
        depth = torch.nn.Parameter(state.depth.clone())
        albedo = torch.nn.Parameter(state.albedo.clone())
        lighting = torch.nn.Parameter(state.lighting.clone())

        if residual_model is None and self.config.use_residual_model:
            residual_model = ResidualAppearanceNet(in_channels=10).to(target.device, dtype=target.dtype)
        if residual_model is not None:
            residual_model = residual_model.to(target.device, dtype=target.dtype)

        optimizer = self._prepare_optimizer(depth, albedo, lighting, residual_model)
        num_iterations = num_iterations or self.config.iterations
        snapshot_every = snapshot_every or self.config.snapshot_every
        history = OptimizationHistory()

        for step in range(num_iterations):
            optimizer.zero_grad(set_to_none=True)

            physical = self.renderer(depth, albedo, lighting, mask=mask, view=view)
            refined_image = physical.image
            residual_output: ResidualNetOutput | None = None

            if residual_model is not None:
                normals = physical.normals if physical.normals.ndim == 4 else physical.normals.unsqueeze(0)
                residual_output = residual_model(
                    physical.image if physical.image.ndim == 4 else physical.image.unsqueeze(0),
                    albedo=albedo,
                    normals=normals,
                    depth=depth,
                )
                refined_image = residual_output.refined

            losses = compute_inverse_rendering_losses(
                refined_image,
                target,
                mask=mask,
                lighting=lighting,
                albedo=albedo,
                depth=depth,
                residual=None if residual_output is None else residual_output.residual,
                perceptual_loss=self.perceptual_loss,
            )

            total = (
                self.config.reconstruction_weight * losses.reconstruction
                + self.config.perceptual_weight * losses.perceptual
                + self.config.lighting_regularization_weight * losses.lighting_regularization
                + self.config.albedo_smoothness_weight * losses.albedo_smoothness
                + self.config.depth_smoothness_weight * losses.depth_smoothness
                + self.config.residual_weight * losses.residual_penalty
            )
            total.backward()
            optimizer.step()

            with torch.no_grad():
                albedo.clamp_(0.0, 1.0)

            history.total.append(float(total.detach().cpu()))
            history.reconstruction.append(float(losses.reconstruction.detach().cpu()))
            history.perceptual.append(float(losses.perceptual.detach().cpu()))
            history.lighting_regularization.append(float(losses.lighting_regularization.detach().cpu()))
            history.albedo_smoothness.append(float(losses.albedo_smoothness.detach().cpu()))
            history.depth_smoothness.append(float(losses.depth_smoothness.detach().cpu()))
            history.residual_penalty.append(float(losses.residual_penalty.detach().cpu()))

            if step == 0 or (step + 1) % snapshot_every == 0 or step == num_iterations - 1:
                history.snapshots.append(
                    {
                        "physical": physical.image.detach().cpu(),
                        "refined": refined_image.detach().cpu(),
                        "target": target.detach().cpu(),
                    }
                )

        final_state = FaceState(depth=depth.detach(), albedo=albedo.detach(), lighting=lighting.detach())
        final_physical = self.renderer(final_state.depth, final_state.albedo, final_state.lighting, mask=mask, view=view)

        final_refined = None
        if residual_model is not None:
            normals = final_physical.normals if final_physical.normals.ndim == 4 else final_physical.normals.unsqueeze(0)
            residual_output = residual_model(
                final_physical.image if final_physical.image.ndim == 4 else final_physical.image.unsqueeze(0),
                albedo=final_state.albedo,
                normals=normals,
                depth=final_state.depth,
            )
            final_refined = RenderResult(
                image=residual_output.refined,
                shading=final_physical.shading,
                normals=final_physical.normals,
                depth=final_physical.depth,
                mask=final_physical.mask,
                warped_albedo=final_physical.warped_albedo,
            )

        return InverseRenderingResult(
            state=final_state,
            physical=final_physical,
            refined=final_refined,
            history=history,
        )

    @torch.no_grad()
    def relight(
        self,
        state: FaceState,
        *,
        presets: list[str] | None = None,
        mask: torch.Tensor | None = None,
        view: ViewTransform | None = None,
    ) -> dict[str, RenderResult]:
        presets = presets or ["front", "side", "sunset", "colored"]
        results: dict[str, RenderResult] = {}
        for preset_name in presets:
            lighting = make_lighting_preset(preset_name, device=state.depth.device, dtype=state.depth.dtype).coefficients
            results[preset_name] = self.renderer(state.depth, state.albedo, lighting, mask=mask, view=view)
        return results

    @torch.no_grad()
    def novel_view_sweep(
        self,
        state: FaceState,
        *,
        yaw_values: list[float],
        pitch_values: list[float],
        lighting: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, RenderResult]:
        lighting = lighting if lighting is not None else state.lighting
        outputs: dict[str, RenderResult] = {}
        for yaw in yaw_values:
            for pitch in pitch_values:
                view = ViewTransform(yaw_degrees=yaw, pitch_degrees=pitch)
                key = f"yaw_{yaw:+.1f}_pitch_{pitch:+.1f}"
                outputs[key] = self.renderer(state.depth, state.albedo, lighting, mask=mask, view=view)
        return outputs
