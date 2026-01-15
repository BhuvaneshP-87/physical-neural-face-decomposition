"""Mitsuba/Dr.Jit inverse-rendering experiments.

This module is intentionally optional. It imports Mitsuba and Dr.Jit lazily so
the rest of the project remains usable without the renderer installed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..renderer.mitsuba_scene import MitsubaSceneConfig, MitsubaSceneTranslator


@dataclass(slots=True)
class MitsubaInverseConfig:
    """Configuration for a Mitsuba/Dr.Jit inverse-rendering run."""

    output_dir: Path = field(default_factory=lambda: Path("outputs/mitsuba_inverse"))
    iterations: int = 32
    spp: int = 16
    evaluation_spp: int = 32
    learning_rate: float = 0.03
    image_size: tuple[int, int] = (128, 128)
    optimize_geometry: bool = False
    optimize_albedo: bool = True
    optimize_light: bool = True
    variant: str = "llvm_ad_rgb"
    scene_name: str = "mitsuba_face_inverse"
    optimization_seed: int = 7
    target_seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


@dataclass(slots=True)
class MitsubaInverseSummary:
    """Result metadata for a Mitsuba inverse-rendering experiment."""

    status: str
    output_dir: Path
    message: str
    losses: list[float] = field(default_factory=list)
    parameter_keys: list[str] = field(default_factory=list)
    optimized_keys: list[str] = field(default_factory=list)
    artifacts: dict[str, Path] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output_dir": str(self.output_dir),
            "message": self.message,
            "losses": self.losses,
            "parameter_keys": self.parameter_keys,
            "optimized_keys": self.optimized_keys,
            "artifacts": {key: str(value) for key, value in self.artifacts.items()},
            "diagnostics": self.diagnostics,
        }


def mitsuba_drjit_available() -> bool:
    """Return whether Mitsuba and Dr.Jit can be imported."""

    try:
        import drjit  # noqa: F401
        import mitsuba  # noqa: F401
    except Exception:
        return False
    return True


def _require_mitsuba(variant: str):
    try:
        import drjit as dr
        import mitsuba as mi
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Mitsuba 3 and Dr.Jit are required for this experiment.") from exc
    mi.set_variant(variant)
    return mi, dr


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _save_mitsuba_bitmap(path: Path, image: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    import mitsuba as mi

    bitmap = mi.Bitmap(image)
    bitmap.write(str(path))
    return path


def _torch_face_prior(size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = size
    y_coords = torch.linspace(-1.0, 1.0, height)
    x_coords = torch.linspace(-1.0, 1.0, width)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    mask = (((grid_x / 0.78) ** 2 + (grid_y / 0.95) ** 2) <= 1.0).float().unsqueeze(0).unsqueeze(0)
    face = torch.exp(-((grid_x / 0.78).square() + (grid_y / 0.96).square()) * 1.25)
    nose = 0.25 * torch.exp(-((grid_x / 0.16).square() + ((grid_y + 0.04) / 0.24).square()) * 1.1)
    brow = 0.04 * torch.exp(-((grid_x / 0.55).square() + ((grid_y - 0.22) / 0.12).square()) * 1.8)
    depth = (face + nose + brow).unsqueeze(0).unsqueeze(0) * mask
    albedo = torch.tensor([0.72, 0.55, 0.46]).view(1, 3, 1, 1).expand(1, 3, height, width).contiguous()
    lighting = torch.zeros(1, 3, 9)
    lighting[:, :, 0] = 0.8
    lighting[:, :, 2] = 0.25
    return depth, albedo, lighting, mask


def _make_scene_bundle(config: MitsubaInverseConfig):
    depth, albedo, lighting, mask = _torch_face_prior(config.image_size)
    scene_config = MitsubaSceneConfig(
        output_dir=config.output_dir / "scene",
        samples_per_pixel=config.spp,
        film_resolution=config.image_size,
        sensor_fov_degrees=35.0,
        mesh_scale=1.0,
        depth_scale=0.22,
        export_closed_head_proxy=True,
        texture_gamma_correction=True,
        proxy_material_color=(0.55, 0.55, 0.52),
    )
    translator = MitsubaSceneTranslator(scene_config)
    bundle = translator.export(
        depth,
        albedo,
        lighting,
        mask=mask,
        output_dir=scene_config.output_dir,
        scene_name=config.scene_name,
    )
    return bundle


def _select_parameter_keys(params: Any, config: MitsubaInverseConfig) -> list[str]:
    keys = [str(key) for key in params.keys()]
    selected: list[str] = []
    if config.optimize_albedo:
        selected.extend(
            key for key in keys
            if "reflectance.data" in key.lower()
        )
    if config.optimize_light:
        selected.extend(
            key for key in keys
            if key.lower().endswith(".scale")
        )
    if config.optimize_geometry:
        selected.extend(
            key for key in keys
            if key.lower().endswith("vertex_positions")
        )
    deduped: list[str] = []
    for key in selected:
        if key not in deduped:
            deduped.append(key)
    return deduped


def _perturb_parameter(key: str, value: Any) -> Any:
    key_lower = key.lower()
    if "reflectance.data" in key_lower:
        return value * 0.72
    if key_lower.endswith(".scale"):
        return value * 0.55
    if key_lower.endswith("vertex_positions"):
        return value * 0.97
    return value


def _project_parameter(key: str, value: Any, dr: Any) -> Any:
    key_lower = key.lower()
    if "reflectance.data" in key_lower:
        return dr.clip(value, 0.02, 1.0)
    if key_lower.endswith(".scale"):
        return dr.maximum(value, 0.02)
    return value


def run_mitsuba_inverse_experiment(config: MitsubaInverseConfig | None = None) -> MitsubaInverseSummary:
    """Run a small Mitsuba/Dr.Jit inverse-rendering optimization.

    The experiment renders a canonical face-like mesh, perturbs selected scene
    parameters, then optimizes them against the target render using Dr.Jit
    gradients. It is designed as a portfolio-level proof that the project can
    traverse and optimize differentiable Mitsuba scene parameters.
    """

    config = config or MitsubaInverseConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not mitsuba_drjit_available():
        summary = MitsubaInverseSummary(
            status="skipped",
            output_dir=output_dir,
            message="Mitsuba 3 / Dr.Jit are not installed. Install with `pip install -e .[render]` to run this experiment.",
        )
        _write_json(output_dir / "mitsuba_inverse_summary.json", summary.to_dict())
        return summary

    mi, dr = _require_mitsuba(config.variant)
    bundle = _make_scene_bundle(config)
    scene = mi.load_file(str(bundle.scene_xml))
    params = mi.traverse(scene)
    parameter_keys = [str(key) for key in params.keys()]

    target = mi.render(scene, params, spp=config.evaluation_spp, seed=config.target_seed)
    _save_mitsuba_bitmap(output_dir / "target.exr", target)

    optimized_keys = _select_parameter_keys(params, config)
    if not optimized_keys:
        summary = MitsubaInverseSummary(
            status="failed",
            output_dir=output_dir,
            message="No differentiable Mitsuba parameters were found for optimization.",
            parameter_keys=parameter_keys,
            artifacts={"scene_xml": bundle.scene_xml},
        )
        _write_json(output_dir / "mitsuba_inverse_summary.json", summary.to_dict())
        return summary

    optimizer = mi.ad.Adam(lr=config.learning_rate)
    active_keys: list[str] = []
    for key in optimized_keys:
        value = params[key]
        try:
            optimizer[key] = _perturb_parameter(key, value)
            dr.enable_grad(optimizer[key])
            active_keys.append(key)
        except TypeError:
            continue

    params.update(optimizer)
    optimized_keys = active_keys
    if not optimized_keys:
        summary = MitsubaInverseSummary(
            status="failed",
            output_dir=output_dir,
            message="Candidate Mitsuba parameters were found, but none were differentiable optimizer variables.",
            parameter_keys=parameter_keys,
            artifacts={"scene_xml": bundle.scene_xml},
        )
        _write_json(output_dir / "mitsuba_inverse_summary.json", summary.to_dict())
        return summary
    initial = mi.render(scene, params, spp=config.evaluation_spp, seed=config.target_seed)
    _save_mitsuba_bitmap(output_dir / "initial.exr", initial)
    initial_eval_loss = dr.mean(dr.sqr(initial - target))

    losses: list[float] = []
    for iteration in range(config.iterations):
        image = mi.render(scene, params, spp=config.spp, seed=config.optimization_seed)
        loss = dr.mean(dr.sqr(image - target))
        dr.backward(loss)
        optimizer.step()
        for key in optimized_keys:
            optimizer[key] = _project_parameter(key, optimizer[key], dr)
        params.update(optimizer)
        losses.append(float(dr.ravel(loss)[0]))

    final = mi.render(scene, params, spp=config.evaluation_spp, seed=config.target_seed)
    final_eval_loss = dr.mean(dr.sqr(final - target))
    initial_loss_value = float(dr.ravel(initial_eval_loss)[0])
    final_loss_value = float(dr.ravel(final_eval_loss)[0])
    loss_reduction = initial_loss_value - final_loss_value
    relative_reduction = loss_reduction / max(initial_loss_value, 1e-12)
    artifacts = {
        "scene_xml": bundle.scene_xml,
        "target_exr": output_dir / "target.exr",
        "initial_exr": output_dir / "initial.exr",
        "final_exr": _save_mitsuba_bitmap(output_dir / "optimized.exr", final),
    }
    summary = MitsubaInverseSummary(
        status="completed",
        output_dir=output_dir,
        message="Mitsuba/Dr.Jit inverse-rendering optimization completed.",
        losses=losses,
        parameter_keys=parameter_keys,
        optimized_keys=optimized_keys,
        artifacts=artifacts,
        diagnostics={
            "initial_eval_mse": initial_loss_value,
            "final_eval_mse": final_loss_value,
            "absolute_mse_reduction": loss_reduction,
            "relative_mse_reduction": relative_reduction,
        },
    )
    _write_json(output_dir / "mitsuba_inverse_summary.json", summary.to_dict())
    _write_json(output_dir / "mitsuba_parameter_keys.json", {"parameter_keys": parameter_keys, "optimized_keys": optimized_keys})
    return summary
