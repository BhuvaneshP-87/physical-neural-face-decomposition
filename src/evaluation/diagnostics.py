"""Research diagnostics for inverse-rendering experiments."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..models.face_prior import FaceState
from ..optimization.inverse_rendering import InverseRenderingResult, OptimizationHistory
from .metrics import compute_evaluation_metrics
from .visualization import save_gif, save_image_grid


@dataclass(slots=True)
class ComponentStatistics:
    """Summary statistics for an estimated scene component."""

    name: str
    shape: tuple[int, ...]
    minimum: float
    maximum: float
    mean: float
    std: float
    l1_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "std": self.std,
            "l1_mean": self.l1_mean,
        }


@dataclass(slots=True)
class DiagnosticSummary:
    """Files and scalar summaries produced for one experiment phase."""

    phase_name: str
    output_dir: Path
    metrics: dict[str, float]
    component_statistics: dict[str, ComponentStatistics] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "output_dir": str(self.output_dir),
            "metrics": self.metrics,
            "component_statistics": {
                key: value.to_dict() for key, value in self.component_statistics.items()
            },
            "artifacts": {key: str(value) for key, value in self.artifacts.items()},
        }


def _ensure_chw(tensor: Tensor) -> Tensor:
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        return tensor[0]
    if tensor.ndim == 3:
        return tensor
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    raise ValueError(f"Expected image-like tensor, got shape {tuple(tensor.shape)}.")


def _normalize_map(tensor: Tensor) -> Tensor:
    tensor = _ensure_chw(tensor).detach().cpu().float()
    min_value = tensor.amin(dim=(-1, -2), keepdim=True)
    max_value = tensor.amax(dim=(-1, -2), keepdim=True)
    return ((tensor - min_value) / (max_value - min_value).clamp_min(1e-6)).clamp(0.0, 1.0)


def _as_rgb(tensor: Tensor) -> Tensor:
    tensor = _ensure_chw(tensor).detach().cpu().float()
    if tensor.shape[0] == 1:
        return tensor.repeat(3, 1, 1).clamp(0.0, 1.0)
    if tensor.shape[0] == 2:
        return torch.cat((tensor, torch.zeros_like(tensor[:1])), dim=0).clamp(0.0, 1.0)
    return tensor[:3].clamp(0.0, 1.0)


def _normal_visualization(normals: Tensor) -> Tensor:
    normals = _ensure_chw(normals).detach().cpu().float()
    return ((normals[:3] + 1.0) * 0.5).clamp(0.0, 1.0)


def _error_heatmap(prediction: Tensor, target: Tensor) -> Tensor:
    prediction = _ensure_chw(prediction).detach().cpu().float()
    target = _ensure_chw(target).detach().cpu().float()
    error = (prediction - target).abs().mean(dim=0, keepdim=True)
    return _normalize_map(error).repeat(3, 1, 1)


def _tensor_stats(name: str, tensor: Tensor) -> ComponentStatistics:
    tensor = tensor.detach().cpu().float()
    return ComponentStatistics(
        name=name,
        shape=tuple(int(value) for value in tensor.shape),
        minimum=float(tensor.min()),
        maximum=float(tensor.max()),
        mean=float(tensor.mean()),
        std=float(tensor.std()),
        l1_mean=float(tensor.abs().mean()),
    )


def _history_to_rows(history: OptimizationHistory) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for index, total in enumerate(history.total):
        rows.append(
            {
                "iteration": float(index + 1),
                "total": total,
                "reconstruction": history.reconstruction[index],
                "perceptual": history.perceptual[index],
                "lighting_regularization": history.lighting_regularization[index],
                "albedo_smoothness": history.albedo_smoothness[index],
                "depth_smoothness": history.depth_smoothness[index],
                "residual_penalty": history.residual_penalty[index],
            }
        )
    return rows


def _write_history_csv(path: Path, history: OptimizationHistory) -> Path:
    rows = _history_to_rows(history)
    if not rows:
        path.write_text("")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _polyline(points: list[tuple[float, float]], color: str) -> str:
    if not points:
        return ""
    point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{point_text}"/>'


def _write_svg_line_plot(
    path: Path,
    series: dict[str, list[float]],
    *,
    title: str,
    y_label: str,
    log_scale: bool = False,
) -> Path | None:
    valid_series = {key: values for key, values in series.items() if values}
    if not valid_series:
        return None

    path = path.with_suffix(".svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 920, 520
    left, right, top, bottom = 70, 210, 50, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]

    max_length = max(len(values) for values in valid_series.values())
    transformed: dict[str, list[float]] = {}
    all_values: list[float] = []
    for key, values in valid_series.items():
        if log_scale:
            mapped = [math.log10(max(float(value), 1e-12)) for value in values]
        else:
            mapped = [float(value) for value in values]
        transformed[key] = mapped
        all_values.extend(mapped)

    y_min = min(all_values)
    y_max = max(all_values)
    if abs(y_max - y_min) < 1e-12:
        y_min -= 1.0
        y_max += 1.0

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.0f}" y="26" text-anchor="middle" font-family="Arial" font-size="20">{title}</text>',
        f'<text x="{width / 2:.0f}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="13">Iteration</text>',
        f'<text x="18" y="{height / 2:.0f}" transform="rotate(-90 18,{height / 2:.0f})" text-anchor="middle" font-family="Arial" font-size="13">{y_label}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#222"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#222"/>',
    ]
    for tick in range(6):
        ratio = tick / 5
        y = top + plot_height * ratio
        value = y_max - (y_max - y_min) * ratio
        label = f"{value:.3f}"
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#ddd"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{label}</text>')

    for index, (key, values) in enumerate(transformed.items()):
        points = []
        for step, value in enumerate(values):
            x = left + plot_width * (step / max(1, max_length - 1))
            y = top + plot_height * (1.0 - (value - y_min) / (y_max - y_min))
            points.append((x, y))
        color = colors[index % len(colors)]
        lines.append(_polyline(points, color))
        legend_y = top + 24 * index
        lines.append(f'<line x1="{width - right + 30}" y1="{legend_y}" x2="{width - right + 55}" y2="{legend_y}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{width - right + 62}" y="{legend_y + 4}" font-family="Arial" font-size="12">{key}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_loss_curve_svg(path: Path, history: OptimizationHistory) -> Path | None:
    return _write_svg_line_plot(
        path,
        {
            "total": history.total,
            "reconstruction": history.reconstruction,
            "perceptual": history.perceptual,
            "albedo_smoothness": history.albedo_smoothness,
            "depth_smoothness": history.depth_smoothness,
            "residual_penalty": history.residual_penalty,
        },
        title="Inverse Rendering Optimization Curves",
        y_label="log10(loss)",
        log_scale=True,
    )


def _write_lighting_svg(path: Path, lighting: Tensor) -> Path | None:
    coeff = lighting.detach().cpu().float()
    if coeff.ndim == 3:
        coeff = coeff[0]
    return _write_svg_line_plot(
        path,
        {
            "red": coeff[0].tolist(),
            "green": coeff[1].tolist(),
            "blue": coeff[2].tolist(),
        },
        title="Recovered Illumination Coefficients",
        y_label="coefficient value",
        log_scale=False,
    )


def _write_lighting_json(path: Path, lighting: Tensor) -> Path:
    coeff = lighting.detach().cpu().float()
    if coeff.ndim == 3:
        coeff = coeff[0]
    payload = {
        "description": "RGB spherical harmonics coefficients used by the differentiable renderer.",
        "channels": ["red", "green", "blue"],
        "basis": [f"SH{i}" for i in range(coeff.shape[-1])],
        "coefficients": coeff.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _write_component_statistics(path: Path, stats: dict[str, ComponentStatistics]) -> Path:
    payload = {key: value.to_dict() for key, value in stats.items()}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _write_diagnostic_markdown(path: Path, summary: DiagnosticSummary) -> Path:
    lines = [
        f"# Diagnostics: {summary.phase_name}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in summary.metrics.items():
        lines.append(f"| {key.upper()} | {value:.6f} |")

    lines.extend(["", "## Component Statistics", "", "| Component | Mean | Std | Min | Max | L1 Mean |"])
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for stat in summary.component_statistics.values():
        lines.append(
            f"| {stat.name} | {stat.mean:.6f} | {stat.std:.6f} | {stat.minimum:.6f} | {stat.maximum:.6f} | {stat.l1_mean:.6f} |"
        )

    lines.extend(["", "## Artifacts", ""])
    for name, artifact_path in summary.artifacts.items():
        lines.append(f"- `{name}`: `{artifact_path}`")
    path.write_text("\n".join(lines) + "\n")
    return path


def export_research_diagnostics(
    output_dir: str | Path,
    *,
    phase_name: str,
    result: InverseRenderingResult,
    target_image: Tensor,
    mask: Tensor | None = None,
) -> DiagnosticSummary:
    """Export plots, tensor visualizations, and JSON summaries for a phase."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction = result.refined.image if result.refined is not None else result.physical.image
    metrics_obj = compute_evaluation_metrics(prediction, target_image.to(prediction.device), use_lpips=False)
    metrics = {
        "psnr": metrics_obj.psnr,
        "ssim": metrics_obj.ssim,
        "lpips": -1.0 if metrics_obj.lpips is None else metrics_obj.lpips,
    }

    residual = None
    if result.refined is not None:
        residual = (result.refined.image - result.physical.image).detach()

    stats = {
        "depth": _tensor_stats("depth", result.state.depth),
        "albedo": _tensor_stats("albedo", result.state.albedo),
        "lighting": _tensor_stats("lighting", result.state.lighting),
        "physical_image": _tensor_stats("physical_image", result.physical.image),
        "prediction": _tensor_stats("prediction", prediction),
    }
    if residual is not None:
        stats["residual"] = _tensor_stats("residual", residual)
    if mask is not None:
        stats["mask"] = _tensor_stats("mask", mask)

    artifacts: dict[str, Path] = {}
    artifacts["history_csv"] = _write_history_csv(output_dir / "optimization_history.csv", result.history)
    artifacts["lighting_json"] = _write_lighting_json(output_dir / "lighting_coefficients.json", result.state.lighting)
    artifacts["component_statistics_json"] = _write_component_statistics(output_dir / "component_statistics.json", stats)

    loss_plot = _write_loss_curve_svg(output_dir / "loss_curves.svg", result.history)
    if loss_plot is not None:
        artifacts["loss_curves"] = loss_plot
    lighting_plot = _write_lighting_svg(output_dir / "lighting_coefficients.svg", result.state.lighting)
    if lighting_plot is not None:
        artifacts["lighting_coefficients_plot"] = lighting_plot

    target = _as_rgb(target_image)
    physical = _as_rgb(result.physical.image)
    prediction_rgb = _as_rgb(prediction)
    albedo = _as_rgb(result.state.albedo)
    shading = _as_rgb(result.physical.shading)
    depth = _normalize_map(result.state.depth).repeat(3, 1, 1)
    normals = _normal_visualization(result.physical.normals)
    error = _error_heatmap(prediction, target_image)

    artifacts["decomposition_grid"] = save_image_grid(
        output_dir / "decomposition_grid.png",
        [target, physical, prediction_rgb, albedo, shading, depth, normals, error],
        nrow=4,
    )
    if residual is not None:
        residual_vis = _normalize_map(residual.abs().mean(dim=1, keepdim=True)).repeat(3, 1, 1)
        artifacts["residual_map"] = save_image_grid(output_dir / "residual_map.png", [residual_vis], nrow=1)

    if result.history.snapshots:
        frames = []
        for snapshot in result.history.snapshots:
            snapshot_pred = snapshot.get("refined", snapshot["physical"])
            frames.append(_as_rgb(snapshot_pred))
        artifacts["optimization_snapshots"] = save_gif(output_dir / "optimization_snapshots.gif", frames, fps=4)

    summary = DiagnosticSummary(
        phase_name=phase_name,
        output_dir=output_dir,
        metrics=metrics,
        component_statistics=stats,
        artifacts=artifacts,
    )
    artifacts["summary_json"] = output_dir / "diagnostic_summary.json"
    artifacts["summary_json"].write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    artifacts["summary_markdown"] = _write_diagnostic_markdown(output_dir / "diagnostic_summary.md", summary)
    return summary


def write_ablation_report(path: str | Path, summaries: list[DiagnosticSummary]) -> Path:
    """Write a compact physical-vs-neural ablation table."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "# Ablation Summary",
        "",
        "| Phase | PSNR | SSIM | LPIPS |",
        "| --- | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        rows.append(
            "| {phase} | {psnr:.6f} | {ssim:.6f} | {lpips:.6f} |".format(
                phase=summary.phase_name,
                psnr=summary.metrics.get("psnr", -1.0),
                ssim=summary.metrics.get("ssim", -1.0),
                lpips=summary.metrics.get("lpips", -1.0),
            )
        )

    if len(summaries) >= 2:
        baseline, improved = summaries[0], summaries[-1]
        rows.extend(
            [
                "",
                "## Delta",
                "",
                f"- PSNR change: {improved.metrics['psnr'] - baseline.metrics['psnr']:.6f}",
                f"- SSIM change: {improved.metrics['ssim'] - baseline.metrics['ssim']:.6f}",
            ]
        )
    path.write_text("\n".join(rows) + "\n")
    return path
