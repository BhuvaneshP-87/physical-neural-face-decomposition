"""High-level orchestration for the four project phases."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .evaluation.metrics import compute_evaluation_metrics
from .evaluation.visualization import save_gif, save_image_grid
from .evaluation.diagnostics import DiagnosticSummary, export_research_diagnostics, write_ablation_report
from .config import ExperimentConfig
from .data.preprocessing import FacePreprocessConfig, FacePreprocessor
from .models.face_prior import FaceState
from .models.face_priors import FacePriorEstimate, build_face_prior_backend
from .models.residual_net import ResidualAppearanceNet
from .optimization.mitsuba_inverse import MitsubaInverseConfig, run_mitsuba_inverse_experiment
from .optimization.inverse_rendering import InverseRenderer, InverseRenderingResult
from .renderer.bust_template import BustTemplateConfig, ProceduralBustTemplate
from .renderer.geometry import ViewTransform
from .renderer.mitsuba_adapter import MitsubaFaceRenderer
from .renderer.mitsuba_scene import MitsubaSceneBundle, MitsubaSceneConfig
from .renderer.torch_renderer import RenderResult, TorchFaceRenderer
from .training.datasets import SyntheticFaceDataset


@dataclass(slots=True)
class FaceDecompositionResult:
    """Collected outputs from the baseline and residual pipelines."""

    inverse: InverseRenderingResult
    metrics: dict[str, float] = field(default_factory=dict)
    relighting: dict[str, RenderResult] = field(default_factory=dict)
    novel_views: dict[str, RenderResult] = field(default_factory=dict)


class FaceDecompositionPipeline:
    """Convenience wrapper for the inverse rendering and relighting workflow."""

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config or ExperimentConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.renderer = TorchFaceRenderer(
            background_color=self.config.renderer.background_color,
            clamp_output=self.config.renderer.clamp_output,
            use_soft_mask=self.config.renderer.use_soft_mask,
        ).to(self.device)
        self.inverse_renderer = InverseRenderer(renderer=self.renderer, config=self.config.optimization)
        self.preprocessor = FacePreprocessor(
            FacePreprocessConfig(output_size=self.config.renderer.image_size)
        )
        self.face_prior_backend = build_face_prior_backend(
            self.config.metadata.face_prior_backend,
            model_path=self.config.metadata.face_prior_model_path,
            device=str(self.device),
        )
        self.mitsuba_renderer = MitsubaFaceRenderer(
            MitsubaSceneConfig(
                output_dir=self.config.metadata.run_dir / "mitsuba",
                sensor_fov_degrees=35.0,
            )
        )
        self.bust_template = ProceduralBustTemplate(
            BustTemplateConfig(output_dir=self.config.metadata.run_dir / "bust_template")
        )

    def _ensure_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.device)
        if image.ndim == 3:
            return image.unsqueeze(0)
        return image

    def preprocess_input(self, image_or_path: str | Path | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, object], FacePriorEstimate]:
        """Load and preprocess a real image, then estimate a face prior for initialization."""

        if isinstance(image_or_path, (str, Path)):
            processed = self.preprocessor(image_or_path)
            target = processed.image.to(self.device)
            mask = processed.mask.to(self.device)
            prior = self.face_prior_backend.estimate(target, mask=mask)
            return target, mask, processed.metadata, prior

        if image_or_path.ndim == 4 and image_or_path.shape[0] == 1:
            image_or_path = image_or_path.squeeze(0)
        processed = self.preprocessor.process(image_or_path)
        target = processed.image.to(self.device)
        mask = processed.mask.to(self.device)
        prior = self.face_prior_backend.estimate(target, mask=mask)
        return target, mask, processed.metadata, prior

    def export_mitsuba_scene(
        self,
        result: InverseRenderingResult,
        *,
        output_dir: str | Path | None = None,
        mask: torch.Tensor | None = None,
        view: ViewTransform | None = None,
        scene_name: str = "face_scene",
        environment_map: str | Path | None = None,
    ) -> MitsubaSceneBundle:
        """Export the recovered face as a Mitsuba scene bundle."""

        return self.mitsuba_renderer.export_scene(
            result.state.depth,
            result.state.albedo,
            result.state.lighting,
            mask=mask,
            view=view,
            output_dir=output_dir,
            scene_name=scene_name,
            environment_map=environment_map,
        )

    def run_phase1(
        self,
        target_image: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        num_iterations: int | None = None,
        initial_state: FaceState | None = None,
    ) -> InverseRenderingResult:
        target = self._ensure_image(target_image)
        mask = mask.to(self.device) if mask is not None else None
        return self.inverse_renderer.optimize(
            target,
            mask=mask,
            num_iterations=num_iterations,
            residual_model=None,
            initial_state=initial_state,
        )

    def run_phase2(
        self,
        target_image: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        num_iterations: int | None = None,
        initial_state: FaceState | None = None,
    ) -> InverseRenderingResult:
        target = self._ensure_image(target_image)
        mask = mask.to(self.device) if mask is not None else None
        residual_model = ResidualAppearanceNet().to(self.device, dtype=target.dtype)
        return self.inverse_renderer.optimize(
            target,
            mask=mask,
            num_iterations=num_iterations,
            residual_model=residual_model,
            initial_state=initial_state,
        )

    @torch.no_grad()
    def relight(
        self,
        result: InverseRenderingResult,
        *,
        presets: list[str] | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, RenderResult]:
        return self.inverse_renderer.relight(result.state, presets=presets, mask=mask)

    @torch.no_grad()
    def novel_view_sweep(
        self,
        result: InverseRenderingResult,
        *,
        yaw_values: list[float] | None = None,
        pitch_values: list[float] | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, RenderResult]:
        yaw_values = yaw_values or [-15.0, -10.0, 0.0, 10.0, 15.0]
        pitch_values = pitch_values or [-10.0, 0.0, 10.0]
        return self.inverse_renderer.novel_view_sweep(result.state, yaw_values=yaw_values, pitch_values=pitch_values, mask=mask)

    def summarize(
        self,
        result: InverseRenderingResult,
        target_image: torch.Tensor,
        *,
        use_lpips: bool = False,
    ) -> dict[str, float]:
        prediction = result.refined.image if result.refined is not None else result.physical.image
        metrics = compute_evaluation_metrics(prediction, target_image.to(prediction.device), use_lpips=use_lpips)
        return {
            "psnr": metrics.psnr,
            "ssim": metrics.ssim,
            "lpips": -1.0 if metrics.lpips is None else metrics.lpips,
        }

    def save_phase_outputs(
        self,
        output_dir: str | Path,
        *,
        result: InverseRenderingResult,
        target_image: torch.Tensor,
        relighting: dict[str, RenderResult] | None = None,
        novel_views: dict[str, RenderResult] | None = None,
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        physical = result.physical.image.squeeze(0).detach().cpu()
        refined = result.refined.image.squeeze(0).detach().cpu() if result.refined is not None else physical
        save_image_grid(output_dir / "reconstruction_grid.png", [target_image.squeeze(0).cpu(), physical, refined], nrow=3)

        if relighting:
            for name, render in relighting.items():
                save_image_grid(output_dir / f"relight_{name}.png", [render.image.squeeze(0).detach().cpu()], nrow=1)

        if novel_views:
            save_gif(
                output_dir / "novel_view.gif",
                [render.image.squeeze(0).detach().cpu() for render in novel_views.values()],
                fps=8,
            )
        return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Physical-neural face decomposition demo")
    parser.add_argument("--config", type=Path, default=None, help="Optional experiment config JSON/TOML file.")
    parser.add_argument("--input", type=Path, default=None, help="Optional face image path.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"), help="Directory for artifacts.")
    parser.add_argument("--iterations", type=int, default=None, help="Override inverse-rendering iterations.")
    parser.add_argument("--synthetic", action="store_true", help="Use the synthetic generator even if an input is provided.")
    parser.add_argument("--phase", choices=["phase1", "phase2", "both"], default="both")
    parser.add_argument("--face-backend", choices=["synthetic", "flame", "deca"], default=None, help="Override the face-prior backend.")
    parser.add_argument("--face-model-path", type=Path, default=None, help="Optional FLAME/DECA checkpoint or model directory.")
    parser.add_argument("--export-mitsuba", action="store_true", help="Export a Mitsuba scene bundle after optimization.")
    parser.add_argument("--export-bust-template", action="store_true", help="Export an offline clay bust template OBJ.")
    parser.add_argument("--bust-only", action="store_true", help="Only export the offline clay bust template and skip optimization.")
    parser.add_argument("--mitsuba-inverse-demo", action="store_true", help="Run the optional Mitsuba/Dr.Jit inverse-rendering gradient demo.")
    parser.add_argument("--mitsuba-iterations", type=int, default=32, help="Iterations for --mitsuba-inverse-demo.")
    parser.add_argument("--mitsuba-spp", type=int, default=16, help="Samples per pixel for --mitsuba-inverse-demo.")
    parser.add_argument("--no-diagnostics", action="store_true", help="Skip research diagnostic plots and JSON exports.")
    parser.add_argument("--scene-name", type=str, default="face_scene", help="Base filename for Mitsuba exports.")
    args = parser.parse_args(argv)

    config = ExperimentConfig.load(args.config) if args.config is not None else ExperimentConfig()
    if args.face_backend is not None:
        config.metadata.face_prior_backend = args.face_backend
    if args.face_model_path is not None:
        config.metadata.face_prior_model_path = args.face_model_path
    config.metadata.run_dir = args.output_dir
    config.training.checkpoint_dir = args.output_dir / "checkpoints"
    config.training.log_dir = args.output_dir / "logs"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.save(args.output_dir / "config.json")

    pipeline = FaceDecompositionPipeline(config=config)
    if args.mitsuba_inverse_demo:
        mitsuba_summary = run_mitsuba_inverse_experiment(
            MitsubaInverseConfig(
                output_dir=args.output_dir / "mitsuba_inverse",
                iterations=args.mitsuba_iterations,
                spp=args.mitsuba_spp,
            )
        )
        print(json.dumps(mitsuba_summary.to_dict(), indent=2, sort_keys=True, default=str))
        if mitsuba_summary.status == "skipped":
            return 0

    if args.export_bust_template or args.bust_only:
        bust_bundle = pipeline.bust_template.export(
            output_dir=args.output_dir / "bust_template",
            scene_name="clay_bust",
        )
        (args.output_dir / "bust_template_bundle.txt").write_text(str(bust_bundle.mesh_obj))
        if args.bust_only:
            print(f"Saved bust template to {bust_bundle.mesh_obj}")
            return 0

    if args.synthetic or args.input is None:
        sample = SyntheticFaceDataset(length=1)[0]
        target_image = sample["image"].to(pipeline.device)
        mask = sample["mask"].to(pipeline.device)
        input_metadata: dict[str, object] = {"source": "synthetic"}
        prior_estimate = pipeline.face_prior_backend.estimate(target_image, mask=mask)
    else:
        target_image, mask, input_metadata, prior_estimate = pipeline.preprocess_input(args.input)

    (args.output_dir / "input_metadata.json").write_text(json.dumps(input_metadata, indent=2, sort_keys=True, default=str))
    preview_mask = mask.detach().cpu()
    if preview_mask.ndim == 4:
        preview_mask = preview_mask.squeeze(0)
    if preview_mask.shape[0] == 1:
        preview_mask = preview_mask.repeat(3, 1, 1)
    save_image_grid(
        args.output_dir / "preprocessed_input.png",
        [target_image.detach().cpu(), preview_mask],
        nrow=2,
    )
    initial_state = prior_estimate.to_face_state(target_image, mask=mask, device=pipeline.device)

    summary_path: Path
    diagnostic_summaries: list[DiagnosticSummary] = []
    export_dir = args.output_dir
    if args.phase == "phase1":
        result = pipeline.run_phase1(target_image, mask=mask, num_iterations=args.iterations, initial_state=initial_state)
        relighting = pipeline.relight(result, mask=mask)
        novel_views = pipeline.novel_view_sweep(result, mask=mask)
        summary = pipeline.summarize(result, target_image)
        pipeline.save_phase_outputs(
            export_dir,
            result=result,
            target_image=target_image,
            relighting=relighting,
            novel_views=novel_views,
        )
        if not args.no_diagnostics:
            diagnostic_summaries.append(
                export_research_diagnostics(
                    export_dir / "diagnostics",
                    phase_name="phase1_physical_only",
                    result=result,
                    target_image=target_image,
                    mask=mask,
                )
            )
        summary_path = export_dir / "metrics.txt"
        summary_path.write_text("\n".join(f"{key}: {value:.4f}" for key, value in summary.items()))
        if args.export_mitsuba:
            bundle = pipeline.export_mitsuba_scene(result, output_dir=export_dir / "mitsuba", mask=mask, scene_name=args.scene_name)
            (export_dir / "mitsuba_bundle.txt").write_text(str(bundle.scene_xml))
    else:
        if args.phase == "phase2":
            result = pipeline.run_phase2(
                target_image,
                mask=mask,
                num_iterations=args.iterations,
                initial_state=initial_state,
            )
            relighting = pipeline.relight(result, mask=mask)
            novel_views = pipeline.novel_view_sweep(result, mask=mask)
            summary = pipeline.summarize(result, target_image)
            pipeline.save_phase_outputs(
                export_dir,
                result=result,
                target_image=target_image,
                relighting=relighting,
                novel_views=novel_views,
            )
            if not args.no_diagnostics:
                diagnostic_summaries.append(
                    export_research_diagnostics(
                        export_dir / "diagnostics",
                        phase_name="phase2_physical_neural",
                        result=result,
                        target_image=target_image,
                        mask=mask,
                    )
                )
            summary_path = export_dir / "metrics.txt"
            summary_path.write_text("\n".join(f"{key}: {value:.4f}" for key, value in summary.items()))
            if args.export_mitsuba:
                bundle = pipeline.export_mitsuba_scene(result, output_dir=export_dir / "mitsuba", mask=mask, scene_name=args.scene_name)
                (export_dir / "mitsuba_bundle.txt").write_text(str(bundle.scene_xml))
        else:
            phase1_result = pipeline.run_phase1(
                target_image,
                mask=mask,
                num_iterations=args.iterations,
                initial_state=initial_state,
            )
            phase1_dir = export_dir / "phase1"
            phase1_relighting = pipeline.relight(phase1_result, mask=mask)
            phase1_novel_views = pipeline.novel_view_sweep(phase1_result, mask=mask)
            phase1_summary = pipeline.summarize(phase1_result, target_image)
            pipeline.save_phase_outputs(
                phase1_dir,
                result=phase1_result,
                target_image=target_image,
                relighting=phase1_relighting,
                novel_views=phase1_novel_views,
            )
            if not args.no_diagnostics:
                diagnostic_summaries.append(
                    export_research_diagnostics(
                        phase1_dir / "diagnostics",
                        phase_name="phase1_physical_only",
                        result=phase1_result,
                        target_image=target_image,
                        mask=mask,
                    )
                )
            phase1_summary_path = phase1_dir / "metrics.txt"
            phase1_summary_path.write_text("\n".join(f"{key}: {value:.4f}" for key, value in phase1_summary.items()))

            result = pipeline.run_phase2(
                target_image,
                mask=mask,
                num_iterations=args.iterations,
                initial_state=phase1_result.state,
            )

            relighting = pipeline.relight(result, mask=mask)
            novel_views = pipeline.novel_view_sweep(result, mask=mask)
            summary = pipeline.summarize(result, target_image)
            phase2_dir = export_dir / "phase2"
            pipeline.save_phase_outputs(
                phase2_dir,
                result=result,
                target_image=target_image,
                relighting=relighting,
                novel_views=novel_views,
            )
            if not args.no_diagnostics:
                diagnostic_summaries.append(
                    export_research_diagnostics(
                        phase2_dir / "diagnostics",
                        phase_name="phase2_physical_neural",
                        result=result,
                        target_image=target_image,
                        mask=mask,
                    )
                )
            summary_path = phase2_dir / "metrics.txt"
            summary_path.write_text("\n".join(f"{key}: {value:.4f}" for key, value in summary.items()))
            if args.export_mitsuba:
                bundle = pipeline.export_mitsuba_scene(result, output_dir=phase2_dir / "mitsuba", mask=mask, scene_name=args.scene_name)
                (phase2_dir / "mitsuba_bundle.txt").write_text(str(bundle.scene_xml))

    if diagnostic_summaries and not args.no_diagnostics:
        write_ablation_report(args.output_dir / "ablation_summary.md", diagnostic_summaries)

    print(f"Saved outputs to {args.output_dir}")
    print(summary_path.read_text())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
