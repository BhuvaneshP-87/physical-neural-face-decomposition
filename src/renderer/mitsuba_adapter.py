"""Optional Mitsuba 3 adapter with lazy imports."""

from __future__ import annotations

from pathlib import Path

import torch

from .geometry import ViewTransform
from .mitsuba_scene import MitsubaSceneBundle, MitsubaSceneConfig, MitsubaSceneTranslator
from .torch_renderer import RenderResult, TorchFaceRenderer


def mitsuba_available() -> bool:
    try:
        import mitsuba  # noqa: F401
        import drjit  # noqa: F401
    except Exception:
        return False
    return True


class MitsubaFaceRenderer:
    """Thin wrapper around Mitsuba 3 for higher-fidelity rendering.

    The implementation intentionally loads Mitsuba lazily so the repository remains
    usable in environments where Mitsuba is not installed.
    """

    def __init__(self, scene_config: MitsubaSceneConfig | None = None) -> None:
        self.scene_config = scene_config or MitsubaSceneConfig()
        self.translator = MitsubaSceneTranslator(self.scene_config)
        self.fallback_renderer = TorchFaceRenderer()

    def _require_backend(self):
        try:
            import mitsuba as mi
            import drjit as dr
        except Exception as exc:  # pragma: no cover - optional backend
            raise RuntimeError(
                "Mitsuba 3 / Dr.Jit are not available. Install the optional render dependencies to use this backend."
            ) from exc
        return mi, dr

    def export_scene(
        self,
        depth: torch.Tensor,
        albedo: torch.Tensor,
        lighting_coefficients: torch.Tensor | None,
        *,
        mask: torch.Tensor | None = None,
        view: ViewTransform | None = None,
        output_dir: str | Path | None = None,
        scene_name: str = "face_scene",
        environment_map: str | Path | None = None,
    ) -> MitsubaSceneBundle:
        return self.translator.export(
            depth,
            albedo,
            lighting_coefficients,
            mask=mask,
            view=view,
            output_dir=output_dir,
            scene_name=scene_name,
            environment_map=environment_map,
        )

    def render(
        self,
        depth: torch.Tensor,
        albedo: torch.Tensor,
        lighting_coefficients: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        view: ViewTransform | None = None,
        output_dir: str | Path | None = None,
        scene_name: str = "face_scene",
        environment_map: str | Path | None = None,
    ) -> RenderResult:  # pragma: no cover - optional backend
        mi, _ = self._require_backend()
        bundle = self.export_scene(
            depth,
            albedo,
            lighting_coefficients,
            mask=mask,
            view=view,
            output_dir=output_dir,
            scene_name=scene_name,
            environment_map=environment_map,
        )
        scene = mi.load_file(str(bundle.scene_xml))
        image = mi.render(scene, spp=bundle.samples_per_pixel)
        import numpy as np

        image_tensor = torch.from_numpy(np.asarray(image)).float()
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(-1).repeat(1, 1, 3)
        if image_tensor.shape[-1] == 4:
            image_tensor = image_tensor[..., :3]
        image_tensor = image_tensor.permute(2, 0, 1).contiguous().clamp(0.0, 1.0)
        physical = self.fallback_renderer(depth, albedo, lighting_coefficients, mask=mask, view=view)
        return RenderResult(
            image=image_tensor.unsqueeze(0),
            shading=physical.shading,
            normals=physical.normals,
            depth=physical.depth,
            mask=physical.mask,
            warped_albedo=physical.warped_albedo,
        )
