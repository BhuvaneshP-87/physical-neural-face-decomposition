"""Renderer implementations and geometry/lighting helpers."""

from .geometry import (
    ViewTransform,
    canonical_grid,
    create_face_mask,
    depth_to_normals,
    depth_to_points,
    rotate_normals,
    view_rotation_matrix,
    warp_grid_for_view,
)
from .lighting import (
    LightingPreset,
    make_lighting_preset,
    spherical_harmonics_basis,
    spherical_harmonics_shading,
)
from .torch_renderer import RenderResult, TorchFaceRenderer
from .bust_template import BustTemplateBundle, BustTemplateConfig, ProceduralBustTemplate

__all__ = [
    "ViewTransform",
    "canonical_grid",
    "create_face_mask",
    "depth_to_normals",
    "depth_to_points",
    "rotate_normals",
    "view_rotation_matrix",
    "warp_grid_for_view",
    "LightingPreset",
    "make_lighting_preset",
    "spherical_harmonics_basis",
    "spherical_harmonics_shading",
    "MitsubaFaceRenderer",
    "MitsubaSceneConfig",
    "DepthMeshExporter",
    "MitsubaSceneBundle",
    "MitsubaSceneTranslator",
    "mitsuba_available",
    "RenderResult",
    "TorchFaceRenderer",
    "BustTemplateBundle",
    "BustTemplateConfig",
    "ProceduralBustTemplate",
]


def __getattr__(name: str):
    if name in {"MitsubaFaceRenderer", "mitsuba_available"}:
        from .mitsuba_adapter import MitsubaFaceRenderer, mitsuba_available

        globals()["MitsubaFaceRenderer"] = MitsubaFaceRenderer
        globals()["mitsuba_available"] = mitsuba_available
        return globals()[name]
    if name in {"MitsubaSceneConfig", "DepthMeshExporter", "MitsubaSceneBundle", "MitsubaSceneTranslator"}:
        from .mitsuba_scene import DepthMeshExporter, MitsubaSceneBundle, MitsubaSceneConfig, MitsubaSceneTranslator

        globals()["MitsubaSceneConfig"] = MitsubaSceneConfig
        globals()["DepthMeshExporter"] = DepthMeshExporter
        globals()["MitsubaSceneBundle"] = MitsubaSceneBundle
        globals()["MitsubaSceneTranslator"] = MitsubaSceneTranslator
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
