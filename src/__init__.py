"""Top-level package for the physical-neural face decomposition project."""

from __future__ import annotations

from .config import ExperimentConfig, ExperimentMetadata, OptimizationConfig, RendererConfig

__all__ = [
    "ExperimentConfig",
    "ExperimentMetadata",
    "OptimizationConfig",
    "RendererConfig",
    "FaceDecompositionPipeline",
    "FaceDecompositionResult",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name in {"FaceDecompositionPipeline", "FaceDecompositionResult"}:
        from .pipeline import FaceDecompositionPipeline, FaceDecompositionResult

        globals()["FaceDecompositionPipeline"] = FaceDecompositionPipeline
        globals()["FaceDecompositionResult"] = FaceDecompositionResult
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
