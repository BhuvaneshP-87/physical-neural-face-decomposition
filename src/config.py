"""Configuration dataclasses used across the project."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:  # pragma: no cover - available in Python 3.11+
    import tomllib
except Exception:  # pragma: no cover - fallback for older interpreters
    tomllib = None


def _path_or_none(value: Any) -> Path | None:
    if value in {None, "", "null"}:
        return None
    return value if isinstance(value, Path) else Path(value)


def _normalize_path(value: Path | None) -> str | None:
    return None if value is None else str(value)


@dataclass(slots=True)
class ExperimentMetadata:
    """Run-level metadata used for experiment tracking."""

    name: str = "physical-neural-face-decomposition"
    run_dir: Path = field(default_factory=lambda: Path("outputs/run"))
    seed: int = 42
    device: str = "auto"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    resume_from: Path | None = None
    face_prior_backend: str = "synthetic"
    face_prior_model_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "run_dir": str(self.run_dir),
            "seed": self.seed,
            "device": self.device,
            "description": self.description,
            "tags": list(self.tags),
            "resume_from": _normalize_path(self.resume_from),
            "face_prior_backend": self.face_prior_backend,
            "face_prior_model_path": _normalize_path(self.face_prior_model_path),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentMetadata":
        tags = data.get("tags") or []
        seed_value = data.get("seed")
        return cls(
            name=str(data.get("name") or "physical-neural-face-decomposition"),
            run_dir=Path(data.get("run_dir") or "outputs/run"),
            seed=42 if seed_value is None else int(seed_value),
            device=str(data.get("device") or "auto"),
            description=str(data.get("description") or ""),
            tags=list(tags),
            resume_from=_path_or_none(data.get("resume_from")),
            face_prior_backend=str(data.get("face_prior_backend") or "synthetic"),
            face_prior_model_path=_path_or_none(data.get("face_prior_model_path")),
        )


@dataclass(slots=True)
class RendererConfig:
    """Rendering-related defaults."""

    image_size: tuple[int, int] = (256, 256)
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    clamp_output: bool = True
    use_soft_mask: bool = True
    epsilon: float = 1e-6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RendererConfig":
        return cls(
            image_size=tuple(data.get("image_size") or (256, 256)),
            background_color=tuple(data.get("background_color") or (0.0, 0.0, 0.0)),
            clamp_output=bool(data.get("clamp_output", True)),
            use_soft_mask=bool(data.get("use_soft_mask", True)),
            epsilon=float(data.get("epsilon", 1e-6)),
        )


@dataclass(slots=True)
class OptimizationConfig:
    """Optimization hyperparameters for inverse rendering."""

    iterations: int = 600
    lr_geometry: float = 2e-2
    lr_albedo: float = 2e-2
    lr_lighting: float = 2e-2
    lr_residual: float = 1e-4
    reconstruction_weight: float = 1.0
    perceptual_weight: float = 0.5
    lighting_regularization_weight: float = 5e-3
    albedo_smoothness_weight: float = 1e-2
    depth_smoothness_weight: float = 1e-2
    residual_weight: float = 2e-1
    use_residual_model: bool = False
    snapshot_every: int = 50

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationConfig":
        return cls(
            iterations=int(data.get("iterations", 600)),
            lr_geometry=float(data.get("lr_geometry", 2e-2)),
            lr_albedo=float(data.get("lr_albedo", 2e-2)),
            lr_lighting=float(data.get("lr_lighting", 2e-2)),
            lr_residual=float(data.get("lr_residual", 1e-4)),
            reconstruction_weight=float(data.get("reconstruction_weight", 1.0)),
            perceptual_weight=float(data.get("perceptual_weight", 0.5)),
            lighting_regularization_weight=float(data.get("lighting_regularization_weight", 5e-3)),
            albedo_smoothness_weight=float(data.get("albedo_smoothness_weight", 1e-2)),
            depth_smoothness_weight=float(data.get("depth_smoothness_weight", 1e-2)),
            residual_weight=float(data.get("residual_weight", 2e-1)),
            use_residual_model=bool(data.get("use_residual_model", False)),
            snapshot_every=int(data.get("snapshot_every", 50)),
        )


@dataclass(slots=True)
class TrainingConfig:
    """Defaults for residual model training."""

    epochs: int = 10
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    checkpoint_dir: Path = field(default_factory=lambda: Path("outputs/checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("outputs/logs"))
    log_every: int = 25
    save_every: int = 1
    keep_last_checkpoints: int = 3

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["checkpoint_dir"] = str(self.checkpoint_dir)
        data["log_dir"] = str(self.log_dir)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        return cls(
            epochs=int(data.get("epochs", 10)),
            batch_size=int(data.get("batch_size", 4)),
            lr=float(data.get("lr", 1e-4)),
            weight_decay=float(data.get("weight_decay", 1e-4)),
            num_workers=int(data.get("num_workers", 0)),
            checkpoint_dir=Path(data.get("checkpoint_dir") or "outputs/checkpoints"),
            log_dir=Path(data.get("log_dir") or "outputs/logs"),
            log_every=int(data.get("log_every", 25)),
            save_every=int(data.get("save_every", 1)),
            keep_last_checkpoints=int(data.get("keep_last_checkpoints", 3)),
        )


@dataclass(slots=True)
class ExperimentConfig:
    """Aggregate experiment configuration."""

    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "renderer": self.renderer.to_dict(),
            "optimization": self.optimization.to_dict(),
            "training": self.training.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            metadata=ExperimentMetadata.from_dict(data.get("metadata", {})),
            renderer=RendererConfig.from_dict(data.get("renderer", {})),
            optimization=OptimizationConfig.from_dict(data.get("optimization", {})),
            training=TrainingConfig.from_dict(data.get("training", {})),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            data = json.loads(path.read_text())
        elif suffix in {".toml", ".tml"}:
            if tomllib is None:
                raise RuntimeError("TOML support is unavailable in this Python runtime.")
            data = tomllib.loads(path.read_text())
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        return cls.from_dict(data)
