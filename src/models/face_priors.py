"""Optional FLAME and DECA integration hooks for facial geometry priors."""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .face_prior import FaceState, create_initial_face_state
from ..renderer.lighting import make_lighting_preset


@dataclass(slots=True)
class FacePriorConfig:
    """Configuration for optional face-prior backends."""

    backend: str = "synthetic"
    model_path: Path | None = None
    module_candidates: tuple[str, ...] = ()
    class_candidates: tuple[str, ...] = ()
    device: str = "cpu"
    image_size: tuple[int, int] = (256, 256)
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["model_path"] = None if self.model_path is None else str(self.model_path)
        return data


@dataclass(slots=True)
class FacePriorEstimate:
    """Recovered face prior information from a model or synthetic fallback."""

    depth: Tensor | None = None
    albedo: Tensor | None = None
    lighting: Tensor | None = None
    mask: Tensor | None = None
    vertices: Tensor | None = None
    normals: Tensor | None = None
    landmarks_2d: Tensor | None = None
    shape_params: Tensor | None = None
    expression_params: Tensor | None = None
    pose_params: Tensor | None = None
    camera_params: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: FacePriorEstimate._to_serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [FacePriorEstimate._to_serializable(item) for item in value]
        if torch.is_tensor(value):
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        if hasattr(value, "item") and callable(value.item):
            try:
                return value.item()
            except Exception:
                return str(value)
        return str(value)

    def to_face_state(
        self,
        image: Tensor,
        *,
        mask: Tensor | None = None,
        device: torch.device | None = None,
    ) -> FaceState:
        """Convert the prior estimate into an initializer for inverse rendering."""

        device = device or image.device
        mask = mask if mask is not None else self.mask
        return create_initial_face_state(image, mask=mask, device=device, prior=self)

    def to_dict(self) -> dict[str, Any]:
        def _shape_or_none(tensor: Tensor | None) -> tuple[int, ...] | None:
            return None if tensor is None else tuple(int(value) for value in tensor.shape)

        return {
            "depth_shape": _shape_or_none(self.depth),
            "albedo_shape": _shape_or_none(self.albedo),
            "lighting_shape": _shape_or_none(self.lighting),
            "mask_shape": _shape_or_none(self.mask),
            "vertices_shape": _shape_or_none(self.vertices),
            "normals_shape": _shape_or_none(self.normals),
            "landmarks_2d_shape": _shape_or_none(self.landmarks_2d),
            "shape_params_shape": _shape_or_none(self.shape_params),
            "expression_params_shape": _shape_or_none(self.expression_params),
            "pose_params_shape": _shape_or_none(self.pose_params),
            "camera_params_shape": _shape_or_none(self.camera_params),
            "metadata": self._to_serializable(self.metadata),
        }


class SyntheticFacePriorBackend:
    """Heuristic fallback prior built from the current inverse-rendering initializer."""

    backend_name = "synthetic"

    def estimate(self, image: Tensor, mask: Tensor | None = None) -> FacePriorEstimate:
        state = create_initial_face_state(image, mask=mask, device=image.device)
        return FacePriorEstimate(
            depth=state.depth,
            albedo=state.albedo,
            lighting=state.lighting,
            mask=mask,
            metadata={"backend": self.backend_name, "fallback": True},
        )


class _ModelPriorBackend:
    """Common lazy-loading logic for FLAME/DECA-style predictors."""

    backend_name = "model"
    module_candidates: tuple[str, ...] = ()
    class_candidates: tuple[str, ...] = ()

    def __init__(
        self,
        config: FacePriorConfig | None = None,
        *,
        model: Any | None = None,
        model_factory: Any | None = None,
    ) -> None:
        self.config = config or FacePriorConfig(
            backend=self.backend_name,
            module_candidates=self.module_candidates,
            class_candidates=self.class_candidates,
        )
        self._model = model
        self._model_factory = model_factory
        self._synthetic = SyntheticFacePriorBackend()

    def available(self) -> bool:
        return self._resolve_model(allow_missing=True) is not None

    def _candidate_imports(self) -> list[str]:
        names = list(self.config.module_candidates or self.module_candidates)
        if self.config.backend:
            names.insert(0, self.config.backend)
        return [name for name in names if name]

    def _instantiate_model(self, model_cls: Any) -> Any:
        kwargs_candidates: list[dict[str, Any]] = [{}]
        if self.config.model_path is not None:
            kwargs_candidates = [
                {"model_path": str(self.config.model_path)},
                {"checkpoint_path": str(self.config.model_path)},
                {"weights_path": str(self.config.model_path)},
                {"path": str(self.config.model_path)},
                {},
            ]
        kwargs_candidates = [
            {**candidate, **(self.config.extra_kwargs or {})} for candidate in kwargs_candidates
        ]
        for kwargs in kwargs_candidates:
            try:
                return model_cls(**kwargs)
            except TypeError:
                continue
        try:
            return model_cls()
        except Exception:
            return None

    def _resolve_model(self, allow_missing: bool = False) -> Any | None:
        if self._model is not None:
            return self._model
        if self._model_factory is not None:
            self._model = self._model_factory()
            return self._model

        for module_name in self._candidate_imports():
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            for class_name in self.config.class_candidates or self.class_candidates:
                model_cls = getattr(module, class_name, None)
                if model_cls is None:
                    continue
                self._model = self._instantiate_model(model_cls)
                if self._model is not None:
                    return self._model
            if callable(module):
                self._model = self._instantiate_model(module)
                if self._model is not None:
                    return self._model

        if allow_missing:
            return None
        return None

    def _call_model(self, model: Any, image: Tensor, mask: Tensor | None) -> Any:
        call_attempts = []
        if mask is not None:
            call_attempts.extend(
                [
                    lambda: model(image, mask=mask),
                    lambda: model.forward(image, mask=mask),
                    lambda: model.encode(image, mask=mask),
                    lambda: model.estimate(image, mask=mask),
                    lambda: model.run(image, mask=mask),
                ]
            )
        call_attempts.extend(
            [
                lambda: model(image),
                lambda: model.forward(image),
                lambda: model.encode(image),
                lambda: model.estimate(image),
                lambda: model.run(image),
            ]
        )
        for attempt in call_attempts:
            try:
                return attempt()
            except TypeError:
                continue
            except Exception:
                continue
        raise RuntimeError(f"Unable to execute the {self.backend_name} face-prior model.")

    @staticmethod
    def _extract_field(output: Any, names: tuple[str, ...]) -> Tensor | None:
        for name in names:
            if isinstance(output, dict) and name in output:
                value = output[name]
            elif hasattr(output, name):
                value = getattr(output, name)
            else:
                continue
            if torch.is_tensor(value):
                return value
            if value is not None:
                try:
                    return torch.as_tensor(value)
                except Exception:
                    continue
        return None

    def _estimate_from_output(self, output: Any, image: Tensor, mask: Tensor | None) -> FacePriorEstimate:
        if isinstance(output, FacePriorEstimate):
            return output
        if isinstance(output, FaceState):
            return FacePriorEstimate(
                depth=output.depth,
                albedo=output.albedo,
                lighting=output.lighting,
                mask=mask,
                metadata={"backend": self.backend_name, "source": "FaceState"},
            )

        depth = self._extract_field(output, ("depth", "depth_map", "z_map", "displacement", "displacement_map"))
        albedo = self._extract_field(output, ("albedo", "texture", "texture_map", "albedo_map", "tex"))
        lighting = self._extract_field(output, ("lighting", "light", "sh_coeffs", "sh", "coefficients"))
        mask_out = self._extract_field(output, ("mask", "face_mask", "segmentation", "silhouette"))
        vertices = self._extract_field(output, ("vertices", "verts", "mesh_vertices"))
        normals = self._extract_field(output, ("normals", "normal", "surface_normals"))
        landmarks = self._extract_field(output, ("landmarks", "landmarks_2d", "lmk2d", "keypoints"))
        shape_params = self._extract_field(output, ("shape_params", "shape", "betas", "identity"))
        expression_params = self._extract_field(output, ("expression_params", "expression", "exp"))
        pose_params = self._extract_field(output, ("pose_params", "pose", "camera_pose"))
        camera_params = self._extract_field(output, ("camera_params", "camera", "intrinsics"))

        if depth is None or albedo is None:
            fallback = self._synthetic.estimate(image, mask=mask)
            depth = depth if depth is not None else fallback.depth
            albedo = albedo if albedo is not None else fallback.albedo
            lighting = lighting if lighting is not None else fallback.lighting
            mask_out = mask_out if mask_out is not None else fallback.mask

        if lighting is None:
            lighting = make_lighting_preset("front", device=image.device, dtype=image.dtype).coefficients.unsqueeze(0)

        metadata = {
            "backend": self.backend_name,
            "model_loaded": self._model is not None,
            "source_type": type(output).__name__,
        }

        return FacePriorEstimate(
            depth=depth,
            albedo=albedo,
            lighting=lighting,
            mask=mask_out if mask_out is not None else mask,
            vertices=vertices,
            normals=normals,
            landmarks_2d=landmarks,
            shape_params=shape_params,
            expression_params=expression_params,
            pose_params=pose_params,
            camera_params=camera_params,
            metadata=metadata,
        )

    def estimate(self, image: Tensor, mask: Tensor | None = None) -> FacePriorEstimate:
        model = self._resolve_model()
        if model is None:
            return self._synthetic.estimate(image, mask=mask)
        output = self._call_model(model, image, mask)
        return self._estimate_from_output(output, image, mask)


class FlameAdapter(_ModelPriorBackend):
    """Best-effort FLAME integration wrapper."""

    backend_name = "flame"
    module_candidates = ("models.FLAME", "flame.FLAME", "flame")
    class_candidates = ("FLAME",)


class DecaAdapter(_ModelPriorBackend):
    """Best-effort DECA integration wrapper."""

    backend_name = "deca"
    module_candidates = ("decalib.deca", "deca.deca", "deca")
    class_candidates = ("DECA", "Deca")


def build_face_prior_backend(
    backend: str,
    *,
    model_path: str | Path | None = None,
    device: str = "cpu",
    extra_kwargs: dict[str, Any] | None = None,
) -> SyntheticFacePriorBackend | FlameAdapter | DecaAdapter:
    """Construct a face-prior backend by name."""

    normalized_backend = backend.lower().strip()
    config = FacePriorConfig(
        backend=normalized_backend,
        model_path=None if model_path is None else Path(model_path),
        device=device,
        extra_kwargs=extra_kwargs or {},
    )
    if normalized_backend in {"synthetic", "none", "fallback"}:
        return SyntheticFacePriorBackend()
    if normalized_backend == "flame":
        return FlameAdapter(config)
    if normalized_backend == "deca":
        return DecaAdapter(config)
    raise ValueError(f"Unsupported face-prior backend: {backend}")
