"""Real-image loading and face preprocessing helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from ..renderer.geometry import create_face_mask


@dataclass(slots=True)
class FaceBoundingBox:
    """A pixel-space bounding box for a detected face."""

    x: int
    y: int
    width: int
    height: int

    def expand(self, padding: float, image_size: tuple[int, int]) -> "FaceBoundingBox":
        """Expand the box to a padded square crop, clamped to the image bounds."""

        image_height, image_width = image_size
        side_length = max(self.width, self.height)
        side_length = int(round(side_length * (1.0 + 2.0 * padding)))
        side_length = max(1, min(side_length, image_width, image_height))

        center_x = self.x + self.width / 2.0
        center_y = self.y + self.height / 2.0
        x0 = int(round(center_x - side_length / 2.0))
        y0 = int(round(center_y - side_length / 2.0))
        x0 = max(0, min(x0, image_width - side_length))
        y0 = max(0, min(y0, image_height - side_length))
        return FaceBoundingBox(x=x0, y=y0, width=side_length, height=side_length)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


@dataclass(slots=True)
class FacePreprocessConfig:
    """Options controlling image loading and canonical face cropping."""

    output_size: tuple[int, int] = (256, 256)
    padding: float = 0.35
    detect_face: bool = True
    linearize: bool = True
    gamma: float = 2.2
    fallback_crop_scale: float = 0.9
    mask_axis_ratio: tuple[float, float] = (0.82, 0.95)
    mask_blur_kernel: int = 9
    remove_background: bool = True
    grabcut_iterations: int = 5
    keep_color_space: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass(slots=True)
class PreprocessedFace:
    """A standardized face crop with an approximate facial support mask."""

    image: Tensor
    mask: Tensor
    bbox: FaceBoundingBox | None
    original_size: tuple[int, int]
    crop_size: tuple[int, int]
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_sample(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "mask": self.mask,
            "bbox": None if self.bbox is None else self.bbox.as_tuple(),
            "original_size": self.original_size,
            "crop_size": self.crop_size,
            "source_path": self.source_path,
            "metadata": self.metadata,
        }


def _load_with_cv2(path: Path) -> Tensor | None:
    try:  # pragma: no cover - optional dependency
        import cv2
        import numpy as np

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor
    except Exception:
        return None


def _load_with_pillow(path: Path) -> Tensor:
    from PIL import Image
    import numpy as np

    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor


def load_image_tensor(path: str | Path) -> Tensor:
    """Load an RGB image from disk as a float tensor in `[0, 1]`."""

    path = Path(path)
    tensor = _load_with_cv2(path)
    if tensor is None:
        try:
            tensor = _load_with_pillow(path)
        except Exception as exc:  # pragma: no cover - user-facing failure
            raise RuntimeError(f"Unable to load image '{path}'. Install OpenCV or Pillow.") from exc
    return tensor


def save_image_tensor(path: str | Path, image: Tensor) -> Path:
    """Save an image tensor using the best available backend."""

    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.detach().cpu().clamp(0.0, 1.0)
    array = image.permute(1, 2, 0).mul(255.0).round().to(torch.uint8).numpy()

    try:  # pragma: no cover - optional dependency
        import cv2

        cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
        return path
    except Exception:
        pass

    try:  # pragma: no cover - optional dependency
        from PIL import Image

        Image.fromarray(array).save(path)
        return path
    except Exception:
        pass

    np.save(path.with_suffix(".npy"), array)
    return path.with_suffix(".npy")


def _largest_bbox(boxes: Any) -> FaceBoundingBox | None:
    if boxes is None or len(boxes) == 0:
        return None
    x, y, width, height = max(boxes, key=lambda box: box[2] * box[3])
    return FaceBoundingBox(int(x), int(y), int(width), int(height))


def detect_face_bbox(image: Tensor) -> FaceBoundingBox | None:
    """Detect the most prominent face using OpenCV Haar cascades when available."""

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected an RGB image tensor shaped [3, H, W].")

    try:  # pragma: no cover - optional dependency
        import cv2

        cascade_paths = [
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml",
            Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml",
        ]
        classifier = None
        for cascade_path in cascade_paths:
            candidate = cv2.CascadeClassifier(str(cascade_path))
            if not candidate.empty():
                classifier = candidate
                break
        if classifier is None:
            return None

        rgb = image.detach().cpu().permute(1, 2, 0).numpy()
        gray = cv2.cvtColor((rgb * 255.0).astype("uint8"), cv2.COLOR_RGB2GRAY)
        boxes = classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(32, 32),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return _largest_bbox(boxes)
    except Exception:
        return None


def _crop_tensor(image: Tensor, bbox: FaceBoundingBox) -> Tensor:
    y0 = max(0, bbox.y)
    x0 = max(0, bbox.x)
    y1 = min(image.shape[1], bbox.y + bbox.height)
    x1 = min(image.shape[2], bbox.x + bbox.width)
    cropped = image[:, y0:y1, x0:x1]
    if cropped.numel() == 0:
        return image
    return cropped


def _resize_tensor(image: Tensor, output_size: tuple[int, int]) -> Tensor:
    return F.interpolate(image.unsqueeze(0), size=output_size, mode="bilinear", align_corners=False).squeeze(0)


def _center_crop_box(image_size: tuple[int, int], scale: float) -> FaceBoundingBox:
    image_height, image_width = image_size
    side_length = int(round(min(image_height, image_width) * scale))
    side_length = max(1, min(side_length, image_width, image_height))
    x0 = (image_width - side_length) // 2
    y0 = (image_height - side_length) // 2
    return FaceBoundingBox(x=x0, y=y0, width=side_length, height=side_length)


def _make_support_mask(output_size: tuple[int, int], axis_ratio: tuple[float, float], blur_kernel: int) -> Tensor:
    mask = create_face_mask(output_size[0], output_size[1], axis_ratio=axis_ratio)
    if blur_kernel > 1:
        padding = blur_kernel // 2
        mask = F.avg_pool2d(mask, kernel_size=blur_kernel, stride=1, padding=padding)
    return mask.clamp(0.0, 1.0)


def _make_grabcut_mask(image: Tensor, iterations: int, blur_kernel: int) -> Tensor | None:
    """Estimate a foreground mask for the canonical crop using OpenCV GrabCut."""

    try:  # pragma: no cover - optional dependency
        import cv2
        import numpy as np

        image_uint8 = image.detach().cpu().clamp(0.0, 1.0)
        image_uint8 = image_uint8.permute(1, 2, 0).mul(255.0).round().to(torch.uint8).numpy()
        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
        height, width = image_bgr.shape[:2]

        margin_x = max(2, int(round(width * 0.08)))
        margin_y = max(2, int(round(height * 0.04)))
        rect = (margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y)
        if rect[2] <= 1 or rect[3] <= 1:
            return None

        grabcut_mask = np.zeros((height, width), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(
            image_bgr,
            grabcut_mask,
            rect,
            bgd_model,
            fgd_model,
            max(1, iterations),
            cv2.GC_INIT_WITH_RECT,
        )
        foreground = np.where(
            (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
            1.0,
            0.0,
        ).astype("float32")

        kernel_size = max(3, blur_kernel | 1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=1)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)
        foreground = cv2.GaussianBlur(foreground, (kernel_size, kernel_size), sigmaX=0)
        return torch.from_numpy(foreground).unsqueeze(0).unsqueeze(0).clamp(0.0, 1.0)
    except Exception:
        return None


class FacePreprocessor:
    """Canonical face cropper with face detection and fallback center cropping."""

    def __init__(self, config: FacePreprocessConfig | None = None) -> None:
        self.config = config or FacePreprocessConfig()

    def __call__(self, image_or_path: str | Path | Tensor, *, source_path: str | None = None) -> PreprocessedFace:
        return self.process(image_or_path, source_path=source_path)

    def process(self, image_or_path: str | Path | Tensor, *, source_path: str | None = None) -> PreprocessedFace:
        if isinstance(image_or_path, (str, Path)):
            path = Path(image_or_path)
            image = load_image_tensor(path)
            source_path = source_path or str(path)
        else:
            image = image_or_path.detach().clone()
            if image.ndim != 3 or image.shape[0] not in {1, 3, 4}:
                raise ValueError("Expected an image tensor shaped [C, H, W].")
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            elif image.shape[0] == 4:
                image = image[:3]

        image = image.float().clamp(0.0, 1.0)
        original_size = (int(image.shape[1]), int(image.shape[2]))

        detected_bbox = detect_face_bbox(image) if self.config.detect_face else None
        bbox = detected_bbox if detected_bbox is not None else _center_crop_box(original_size, self.config.fallback_crop_scale)
        bbox = bbox.expand(self.config.padding, original_size)

        cropped = _crop_tensor(image, bbox)
        if cropped.shape[1] < 2 or cropped.shape[2] < 2:
            cropped = image
            bbox = _center_crop_box(original_size, self.config.fallback_crop_scale)

        resized_srgb = _resize_tensor(cropped, self.config.output_size)
        foreground_mask = (
            _make_grabcut_mask(resized_srgb, self.config.grabcut_iterations, self.config.mask_blur_kernel)
            if self.config.remove_background
            else None
        )
        background_removed = foreground_mask is not None

        resized = resized_srgb
        if self.config.linearize and not self.config.keep_color_space:
            resized = resized.clamp(0.0, 1.0).pow(self.config.gamma)

        support_mask = _make_support_mask(self.config.output_size, self.config.mask_axis_ratio, self.config.mask_blur_kernel)
        mask = support_mask if foreground_mask is None else (support_mask * foreground_mask).clamp(0.0, 1.0)
        if self.config.remove_background:
            image_mask = mask.squeeze(0)
            background_color = torch.full_like(resized, 0.5)
            resized = resized * image_mask + background_color * (1.0 - image_mask)

        metadata = {
            "face_bbox_detected": detected_bbox is not None,
            "face_bbox_source": "opencv_haar" if detected_bbox is not None else "center_crop",
            "background_removed": background_removed,
            "background_removal_method": "opencv_grabcut" if background_removed else "ellipse_fallback",
            "padding": self.config.padding,
            **self.config.metadata,
        }
        return PreprocessedFace(
            image=resized.contiguous(),
            mask=mask.contiguous(),
            bbox=bbox,
            original_size=original_size,
            crop_size=(bbox.height, bbox.width),
            source_path=source_path,
            metadata=metadata,
        )

    def batch_process(self, image_or_paths: list[str | Path | Tensor]) -> list[PreprocessedFace]:
        return [self.process(item) for item in image_or_paths]
