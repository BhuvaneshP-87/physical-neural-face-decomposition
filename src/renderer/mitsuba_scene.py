"""Scene export utilities for Mitsuba 3."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from ..data.preprocessing import save_image_tensor
from .geometry import ViewTransform, canonical_grid, depth_to_normals
from .lighting import spherical_harmonics_shading

MeshFace = tuple[int, int, int] | tuple[int, int, int, str]


@dataclass(slots=True)
class MitsubaSceneConfig:
    """Configuration for exporting a Mitsuba scene bundle."""

    output_dir: Path = field(default_factory=lambda: Path("outputs/mitsuba"))
    mesh_scale: float = 1.0
    depth_scale: float = 0.15
    camera_distance: float = 2.5
    sensor_fov_degrees: float = 35.0
    samples_per_pixel: int = 64
    film_resolution: tuple[int, int] = (512, 512)
    environment_map_resolution: tuple[int, int] = (64, 128)
    use_generated_environment_map: bool = True
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    texture_gamma_correction: bool = False
    vertical_flip: bool = True
    face_mask_threshold: float = 0.25
    smooth_depth_for_export: bool = True
    depth_smoothing_passes: int = 8
    learned_depth_weight: float = 0.25
    export_closed_head_proxy: bool = True
    head_depth: float = 4.0
    proxy_material_color: tuple[float, float, float] = (0.58, 0.46, 0.38)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


@dataclass(slots=True)
class MitsubaSceneBundle:
    """Files generated for a Mitsuba rendering pass."""

    scene_xml: Path
    mesh_obj: Path
    albedo_texture: Path
    environment_map: Path | None
    metadata_path: Path
    output_dir: Path
    samples_per_pixel: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _ensure_4d(tensor: Tensor) -> Tensor:
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected a 3D or 4D tensor, got shape {tuple(tensor.shape)}.")
    return tensor


def _first_sample(tensor: Tensor) -> Tensor:
    tensor = _ensure_4d(tensor)
    return tensor[0]


def _as_mask(mask: Tensor | None, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if mask is None:
        return torch.ones(1, 1, height, width, device=device, dtype=dtype)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.ndim != 4:
        raise ValueError("Mask must be [H, W], [1, H, W], or [B, 1, H, W].")
    return mask.to(device=device, dtype=dtype)


def _lighting_to_environment_map(
    lighting_coefficients: Tensor,
    resolution: tuple[int, int],
) -> Tensor:
    lighting_coefficients = _ensure_4d(lighting_coefficients) if lighting_coefficients.ndim == 4 else lighting_coefficients
    if lighting_coefficients.ndim == 3:
        coeff = lighting_coefficients[0]
    elif lighting_coefficients.ndim == 2:
        coeff = lighting_coefficients
    else:
        raise ValueError("Lighting coefficients must have shape [3, 9] or [B, 3, 9].")

    height, width = resolution
    theta = torch.linspace(0.0, math.pi, height, device=coeff.device, dtype=coeff.dtype)
    phi = torch.linspace(-math.pi, math.pi, width, device=coeff.device, dtype=coeff.dtype)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")

    x = torch.sin(theta_grid) * torch.cos(phi_grid)
    y = torch.cos(theta_grid)
    z = torch.sin(theta_grid) * torch.sin(phi_grid)
    normals = torch.stack((x, y, z), dim=0).unsqueeze(0)
    envmap = spherical_harmonics_shading(normals, coeff)
    envmap = envmap.squeeze(0).clamp(0.0, 1.0)
    return envmap


def _smooth_masked_depth(depth_map: Tensor, mask: Tensor, passes: int) -> Tensor:
    """Smooth depth inside the face mask without bleeding in the background."""

    depth = depth_map.unsqueeze(0).unsqueeze(0)
    mask_batched = mask.unsqueeze(0).unsqueeze(0).clamp(0.0, 1.0)
    for _ in range(max(0, passes)):
        weighted_depth = F.avg_pool2d(depth * mask_batched, kernel_size=5, stride=1, padding=2)
        weights = F.avg_pool2d(mask_batched, kernel_size=5, stride=1, padding=2).clamp_min(1e-6)
        depth = weighted_depth / weights
    return depth.squeeze(0).squeeze(0)


def _analytic_face_depth(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Create a simple convex face-shaped depth prior for stable mesh export."""

    grid = canonical_grid(height, width, device=device, dtype=dtype)
    x = grid[..., 0]
    y = grid[..., 1]
    face = torch.exp(-((x / 0.78).square() + (y / 0.96).square()) * 1.25)
    nose = 0.35 * torch.exp(-((x / 0.18).square() + ((y + 0.04) / 0.26).square()) * 1.2)
    brow = 0.05 * torch.exp(-((x / 0.58).square() + ((y - 0.22) / 0.16).square()) * 2.0)
    chin = 0.07 * torch.exp(-((x / 0.42).square() + ((y + 0.62) / 0.18).square()) * 1.5)
    return (face + nose + brow + chin).clamp(0.0, 1.0)


def _prepare_depth_for_mesh(depth_map: Tensor, mask: Tensor, config: MitsubaSceneConfig) -> Tensor:
    """Turn optimized depth into a stable face-like displacement map for OBJ export."""

    if not config.smooth_depth_for_export:
        return depth_map

    mask = mask.clamp(0.0, 1.0)
    smoothed = _smooth_masked_depth(depth_map, mask, config.depth_smoothing_passes)
    valid = mask > config.face_mask_threshold
    if bool(valid.any()):
        values = smoothed[valid]
        mean = values.mean()
        std = values.std().clamp_min(1e-4)
        smoothed = smoothed.clamp(mean - 2.0 * std, mean + 2.0 * std)
        min_value = smoothed[valid].min()
        max_value = smoothed[valid].max()
        smoothed = (smoothed - min_value) / (max_value - min_value).clamp_min(1e-6)
    else:
        smoothed = torch.zeros_like(smoothed)

    prior = _analytic_face_depth(
        depth_map.shape[0],
        depth_map.shape[1],
        device=depth_map.device,
        dtype=depth_map.dtype,
    )
    learned_weight = min(1.0, max(0.0, config.learned_depth_weight))
    depth = learned_weight * smoothed + (1.0 - learned_weight) * prior
    return depth * mask


def _remove_texture_background(texture: Tensor, mask: Tensor) -> Tensor:
    """Replace background texels with the mean foreground color before export."""

    if mask.ndim == 4:
        mask = mask[0]
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    mask = mask.to(device=texture.device, dtype=texture.dtype).clamp(0.0, 1.0)
    valid = mask > 0.5
    if bool(valid.any()):
        foreground = texture * mask
        mean_color = foreground.sum(dim=(1, 2), keepdim=True) / mask.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    else:
        mean_color = texture.mean(dim=(1, 2), keepdim=True)
    return (texture * mask + mean_color * (1.0 - mask)).clamp(0.0, 1.0)


def _ellipsoid_back_depth(mask: Tensor, config: MitsubaSceneConfig) -> Tensor:
    """Create a rounded back surface so side views have visible head volume."""

    height, width = mask.shape
    grid = canonical_grid(height, width, device=mask.device, dtype=mask.dtype)
    x = grid[..., 0] / 0.86
    y = grid[..., 1] / 1.02
    radius = (x.square() + y.square()).clamp(0.0, 1.0)
    rounded_back = torch.sqrt((1.0 - radius).clamp_min(0.0))
    return -config.head_depth * rounded_back * mask


def _mesh_normals(vertices: Tensor, fallback_depth: Tensor, front_vertex_count: int) -> Tensor:
    """Approximate normals for a combined front/back proxy mesh."""

    height, width = fallback_depth.shape
    front_normals = depth_to_normals(fallback_depth.unsqueeze(0).unsqueeze(0))[0].permute(1, 2, 0).contiguous()
    back_vertices = vertices.reshape(-1, 3)[front_vertex_count:]
    back_normals = back_vertices.clone()
    back_normals[..., 2] = -back_normals[..., 2].abs().clamp_min(1e-4)
    back_normals = F.normalize(back_normals, dim=-1)
    normals = torch.cat((front_normals.reshape(-1, 3), back_normals), dim=0)
    return normals.reshape(2, height, width, 3)


class DepthMeshExporter:
    """Convert a depth-albedo pair into a textured OBJ mesh."""

    def __init__(self, config: MitsubaSceneConfig) -> None:
        self.config = config

    def build_mesh(
        self,
        depth: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, list[MeshFace]]:
        depth = _first_sample(depth)
        if depth.ndim != 3 or depth.shape[0] != 1:
            raise ValueError("Depth must have shape [1, H, W] or [B, 1, H, W].")
        depth_map = depth[0]
        height, width = depth_map.shape
        mask_tensor = _as_mask(mask, height, width, depth.device, depth.dtype)[0, 0]
        depth_map = _prepare_depth_for_mesh(depth_map, mask_tensor, self.config)

        grid = canonical_grid(height, width, device=depth.device, dtype=depth.dtype)
        vertices_xy = grid.clone()
        if self.config.vertical_flip:
            vertices_xy[..., 1] = -vertices_xy[..., 1]

        front_z = depth_map * self.config.depth_scale
        front_vertices = torch.stack(
            (
                vertices_xy[..., 0] * self.config.mesh_scale,
                vertices_xy[..., 1] * self.config.mesh_scale,
                front_z,
            ),
            dim=-1,
        )

        if self.config.export_closed_head_proxy:
            back_z = _ellipsoid_back_depth(mask_tensor, self.config) * self.config.depth_scale
            back_vertices = torch.stack(
                (
                    vertices_xy[..., 0] * self.config.mesh_scale,
                    vertices_xy[..., 1] * self.config.mesh_scale,
                    back_z,
                ),
                dim=-1,
            )
            vertices = torch.stack((front_vertices, back_vertices), dim=0)
            normals = _mesh_normals(vertices, depth_map, height * width)
        else:
            vertices = front_vertices
            normals = depth_to_normals(depth_map.unsqueeze(0).unsqueeze(0))[0].permute(1, 2, 0).contiguous()

        if self.config.vertical_flip:
            normals[..., 1] = -normals[..., 1]

        u = (grid[..., 0] + 1.0) * 0.5
        v = 1.0 - (grid[..., 1] + 1.0) * 0.5
        uv = torch.stack((u, v), dim=-1)
        if self.config.export_closed_head_proxy:
            uv = torch.stack((uv, uv), dim=0)

        faces: list[MeshFace] = []
        valid_mask = mask_tensor >= self.config.face_mask_threshold
        back_offset = height * width

        def front_index(row: int, col: int) -> int:
            return row * width + col + 1

        def back_index(row: int, col: int) -> int:
            return back_offset + row * width + col + 1

        for row in range(height - 1):
            for col in range(width - 1):
                quad_mask = mask_tensor[row : row + 2, col : col + 2].mean().item()
                if quad_mask < self.config.face_mask_threshold:
                    continue
                index00 = front_index(row, col)
                index10 = front_index(row, col + 1)
                index01 = front_index(row + 1, col)
                index11 = front_index(row + 1, col + 1)
                faces.append((index00, index01, index11, "face_material"))
                faces.append((index00, index11, index10, "face_material"))
                if self.config.export_closed_head_proxy:
                    back00 = back_index(row, col)
                    back10 = back_index(row, col + 1)
                    back01 = back_index(row + 1, col)
                    back11 = back_index(row + 1, col + 1)
                    faces.append((back00, back10, back11, "proxy_material"))
                    faces.append((back00, back11, back01, "proxy_material"))

        if self.config.export_closed_head_proxy:
            for row in range(height):
                for col in range(width - 1):
                    if not bool(valid_mask[row, col] and valid_mask[row, col + 1]):
                        continue
                    above_outside = row == 0 or not bool(valid_mask[row - 1, col] and valid_mask[row - 1, col + 1])
                    below_outside = row == height - 1 or not bool(valid_mask[row + 1, col] and valid_mask[row + 1, col + 1])
                    if above_outside:
                        faces.append((front_index(row, col), back_index(row, col), back_index(row, col + 1), "proxy_material"))
                        faces.append((front_index(row, col), back_index(row, col + 1), front_index(row, col + 1), "proxy_material"))
                    if below_outside:
                        faces.append((front_index(row, col + 1), back_index(row, col + 1), back_index(row, col), "proxy_material"))
                        faces.append((front_index(row, col + 1), back_index(row, col), front_index(row, col), "proxy_material"))

            for row in range(height - 1):
                for col in range(width):
                    if not bool(valid_mask[row, col] and valid_mask[row + 1, col]):
                        continue
                    left_outside = col == 0 or not bool(valid_mask[row, col - 1] and valid_mask[row + 1, col - 1])
                    right_outside = col == width - 1 or not bool(valid_mask[row, col + 1] and valid_mask[row + 1, col + 1])
                    if left_outside:
                        faces.append((front_index(row + 1, col), back_index(row + 1, col), back_index(row, col), "proxy_material"))
                        faces.append((front_index(row + 1, col), back_index(row, col), front_index(row, col), "proxy_material"))
                    if right_outside:
                        faces.append((front_index(row, col), back_index(row, col), back_index(row + 1, col), "proxy_material"))
                        faces.append((front_index(row, col), back_index(row + 1, col), front_index(row + 1, col), "proxy_material"))

        return vertices, normals, uv, faces

    @staticmethod
    def write_obj(
        path: Path,
        vertices: Tensor,
        normals: Tensor,
        uv: Tensor,
        faces: list[MeshFace],
        *,
        material_name: str = "face_material",
        mtl_filename: str = "face_material.mtl",
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"mtllib {mtl_filename}", f"usemtl {material_name}"]

        vertices_flat = vertices.reshape(-1, 3).detach().cpu().tolist()
        normals_flat = normals.reshape(-1, 3).detach().cpu().tolist()
        uv_flat = uv.reshape(-1, 2).detach().cpu().tolist()

        for vertex in vertices_flat:
            lines.append("v {:.6f} {:.6f} {:.6f}".format(*vertex))
        for texcoord in uv_flat:
            lines.append("vt {:.6f} {:.6f}".format(*texcoord))
        for normal in normals_flat:
            lines.append("vn {:.6f} {:.6f} {:.6f}".format(*normal))
        def face_material_name(face: MeshFace) -> str:
            return face[3] if len(face) == 4 else material_name

        ordered_faces = sorted(faces, key=lambda face: face_material_name(face))
        current_material = material_name
        for face in ordered_faces:
            if len(face) == 4:
                index0, index1, index2, face_material = face
                if face_material != current_material:
                    lines.append(f"usemtl {face_material}")
                    current_material = face_material
            else:
                index0, index1, index2 = face
            lines.append(f"f {index0}/{index0}/{index0} {index1}/{index1}/{index1} {index2}/{index2}/{index2}")

        path.write_text("\n".join(lines) + "\n")
        return path

    @staticmethod
    def write_mtl(path: Path, albedo_filename: str, proxy_color: tuple[float, float, float] = (0.58, 0.46, 0.38)) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "newmtl face_material",
            "Ka 1.000000 1.000000 1.000000",
            "Kd 1.000000 1.000000 1.000000",
            "Ks 0.000000 0.000000 0.000000",
            f"map_Kd {albedo_filename}",
            "",
            "newmtl proxy_material",
            "Ka {:.6f} {:.6f} {:.6f}".format(*proxy_color),
            "Kd {:.6f} {:.6f} {:.6f}".format(*proxy_color),
            "Ks 0.000000 0.000000 0.000000",
        ]
        path.write_text("\n".join(lines) + "\n")
        return path

    @staticmethod
    def write_scene_xml(
        path: Path,
        *,
        mesh_filename: str,
        albedo_filename: str,
        environment_filename: str | None,
        config: MitsubaSceneConfig,
        view: ViewTransform | None = None,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        view = view or ViewTransform()
        yaw = math.radians(view.yaw_degrees)
        pitch = math.radians(view.pitch_degrees)
        distance = config.camera_distance
        camera_x = distance * math.sin(yaw) * math.cos(pitch)
        camera_y = distance * math.sin(pitch)
        camera_z = distance * math.cos(yaw) * math.cos(pitch)

        xml_lines = [
            '<scene version="3.0.0">',
            '  <integrator type="path"/>',
            '  <sensor type="perspective">',
            f'    <float name="fov" value="{config.sensor_fov_degrees:.4f}"/>',
            '    <transform name="to_world">',
            f'      <lookat origin="{camera_x:.6f},{camera_y:.6f},{camera_z:.6f}" target="0,0,0" up="0,1,0"/>',
            '    </transform>',
            '    <sampler type="independent">',
            f'      <integer name="sample_count" value="{config.samples_per_pixel}"/>',
            '    </sampler>',
            '    <film type="hdrfilm">',
            f'      <integer name="width" value="{config.film_resolution[1]}"/>',
            f'      <integer name="height" value="{config.film_resolution[0]}"/>',
            '      <rfilter type="tent"/>',
            '    </film>',
            '  </sensor>',
        ]

        if environment_filename is not None:
            xml_lines.extend(
                [
                    '  <emitter type="envmap">',
                    f'    <string name="filename" value="{environment_filename}"/>',
                    "  </emitter>",
                ]
            )

        xml_lines.extend(
            [
                '  <shape type="obj">',
                f'    <string name="filename" value="{mesh_filename}"/>',
                '    <bsdf type="diffuse">',
                '      <texture name="reflectance" type="bitmap">',
                f'        <string name="filename" value="{albedo_filename}"/>',
                '      </texture>',
                '    </bsdf>',
                '  </shape>',
                '</scene>',
            ]
        )

        path.write_text("\n".join(xml_lines) + "\n")
        return path

    def export(
        self,
        depth: Tensor,
        albedo: Tensor,
        lighting_coefficients: Tensor | None,
        *,
        mask: Tensor | None = None,
        view: ViewTransform | None = None,
        output_dir: str | Path | None = None,
        scene_name: str = "face_scene",
        environment_map: str | Path | None = None,
    ) -> MitsubaSceneBundle:
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        depth_sample = _first_sample(depth)
        albedo_sample = _first_sample(albedo)
        mask_sample = _as_mask(mask, depth_sample.shape[-2], depth_sample.shape[-1], depth_sample.device, depth_sample.dtype)
        vertices, normals, uv, faces = self.build_mesh(depth_sample.unsqueeze(0), mask=mask_sample)

        mesh_obj = output_dir / f"{scene_name}.obj"
        albedo_texture = output_dir / f"{scene_name}_albedo.png"
        material_path = output_dir / f"{scene_name}.mtl"
        scene_xml = output_dir / f"{scene_name}.xml"
        metadata_path = output_dir / f"{scene_name}_metadata.json"

        texture_to_save = _remove_texture_background(albedo_sample, mask_sample)
        if self.config.texture_gamma_correction:
            texture_to_save = texture_to_save.clamp(0.0, 1.0).pow(1.0 / 2.2)
        save_image_tensor(albedo_texture, texture_to_save)
        self.write_mtl(material_path, albedo_texture.name, self.config.proxy_material_color)
        self.write_obj(mesh_obj, vertices, normals, uv, faces, mtl_filename=material_path.name)

        if environment_map is not None:
            environment_path = Path(environment_map)
            if environment_path.resolve() != (output_dir / environment_path.name).resolve():
                copied_path = output_dir / environment_path.name
                shutil.copy2(environment_path, copied_path)
                environment_path = copied_path
        elif lighting_coefficients is not None and self.config.use_generated_environment_map:
            environment_path = output_dir / f"{scene_name}_envmap.png"
            envmap = _lighting_to_environment_map(lighting_coefficients, self.config.environment_map_resolution)
            save_image_tensor(environment_path, envmap)
        elif any(channel != 0.0 for channel in self.config.background_color):
            environment_path = output_dir / f"{scene_name}_background.png"
            background = torch.tensor(self.config.background_color, device=albedo_sample.device, dtype=albedo_sample.dtype).view(3, 1, 1)
            background = background.expand(3, *self.config.environment_map_resolution)
            save_image_tensor(environment_path, background)
        else:
            environment_path = None

        self.write_scene_xml(
            scene_xml,
            mesh_filename=mesh_obj.name,
            albedo_filename=albedo_texture.name,
            environment_filename=None if environment_path is None else environment_path.name,
            config=self.config,
            view=view,
        )

        metadata = {
            "scene_name": scene_name,
            "mesh_obj": mesh_obj.name,
            "albedo_texture": albedo_texture.name,
            "material_path": material_path.name,
            "environment_map": None if environment_path is None else environment_path.name,
            "samples_per_pixel": self.config.samples_per_pixel,
            "camera_distance": self.config.camera_distance,
            "sensor_fov_degrees": self.config.sensor_fov_degrees,
            "view": None if view is None else {
                "yaw_degrees": view.yaw_degrees,
                "pitch_degrees": view.pitch_degrees,
                "roll_degrees": view.roll_degrees,
                "depth_parallax": view.depth_parallax,
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

        return MitsubaSceneBundle(
            scene_xml=scene_xml,
            mesh_obj=mesh_obj,
            albedo_texture=albedo_texture,
            environment_map=environment_path,
            metadata_path=metadata_path,
            output_dir=output_dir,
            samples_per_pixel=self.config.samples_per_pixel,
            metadata=metadata,
        )


class MitsubaSceneTranslator(DepthMeshExporter):
    """Compatibility wrapper that exposes the exporter under a translation-style API."""

    def translate(
        self,
        depth: Tensor,
        albedo: Tensor,
        lighting_coefficients: Tensor | None,
        *,
        mask: Tensor | None = None,
        view: ViewTransform | None = None,
        output_dir: str | Path | None = None,
        scene_name: str = "face_scene",
        environment_map: str | Path | None = None,
    ) -> MitsubaSceneBundle:
        return self.export(
            depth,
            albedo,
            lighting_coefficients,
            mask=mask,
            view=view,
            output_dir=output_dir,
            scene_name=scene_name,
            environment_map=environment_map,
        )
