"""Offline procedural clay bust mesh export."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class BustTemplateConfig:
    """Controls the generic clay bust template."""

    output_dir: Path = field(default_factory=lambda: Path("outputs/bust_template"))
    radial_segments: int = 72
    vertical_segments: int = 48
    material_color: tuple[float, float, float] = (0.50, 0.51, 0.49)
    scene_name: str = "clay_bust"

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data


@dataclass(slots=True)
class BustTemplateBundle:
    """Files generated for the procedural bust."""

    mesh_obj: Path
    material_mtl: Path
    metadata_path: Path
    output_dir: Path


class ProceduralBustTemplate:
    """Create a neutral head, neck, shoulder, ear, and hair-cap bust mesh."""

    def __init__(self, config: BustTemplateConfig | None = None) -> None:
        self.config = config or BustTemplateConfig()

    @staticmethod
    def _add_vertex(
        vertices: list[tuple[float, float, float]],
        normals: list[tuple[float, float, float]],
        vertex: tuple[float, float, float],
        normal: tuple[float, float, float],
    ) -> int:
        vertices.append(vertex)
        length = math.sqrt(sum(value * value for value in normal)) or 1.0
        normals.append(tuple(value / length for value in normal))
        return len(vertices)

    def _add_ellipsoid(
        self,
        vertices: list[tuple[float, float, float]],
        normals: list[tuple[float, float, float]],
        faces: list[tuple[int, int, int]],
        *,
        center: tuple[float, float, float],
        radius: tuple[float, float, float],
        theta_min: float = 0.0,
        theta_max: float = math.pi,
        phi_min: float = -math.pi,
        phi_max: float = math.pi,
    ) -> None:
        rows = max(4, self.config.vertical_segments)
        cols = max(8, self.config.radial_segments)
        indices: list[list[int]] = []
        for row in range(rows + 1):
            theta = theta_min + (theta_max - theta_min) * row / rows
            row_indices = []
            for col in range(cols + 1):
                phi = phi_min + (phi_max - phi_min) * col / cols
                unit_x = math.sin(theta) * math.cos(phi)
                unit_y = math.cos(theta)
                unit_z = math.sin(theta) * math.sin(phi)
                vertex = (
                    center[0] + radius[0] * unit_x,
                    center[1] + radius[1] * unit_y,
                    center[2] + radius[2] * unit_z,
                )
                normal = (unit_x / radius[0], unit_y / radius[1], unit_z / radius[2])
                row_indices.append(self._add_vertex(vertices, normals, vertex, normal))
            indices.append(row_indices)

        for row in range(rows):
            for col in range(cols):
                index00 = indices[row][col]
                index10 = indices[row][col + 1]
                index01 = indices[row + 1][col]
                index11 = indices[row + 1][col + 1]
                faces.append((index00, index01, index11))
                faces.append((index00, index11, index10))

    def _add_cylinder(
        self,
        vertices: list[tuple[float, float, float]],
        normals: list[tuple[float, float, float]],
        faces: list[tuple[int, int, int]],
        *,
        center: tuple[float, float, float],
        radius_x: float,
        radius_z: float,
        height: float,
        rows: int,
    ) -> None:
        cols = max(8, self.config.radial_segments)
        indices: list[list[int]] = []
        for row in range(rows + 1):
            y = center[1] - height * 0.5 + height * row / rows
            neck_taper = 0.88 + 0.12 * row / rows
            row_indices = []
            for col in range(cols + 1):
                phi = -math.pi + 2.0 * math.pi * col / cols
                cos_phi = math.cos(phi)
                sin_phi = math.sin(phi)
                vertex = (
                    center[0] + radius_x * neck_taper * cos_phi,
                    y,
                    center[2] + radius_z * neck_taper * sin_phi,
                )
                row_indices.append(self._add_vertex(vertices, normals, vertex, (cos_phi, 0.15, sin_phi)))
            indices.append(row_indices)

        for row in range(rows):
            for col in range(cols):
                index00 = indices[row][col]
                index10 = indices[row][col + 1]
                index01 = indices[row + 1][col]
                index11 = indices[row + 1][col + 1]
                faces.append((index00, index01, index11))
                faces.append((index00, index11, index10))

    def _add_feature_ellipsoid(
        self,
        vertices: list[tuple[float, float, float]],
        normals: list[tuple[float, float, float]],
        faces: list[tuple[int, int, int]],
        center: tuple[float, float, float],
        radius: tuple[float, float, float],
    ) -> None:
        old_vertical = self.config.vertical_segments
        old_radial = self.config.radial_segments
        self.config.vertical_segments = max(8, old_vertical // 3)
        self.config.radial_segments = max(16, old_radial // 3)
        self._add_ellipsoid(vertices, normals, faces, center=center, radius=radius)
        self.config.vertical_segments = old_vertical
        self.config.radial_segments = old_radial

    def _add_face_relief(
        self,
        vertices: list[tuple[float, float, float]],
        normals: list[tuple[float, float, float]],
        faces: list[tuple[int, int, int]],
    ) -> None:
        """Add a continuous sculpted facial surface over the front of the head."""

        rows = 58
        cols = 42
        indices: list[list[int]] = []

        def bump(x: float, y: float, cx: float, cy: float, sx: float, sy: float, amount: float) -> float:
            return amount * math.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))

        for row in range(rows + 1):
            y = 0.50 + 1.00 * row / rows
            y_norm = (y - 1.02) / 0.72
            half_width = 0.43 * max(0.2, math.sqrt(max(0.0, 1.0 - 0.55 * y_norm * y_norm)))
            row_indices = []
            for col in range(cols + 1):
                u = -1.0 + 2.0 * col / cols
                x = half_width * u
                head_term = 1.0 - (x / 0.57) ** 2 - ((y - 1.10) / 0.78) ** 2
                base_z = 0.49 * math.sqrt(max(0.0, head_term))

                z = base_z
                z += bump(x, y, 0.0, 1.11, 0.095, 0.24, 0.19)  # nose bridge and tip
                z += bump(x, y, 0.0, 0.93, 0.13, 0.07, 0.07)  # philtrum
                z += bump(x, y, -0.20, 1.20, 0.14, 0.07, 0.045)  # left brow
                z += bump(x, y, 0.20, 1.20, 0.14, 0.07, 0.045)  # right brow
                z += bump(x, y, -0.25, 1.04, 0.15, 0.22, 0.055)  # cheek
                z += bump(x, y, 0.25, 1.04, 0.15, 0.22, 0.055)
                z += bump(x, y, 0.0, 0.72, 0.24, 0.13, 0.035)  # chin
                z -= bump(x, y, -0.19, 1.11, 0.12, 0.065, 0.055)  # eye socket
                z -= bump(x, y, 0.19, 1.11, 0.12, 0.065, 0.055)
                z -= bump(x, y, 0.0, 0.84, 0.20, 0.045, 0.032)  # mouth separation
                z += bump(x, y, 0.0, 0.86, 0.22, 0.035, 0.045)  # upper lip
                z += bump(x, y, 0.0, 0.80, 0.24, 0.045, 0.052)  # lower lip

                row_indices.append(self._add_vertex(vertices, normals, (x, y, z + 0.02), (0.0, 0.08, 1.0)))
            indices.append(row_indices)

        for row in range(rows):
            for col in range(cols):
                index00 = indices[row][col]
                index10 = indices[row][col + 1]
                index01 = indices[row + 1][col]
                index11 = indices[row + 1][col + 1]
                faces.append((index00, index01, index11))
                faces.append((index00, index11, index10))

    def _add_hair_cap(
        self,
        vertices: list[tuple[float, float, float]],
        normals: list[tuple[float, float, float]],
        faces: list[tuple[int, int, int]],
    ) -> None:
        """Add a subtle wavy cap instead of separate cartoon hair blobs."""

        rows = 22
        cols = max(24, self.config.radial_segments)
        indices: list[list[int]] = []
        for row in range(rows + 1):
            theta = 0.05 + (math.pi * 0.58 - 0.05) * row / rows
            row_indices = []
            for col in range(cols + 1):
                phi = -math.pi + 2.0 * math.pi * col / cols
                wave = 0.025 * math.sin(7.0 * phi + 2.0 * row / rows) + 0.018 * math.sin(13.0 * phi)
                rx = 0.60 + wave
                ry = 0.36 + 0.4 * wave
                rz = 0.52 + wave
                unit_x = math.sin(theta) * math.cos(phi)
                unit_y = math.cos(theta)
                unit_z = math.sin(theta) * math.sin(phi)
                y = 1.48 + ry * unit_y
                if y < 1.27 and unit_z > 0.0:
                    y += 0.07 * unit_z
                vertex = (rx * unit_x, y, -0.04 + rz * unit_z)
                row_indices.append(self._add_vertex(vertices, normals, vertex, (unit_x, unit_y, unit_z)))
            indices.append(row_indices)

        for row in range(rows):
            for col in range(cols):
                index00 = indices[row][col]
                index10 = indices[row][col + 1]
                index01 = indices[row + 1][col]
                index11 = indices[row + 1][col + 1]
                faces.append((index00, index01, index11))
                faces.append((index00, index11, index10))

    def build_mesh(self) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]], list[tuple[int, int, int]]]:
        vertices: list[tuple[float, float, float]] = []
        normals: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []

        self._add_ellipsoid(vertices, normals, faces, center=(0.0, 1.08, -0.02), radius=(0.56, 0.76, 0.48))
        self._add_face_relief(vertices, normals, faces)
        self._add_cylinder(vertices, normals, faces, center=(0.0, 0.03, -0.05), radius_x=0.25, radius_z=0.22, height=0.94, rows=20)
        self._add_ellipsoid(
            vertices,
            normals,
            faces,
            center=(0.0, -0.58, -0.05),
            radius=(1.10, 0.32, 0.36),
            theta_min=0.0,
            theta_max=math.pi * 0.68,
        )

        self._add_feature_ellipsoid(vertices, normals, faces, center=(-0.57, 1.08, -0.02), radius=(0.055, 0.17, 0.055))
        self._add_feature_ellipsoid(vertices, normals, faces, center=(0.57, 1.08, -0.02), radius=(0.055, 0.17, 0.055))
        self._add_hair_cap(vertices, normals, faces)

        return vertices, normals, faces

    @staticmethod
    def write_mtl(path: Path, color: tuple[float, float, float]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "newmtl clay_material",
            "Ka {:.6f} {:.6f} {:.6f}".format(*color),
            "Kd {:.6f} {:.6f} {:.6f}".format(*color),
            "Ks 0.035000 0.035000 0.035000",
            "Ns 8.000000",
        ]
        path.write_text("\n".join(lines) + "\n")
        return path

    @staticmethod
    def write_obj(
        path: Path,
        vertices: list[tuple[float, float, float]],
        normals: list[tuple[float, float, float]],
        faces: list[tuple[int, int, int]],
        *,
        mtl_filename: str,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"mtllib {mtl_filename}", "usemtl clay_material"]
        for vertex in vertices:
            lines.append("v {:.6f} {:.6f} {:.6f}".format(*vertex))
        for normal in normals:
            lines.append("vn {:.6f} {:.6f} {:.6f}".format(*normal))
        for face in faces:
            lines.append("f {0}//{0} {1}//{1} {2}//{2}".format(*face))
        path.write_text("\n".join(lines) + "\n")
        return path

    def export(self, output_dir: str | Path | None = None, scene_name: str | None = None) -> BustTemplateBundle:
        output_dir = Path(output_dir or self.config.output_dir)
        scene_name = scene_name or self.config.scene_name
        output_dir.mkdir(parents=True, exist_ok=True)

        mesh_obj = output_dir / f"{scene_name}.obj"
        material_mtl = output_dir / f"{scene_name}.mtl"
        metadata_path = output_dir / f"{scene_name}_metadata.json"

        vertices, normals, faces = self.build_mesh()
        self.write_mtl(material_mtl, self.config.material_color)
        self.write_obj(mesh_obj, vertices, normals, faces, mtl_filename=material_mtl.name)

        metadata = {
            "scene_name": scene_name,
            "mesh_obj": mesh_obj.name,
            "material_mtl": material_mtl.name,
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "config": self.config.to_dict(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        return BustTemplateBundle(
            mesh_obj=mesh_obj,
            material_mtl=material_mtl,
            metadata_path=metadata_path,
            output_dir=output_dir,
        )
