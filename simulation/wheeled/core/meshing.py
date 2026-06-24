"""Build Ursina triangle meshes from 2D polygon footprints."""

from __future__ import annotations

from shapely.geometry import Polygon
from shapely.ops import triangulate

from .geometry import world_to_scene


def extrude_polygon_mesh(
    poly,
    world_width: float,
    world_height: float,
    height: float,
    base_y: float = 0.05,
) -> tuple[list[tuple[float, float, float]], list[int]]:
    """
    Extrude a training-space polygon footprint to a fixed-height 3D prism.

    Returns (vertices, triangles) suitable for ``ursina.Mesh``.
    """
    ring: list[tuple[float, float]] = []
    for vx, vy in poly:
        px, _, pz = world_to_scene(float(vx), float(vy), world_width, world_height)
        ring.append((px, pz))

    n = len(ring)
    if n < 3 or height <= 0:
        return [], []

    vertices: list[tuple[float, float, float]] = []
    index: dict[tuple[float, float, float], int] = {}

    def add_vertex(x: float, y: float, z: float) -> int:
        key = (round(x, 5), round(y, 5), round(z, 5))
        if key not in index:
            index[key] = len(vertices)
            vertices.append((x, y, z))
        return index[key]

    triangles: list[int] = []

    footprint = Polygon(ring)
    for tri in triangulate(footprint):
        if not footprint.contains(tri.representative_point()):
            continue
        coords = list(tri.exterior.coords)[:3]
        bottom = [add_vertex(x, base_y, z) for x, z in coords]
        top = [add_vertex(x, base_y + height, z) for x, z in coords]
        triangles.extend([bottom[0], bottom[2], bottom[1]])
        triangles.extend([top[0], top[1], top[2]])

    for i in range(n):
        j = (i + 1) % n
        x0, z0 = ring[i]
        x1, z1 = ring[j]
        v0 = add_vertex(x0, base_y, z0)
        v1 = add_vertex(x1, base_y, z1)
        v2 = add_vertex(x1, base_y + height, z1)
        v3 = add_vertex(x0, base_y + height, z0)
        triangles.extend([v0, v1, v2, v0, v2, v3])

    return vertices, triangles


def make_lit_mesh(Mesh, vertices, triangles, *, smooth: bool = False):
    """
    Build an Ursina mesh with face normals so directional/ambient lights apply.

    Custom vertex-only meshes default to flat shading without normals; built-in
    primitives and OBJ models include them automatically.
    """
    if not vertices or not triangles:
        return None
    mesh = Mesh(vertices=vertices, triangles=triangles, mode="triangle")
    mesh.generate_normals(smooth=smooth)
    return mesh


def merge_meshes(
    parts: list[tuple[list[tuple[float, float, float]], list[int]]],
) -> tuple[list[tuple[float, float, float]], list[int]]:
    """Concatenate multiple mesh parts into one indexed mesh."""
    vertices: list[tuple[float, float, float]] = []
    triangles: list[int] = []
    offset = 0
    for verts, tris in parts:
        vertices.extend(verts)
        triangles.extend(i + offset for i in tris)
        offset += len(verts)
    return vertices, triangles
