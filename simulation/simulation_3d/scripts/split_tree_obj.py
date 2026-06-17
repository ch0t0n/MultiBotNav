#!/usr/bin/env python3
"""Split Quaternius tree OBJ files into bark and leaf meshes for Ursina textures."""

from __future__ import annotations

import argparse
import os
import sys

BARK_MATERIALS = {"Bark", "Birch_Bark"}
LEAF_MATERIALS = {"Tree_Leaves", "Birch_Leaves", "Pine_Leaves"}

Corner = tuple[int, int | None, int | None]


def _parse_corner(token: str) -> Corner:
    parts = token.split("/")
    vi = int(parts[0]) - 1
    # Quaternius Blender exports often use vtx-only faces (f i j k). In that case
    # the vertex index doubles as the vtx/vn index (same convention as Ursina's OBJ loader).
    vti = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else vi
    vni = int(parts[2]) - 1 if len(parts) > 2 and parts[2] else vi
    return vi, vti, vni


def _triangulate_corners(corners: list[Corner]) -> list[tuple[Corner, Corner, Corner]]:
    if len(corners) < 3:
        return []
    if len(corners) == 3:
        return [(corners[0], corners[1], corners[2])]
    tris: list[tuple[Corner, Corner, Corner]] = []
    for i in range(1, len(corners) - 1):
        tris.append((corners[0], corners[i], corners[i + 1]))
    return tris


def _write_submesh(
    dst: str,
    vertices: list[tuple[float, float, float]],
    uvs: list[tuple[float, float]],
    normals: list[tuple[float, float, float]],
    corner_tris: list[tuple[Corner, Corner, Corner]],
    label: str,
):
    if not corner_tris:
        return False

    index_map: dict[Corner, int] = {}
    out_v: list[tuple[float, float, float]] = []
    out_vt: list[tuple[float, float]] = []
    out_vn: list[tuple[float, float, float]] = []
    out_faces: list[str] = []

    def map_corner(corner: Corner) -> int:
        if corner in index_map:
            return index_map[corner]
        vi, vti, vni = corner
        idx = len(out_v) + 1
        index_map[corner] = idx
        out_v.append(vertices[vi])
        if 0 <= vti < len(uvs):
            out_vt.append(uvs[vti])
        else:
            out_vt.append((0.0, 0.0))
        if 0 <= vni < len(normals):
            out_vn.append(normals[vni])
        else:
            out_vn.append((0.0, 1.0, 0.0))
        return idx

    for a, b, c in corner_tris:
        ia, ib, ic = map_corner(a), map_corner(b), map_corner(c)
        out_faces.append(f"f {ia}/{ia}/{ia} {ib}/{ib}/{ib} {ic}/{ic}/{ic}")

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(f"# Split {label} mesh for Ursina textured rendering\n")
        f.write(f"o {label}\n")
        for x, y, z in out_v:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in out_vt:
            f.write(f"vt {u:.6f} {v:.6f}\n")
        for nx, ny, nz in out_vn:
            f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
        for face in out_faces:
            f.write(face + "\n")
    return True


def split_tree_obj_file(src: str, bark_dst: str, leaves_dst: str) -> tuple[bool, bool]:
    vertices: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    normals: list[tuple[float, float, float]] = []
    bark_tris: list[tuple[Corner, Corner, Corner]] = []
    leaf_tris: list[tuple[Corner, Corner, Corner]] = []
    current_bucket: str | None = None

    with open(src, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                continue
            if line.startswith("vt "):
                parts = line.split()
                uvs.append((float(parts[1]), float(parts[2])))
                continue
            if line.startswith("vn "):
                parts = line.split()
                normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
                continue
            if line.startswith("usemtl "):
                material = line.split(maxsplit=1)[1].strip()
                if material in BARK_MATERIALS:
                    current_bucket = "bark"
                elif material in LEAF_MATERIALS:
                    current_bucket = "leaves"
                else:
                    current_bucket = None
                continue
            if not line.startswith("f "):
                continue
            if current_bucket is None:
                continue
            corners = [_parse_corner(tok) for tok in line.split()[1:]]
            for tri in _triangulate_corners(corners):
                if current_bucket == "bark":
                    bark_tris.append(tri)
                else:
                    leaf_tris.append(tri)

    stem = os.path.splitext(os.path.basename(src))[0]
    wrote_bark = _write_submesh(bark_dst, vertices, uvs, normals, bark_tris, f"{stem}_bark")
    wrote_leaves = _write_submesh(
        leaves_dst, vertices, uvs, normals, leaf_tris, f"{stem}_leaves"
    )
    return wrote_bark, wrote_leaves


def main() -> int:
    parser = argparse.ArgumentParser(description="Split tree OBJ into bark and leaf parts.")
    parser.add_argument("src")
    parser.add_argument("bark_dst", nargs="?")
    parser.add_argument("leaves_dst", nargs="?")
    args = parser.parse_args()
    src = args.src
    base, _ = os.path.splitext(src)
    bark_dst = args.bark_dst or f"{base}_bark.obj"
    leaves_dst = args.leaves_dst or f"{base}_leaves.obj"
    bark, leaves = split_tree_obj_file(src, bark_dst, leaves_dst)
    print(f"Wrote bark={bark} leaves={leaves}: {bark_dst}, {leaves_dst}")
    return 0 if bark and leaves else 1


if __name__ == "__main__":
    sys.exit(main())
