#!/usr/bin/env python3
"""Download and prepare CC0 3D models for the Ursina simulation (run once if assets are missing)."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import zipfile
from urllib.request import urlretrieve

SIM_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(SIM_ROOT, "assets", "models")

DOWNLOADS = {
    "kenney_car-kit.zip": "https://opengameart.org/sites/default/files/kenney_car-kit_3.1.zip",
    "mars_rover.zip": "https://opengameart.org/sites/default/files/001_mars_rover.zip",
    "textured_stylized_trees.zip": "https://opengameart.org/sites/default/files/textured_stylized_trees_-_may_2020.zip",
}

TREE_VARIANTS = (
    "Tree_1", "Tree_2", "Tree_3", "Tree_4", "Tree_5",
    "Birch_1", "Birch_2", "Pine_1", "Pine_2",
)

TREE_TEXTURES = (
    "Tree_Bark.jpg", "Tree_Leaves.png",
    "Birch_Bark.png", "Birch_Leaves_Green.png", "Pine_Leaves.png",
)

BARK_MATERIALS = {"Bark", "Birch_Bark"}
LEAF_MATERIALS = {"Tree_Leaves", "Birch_Leaves", "Pine_Leaves"}
Corner = tuple[int, int | None, int | None]


def _copy(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  -> {os.path.relpath(dst, SIM_ROOT)}")


def _triangulate_obj_text(text: str) -> str:
    out_lines: list[str] = []
    for line in text.splitlines():
        if not line.startswith("f "):
            out_lines.append(line)
            continue
        verts = [tok.split("/")[0] for tok in line.strip().split()[1:]]
        if len(verts) < 3:
            continue
        if len(verts) == 3:
            out_lines.append("f " + " ".join(verts))
            continue
        anchor = verts[0]
        for i in range(1, len(verts) - 1):
            out_lines.append(f"f {anchor} {verts[i]} {verts[i + 1]}")
    return "\n".join(out_lines) + "\n"


def _triangulate_obj_file(src: str, dst: str) -> None:
    with open(src, "r", encoding="utf-8") as f:
        text = f.read()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(_triangulate_obj_text(text))


def _parse_corner(token: str) -> Corner:
    parts = token.split("/")
    vi = int(parts[0]) - 1
    vti = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else vi
    vni = int(parts[2]) - 1 if len(parts) > 2 and parts[2] else vi
    return vi, vti, vni


def _triangulate_corners(corners: list[Corner]) -> list[tuple[Corner, Corner, Corner]]:
    if len(corners) < 3:
        return []
    if len(corners) == 3:
        return [(corners[0], corners[1], corners[2])]
    return [(corners[0], corners[i], corners[i + 1]) for i in range(1, len(corners) - 1)]


def _write_submesh(
    dst: str,
    vertices: list[tuple[float, float, float]],
    uvs: list[tuple[float, float]],
    normals: list[tuple[float, float, float]],
    corner_tris: list[tuple[Corner, Corner, Corner]],
    label: str,
) -> bool:
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
        out_vt.append(uvs[vti] if 0 <= vti < len(uvs) else (0.0, 0.0))
        out_vn.append(normals[vni] if 0 <= vni < len(normals) else (0.0, 1.0, 0.0))
        return idx

    for a, b, c in corner_tris:
        ia, ib, ic = map_corner(a), map_corner(b), map_corner(c)
        out_faces.append(f"f {ia}/{ia}/{ia} {ib}/{ib}/{ib} {ic}/{ic}/{ic}")

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(f"# Split {label} mesh for Ursina textured rendering\no {label}\n")
        for x, y, z in out_v:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in out_vt:
            f.write(f"vt {u:.6f} {v:.6f}\n")
        for nx, ny, nz in out_vn:
            f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
        f.writelines(face + "\n" for face in out_faces)
    return True


def _split_tree_obj(src: str, bark_dst: str, leaves_dst: str) -> tuple[bool, bool]:
    vertices: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    normals: list[tuple[float, float, float]] = []
    bark_tris: list[tuple[Corner, Corner, Corner]] = []
    leaf_tris: list[tuple[Corner, Corner, Corner]] = []
    bucket: str | None = None

    with open(src, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                p = line.split()
                vertices.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("vt "):
                p = line.split()
                uvs.append((float(p[1]), float(p[2])))
            elif line.startswith("vn "):
                p = line.split()
                normals.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("usemtl "):
                mat = line.split(maxsplit=1)[1].strip()
                if mat in BARK_MATERIALS:
                    bucket = "bark"
                elif mat in LEAF_MATERIALS:
                    bucket = "leaves"
                else:
                    bucket = None
            elif line.startswith("f ") and bucket:
                corners = [_parse_corner(tok) for tok in line.split()[1:]]
                tris = _triangulate_corners(corners)
                (bark_tris if bucket == "bark" else leaf_tris).extend(tris)

    stem = os.path.splitext(os.path.basename(src))[0]
    return (
        _write_submesh(bark_dst, vertices, uvs, normals, bark_tris, f"{stem}_bark"),
        _write_submesh(leaves_dst, vertices, uvs, normals, leaf_tris, f"{stem}_leaves"),
    )


def _install_quaternius_trees(tmp: str):
    with zipfile.ZipFile(os.path.join(tmp, "textured_stylized_trees.zip")) as zf:
        extract_root = os.path.join(tmp, "quaternius_trees")
        zf.extractall(extract_root)

    pack_root = None
    for root, dirs, _files in os.walk(extract_root):
        if "OBJ" in dirs and "Textures" in dirs:
            pack_root = root
            break
    if pack_root is None:
        raise FileNotFoundError("Could not locate Quaternius tree pack layout in archive.")

    obj_src = os.path.join(pack_root, "OBJ")
    tex_src = os.path.join(pack_root, "Textures")
    tree_dst = os.path.join(MODELS, "trees", "quaternius")
    tex_dst = os.path.join(MODELS, "trees", "Textures")

    for tex_name in TREE_TEXTURES:
        _copy(os.path.join(tex_src, tex_name), os.path.join(tex_dst, tex_name))

    for variant in TREE_VARIANTS:
        src_obj = os.path.join(obj_src, f"{variant}.obj")
        tri_obj = os.path.join(tree_dst, f"{variant}.obj")
        _triangulate_obj_file(src_obj, tri_obj)
        bark, leaves = _split_tree_obj(
            tri_obj,
            os.path.join(tree_dst, f"{variant}_bark.obj"),
            os.path.join(tree_dst, f"{variant}_leaves.obj"),
        )
        if not (bark and leaves):
            raise RuntimeError(f"Failed to split tree mesh for {variant}")
        if os.path.isfile(tri_obj):
            os.remove(tri_obj)
        print(f"  -> assets/models/trees/quaternius/{variant}_{{bark,leaves}}.obj")

    license_note = os.path.join(MODELS, "trees", "License_quaternius_trees.txt")
    with open(license_note, "w", encoding="utf-8") as f:
        f.write(
            "Textured Stylized Trees (May 2020) by Quaternius\n"
            "https://opengameart.org/content/lowpoly-textured-trees\n"
            "CC0 1.0 — public domain\n"
        )
    print(f"  -> {os.path.relpath(license_note, SIM_ROOT)}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        for name, url in DOWNLOADS.items():
            path = os.path.join(tmp, name)
            print(f"Downloading {name} ...")
            urlretrieve(url, path)

        with zipfile.ZipFile(os.path.join(tmp, "kenney_car-kit.zip")) as zf:
            zf.extractall(os.path.join(tmp, "kenney"))
        kenney_obj = os.path.join(tmp, "kenney", "Models", "OBJ format")
        car_tex = os.path.join(kenney_obj, "Textures", "colormap.png")
        for stem in ("tractor", "tractor-shovel", "delivery"):
            out = stem.replace("-", "_")
            _copy(os.path.join(kenney_obj, f"{stem}.obj"), os.path.join(MODELS, "robot", f"{out}.obj"))
            _copy(os.path.join(kenney_obj, f"{stem}.mtl"), os.path.join(MODELS, "robot", f"{out}.mtl"))
        _copy(car_tex, os.path.join(MODELS, "robot", "Textures", "colormap.png"))
        license_src = os.path.join(tmp, "kenney", "License.txt")
        if os.path.isfile(license_src):
            _copy(license_src, os.path.join(MODELS, "robot", "License_kenney.txt"))

        with zipfile.ZipFile(os.path.join(tmp, "mars_rover.zip")) as zf:
            zf.extractall(os.path.join(tmp, "rover"))
        _copy(os.path.join(tmp, "rover", "rover_mesh.obj"), os.path.join(MODELS, "robot", "mars_rover.obj"))

        print("Installing Quaternius stylized trees ...")
        _install_quaternius_trees(tmp)

    print("Done. Assets ready in assets/models/")


if __name__ == "__main__":
    sys.exit(main())
