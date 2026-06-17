#!/usr/bin/env python3
"""Download CC0 3D models from OpenGameArt into assets/models/."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import zipfile
from urllib.request import urlretrieve

SIM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(SIM_ROOT, "assets")
MODELS = os.path.join(ASSETS, "models")

DOWNLOADS = {
    "kenney_car-kit.zip": "https://opengameart.org/sites/default/files/kenney_car-kit_3.1.zip",
    "mars_rover.zip": "https://opengameart.org/sites/default/files/001_mars_rover.zip",
    "nature_crops_pack.zip": "https://opengameart.org/sites/default/files/nature_crops_pack_by_quaternius.zip",
    "corn.obj": "https://opengameart.org/sites/default/files/corn.obj",
    "textured_stylized_trees.zip": "https://opengameart.org/sites/default/files/textured_stylized_trees_-_may_2020.zip",
}

TREE_VARIANTS = (
    "Tree_1",
    "Tree_2",
    "Tree_3",
    "Tree_4",
    "Tree_5",
    "Birch_1",
    "Birch_2",
    "Pine_1",
    "Pine_2",
)

TREE_TEXTURES = (
    "Tree_Bark.jpg",
    "Tree_Leaves.png",
    "Birch_Bark.png",
    "Birch_Leaves_Green.png",
    "Pine_Leaves.png",
)

MTL_TEXTURE_MAP = {
    "Bark": "../Textures/Tree_Bark.jpg",
    "Tree_Leaves": "../Textures/Tree_Leaves.png",
    "Birch_Bark": "../Textures/Birch_Bark.png",
    "Birch_Leaves": "../Textures/Birch_Leaves_Green.png",
    "Pine_Leaves": "../Textures/Pine_Leaves.png",
}


def _copy(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  -> {os.path.relpath(dst, SIM_ROOT)}")


def _patch_tree_mtl(mtl_path: str):
    with open(mtl_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    out: list[str] = []
    current_material: str | None = None
    patched: set[str] = set()

    for line in lines:
        if line.startswith("newmtl "):
            current_material = line.split(maxsplit=1)[1].strip()
            out.append(line)
            continue
        if line.startswith("map_Kd ") and current_material in MTL_TEXTURE_MAP:
            continue
        out.append(line)
        if (
            current_material in MTL_TEXTURE_MAP
            and current_material not in patched
            and line.startswith("illum ")
        ):
            out.append(f"map_Kd {MTL_TEXTURE_MAP[current_material]}")
            patched.add(current_material)

    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def _install_quaternius_trees(tmp: str):
    trees_zip = os.path.join(tmp, "textured_stylized_trees.zip")
    extract_root = os.path.join(tmp, "quaternius_trees")
    with zipfile.ZipFile(trees_zip) as zf:
        zf.extractall(extract_root)

    pack_root = None
    for root, dirs, files in os.walk(extract_root):
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

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from triangulate_obj import triangulate_obj_file
    from split_tree_obj import split_tree_obj_file

    for variant in TREE_VARIANTS:
        src_obj = os.path.join(obj_src, f"{variant}.obj")
        src_mtl = os.path.join(obj_src, f"{variant}.mtl")
        dst_obj = os.path.join(tree_dst, f"{variant}.obj")
        dst_mtl = os.path.join(tree_dst, f"{variant}.mtl")
        triangulate_obj_file(src_obj, dst_obj)
        _copy(src_mtl, dst_mtl)
        _patch_tree_mtl(dst_mtl)
        split_tree_obj_file(
            dst_obj,
            os.path.join(tree_dst, f"{variant}_bark.obj"),
            os.path.join(tree_dst, f"{variant}_leaves.obj"),
        )
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

        car_zip = os.path.join(tmp, "kenney_car-kit.zip")
        with zipfile.ZipFile(car_zip) as zf:
            zf.extractall(os.path.join(tmp, "kenney"))
        car_tex = os.path.join(tmp, "kenney", "Models", "OBJ format", "Textures", "colormap.png")
        kenney_obj = os.path.join(tmp, "kenney", "Models", "OBJ format")
        for stem in ("tractor", "tractor-shovel", "delivery"):
            _copy(os.path.join(kenney_obj, f"{stem}.obj"), os.path.join(MODELS, "robot", f"{stem.replace('-', '_')}.obj"))
            _copy(os.path.join(kenney_obj, f"{stem}.mtl"), os.path.join(MODELS, "robot", f"{stem.replace('-', '_')}.mtl"))
        _copy(car_tex, os.path.join(MODELS, "robot", "Textures", "colormap.png"))
        license_src = os.path.join(tmp, "kenney", "License.txt")
        if os.path.isfile(license_src):
            _copy(license_src, os.path.join(MODELS, "robot", "License_kenney.txt"))

        rover_zip = os.path.join(tmp, "mars_rover.zip")
        with zipfile.ZipFile(rover_zip) as zf:
            zf.extractall(os.path.join(tmp, "rover"))
        rover_obj = os.path.join(tmp, "rover", "rover_mesh.obj")
        _copy(rover_obj, os.path.join(MODELS, "robot", "mars_rover.obj"))

        crops_zip = os.path.join(tmp, "nature_crops_pack.zip")
        with zipfile.ZipFile(crops_zip) as zf:
            zf.extractall(os.path.join(tmp, "crops"))
        crop_root = os.path.join(tmp, "crops", "Nature Crops Pack - Jan 2020", "OBJ")
        _copy(os.path.join(crop_root, "Corn_4.obj"), os.path.join(MODELS, "corn", "Corn_4.obj"))
        _copy(os.path.join(crop_root, "Corn_4.mtl"), os.path.join(MODELS, "corn", "Corn_4.mtl"))
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from triangulate_obj import triangulate_obj_file

        triangulate_obj_file(
            os.path.join(MODELS, "corn", "Corn_4.obj"),
            os.path.join(MODELS, "corn", "Corn_4_ursina.obj"),
        )
        print("  -> assets/models/corn/Corn_4_ursina.obj (triangulated for Ursina)")

        _copy(os.path.join(tmp, "corn.obj"), os.path.join(MODELS, "corn", "corn.obj"))

        print("Installing Quaternius stylized trees ...")
        _install_quaternius_trees(tmp)

    print("Done. Assets ready in assets/models/")


if __name__ == "__main__":
    sys.exit(main())
