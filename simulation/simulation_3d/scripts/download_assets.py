#!/usr/bin/env python3
"""Download CC0 robot and corn models from OpenGameArt into assets/models/."""

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
}


def _copy(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  -> {os.path.relpath(dst, SIM_ROOT)}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        for name, url in DOWNLOADS.items():
            path = os.path.join(tmp, name)
            print(f"Downloading {name} ...")
            urlretrieve(url, path)

        car_zip = os.path.join(tmp, "kenney_car-kit.zip")
        with zipfile.ZipFile(car_zip) as zf:
            zf.extractall(os.path.join(tmp, "kenney"))
        car_obj = os.path.join(tmp, "kenney", "Models", "OBJ format", "tractor.obj")
        car_mtl = os.path.join(tmp, "kenney", "Models", "OBJ format", "tractor.mtl")
        car_tex = os.path.join(tmp, "kenney", "Models", "OBJ format", "Textures", "colormap.png")
        _copy(car_obj, os.path.join(MODELS, "robot", "tractor.obj"))
        _copy(car_mtl, os.path.join(MODELS, "robot", "tractor.mtl"))
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

    print("Done. Assets ready in assets/models/")


if __name__ == "__main__":
    sys.exit(main())
