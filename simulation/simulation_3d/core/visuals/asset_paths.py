"""Resolve bundled 3D asset paths for Ursina."""

from __future__ import annotations

import os
from pathlib import Path

from core.paths import SIM_ROOT

ASSETS_ROOT = os.path.join(SIM_ROOT, "assets")


def assets_root() -> str:
    return ASSETS_ROOT


def asset_file(*parts: str) -> str:
    """Absolute path to a file under ``simulation_3d/assets``."""
    return os.path.join(ASSETS_ROOT, *parts)


def model_relative(*parts: str) -> str:
    """Model path relative to ``simulation_3d/assets`` (for Ursina Entity.model)."""
    return "/".join(parts)


def model_path(*parts: str) -> Path:
    """Absolute path to a bundled model file."""
    return Path(asset_file(*parts))


def model_exists(*parts: str) -> bool:
    return os.path.isfile(asset_file(*parts))


def configure_ursina_assets() -> Path:
    """
    Point Ursina's asset folder at ``simulation_3d/assets``.

    Must run after ``Ursina()`` is constructed. Ursina 7+ requires a ``Path``,
    not a plain string (otherwise ``load_model`` crashes on ``.glob()``).
    """
    from ursina import application

    folder = Path(ASSETS_ROOT).resolve()
    application.asset_folder = folder
    return folder


def builtin_ursina_texture(stem: str) -> Path | None:
    """Locate a built-in Ursina texture inside the installed package."""
    from ursina import application

    package = Path(application.package_folder)
    for rel in (
        f"textures/{stem}.jpg",
        f"textures/{stem}.png",
        f"assets/Textures/{stem}.jpg",
        f"assets/Textures/{stem}.png",
    ):
        candidate = package / rel
        if candidate.is_file():
            return candidate
    return None
