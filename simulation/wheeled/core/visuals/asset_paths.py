"""Resolve bundled 3D asset paths for Ursina."""

from __future__ import annotations

import os
from pathlib import Path

from core.paths import SIM_ROOT

ASSETS_ROOT = os.path.join(SIM_ROOT, "assets")


def assets_root() -> str:
    return ASSETS_ROOT


def asset_file(*parts: str) -> str:
    """Absolute path to a file under ``simulation/wheeled/assets``."""
    return os.path.join(ASSETS_ROOT, *parts)


def model_relative(*parts: str) -> str:
    """Model path relative to ``simulation/wheeled/assets`` (for Ursina Entity.model)."""
    return "/".join(parts)


def model_exists(*parts: str) -> bool:
    return os.path.isfile(asset_file(*parts))


def configure_ursina_assets() -> Path:
    """
    Point Ursina's asset folder at ``simulation/wheeled/assets``.

    Must run after ``Ursina()`` is constructed. Ursina 7+ requires a ``Path``,
    not a plain string (otherwise ``load_model`` crashes on ``.glob()``).
    """
    from ursina import application

    folder = Path(ASSETS_ROOT).resolve()
    application.asset_folder = folder
    return folder
