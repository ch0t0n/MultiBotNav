"""CC0 Quaternius stylized tree models (OpenGameArt) for goals and scenery."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from core.visuals.asset_paths import model_exists, model_relative

if TYPE_CHECKING:
    from ursina import Entity

DEFAULT_GOAL_VARIANT = "Tree_3"
DEFAULT_SCENERY_VARIANTS = (
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

DEFAULT_FOLIAGE_TINT = (0.28, 0.46, 0.16, 1.0)


def foliage_tint_from_cfg(cfg: dict | None) -> tuple[float, float, float, float]:
    """Read optional foliage tint from a scene section or the shared ``trees`` block."""
    if not cfg:
        return DEFAULT_FOLIAGE_TINT
    if "foliage_tint" in cfg:
        return tuple(float(c) for c in cfg["foliage_tint"])  # type: ignore[return-value]
    trees = cfg.get("trees") or {}
    if "foliage_tint" in trees:
        return tuple(float(c) for c in trees["foliage_tint"])  # type: ignore[return-value]
    return DEFAULT_FOLIAGE_TINT


NATIVE_HEIGHTS: dict[str, float] = {
    "Tree_1": 5.54,
    "Tree_2": 5.54,
    "Tree_3": 5.72,
    "Tree_4": 5.72,
    "Tree_5": 5.72,
    "Birch_1": 9.48,
    "Birch_2": 9.48,
    "Pine_1": 7.58,
    "Pine_2": 7.58,
}

# Ursina's OBJ importer ignores map_Kd in MTL files, so bark/leaves are split meshes
# with textures assigned explicitly (see download_assets.py).
VARIANT_TEXTURES: dict[str, tuple[str, str]] = {
    "deciduous": (
        "models/trees/Textures/Tree_Bark.jpg",
        "models/trees/Textures/Tree_Leaves.png",
    ),
    "birch": (
        "models/trees/Textures/Birch_Bark.png",
        "models/trees/Textures/Birch_Leaves_Green.png",
    ),
    "pine": (
        "models/trees/Textures/Tree_Bark.jpg",
        "models/trees/Textures/Pine_Leaves.png",
    ),
}


def _variant_family(variant: str) -> str:
    if variant.startswith("Birch"):
        return "birch"
    if variant.startswith("Pine"):
        return "pine"
    return "deciduous"


def _part_rel_path(variant: str, part: str) -> str:
    return f"models/trees/quaternius/{variant}_{part}.obj"


def resolve_tree_part(variant: str, part: str) -> str | None:
    parts = _part_rel_path(variant, part).replace("\\", "/").split("/")
    if model_exists(*parts):
        return model_relative(*parts)
    return None


def resolve_tree_textures(variant: str) -> tuple[str, str] | None:
    family = _variant_family(variant)
    bark_rel, leaf_rel = VARIANT_TEXTURES[family]
    bark_parts = bark_rel.split("/")
    leaf_parts = leaf_rel.split("/")
    if model_exists(*bark_parts) and model_exists(*leaf_parts):
        return model_relative(*bark_parts), model_relative(*leaf_parts)
    return None


def available_variants(candidates: tuple[str, ...] | list[str]) -> list[str]:
    out = []
    for name in candidates:
        if resolve_tree_part(name, "bark") and resolve_tree_part(name, "leaves"):
            out.append(name)
    return out


def goal_variant(goal_cfg: dict) -> str:
    preferred = goal_cfg.get("tree_variant")
    if preferred and resolve_tree_part(preferred, "bark"):
        return preferred
    configured = goal_cfg.get("tree_variants") or goal_cfg.get("variants")
    if configured:
        found = available_variants(tuple(configured))
        if found:
            return found[0]
    if resolve_tree_part(DEFAULT_GOAL_VARIANT, "bark"):
        return DEFAULT_GOAL_VARIANT
    found = available_variants(DEFAULT_SCENERY_VARIANTS)
    if not found:
        raise FileNotFoundError(
            "No tree models found. Run: python download_assets.py"
        )
    return found[0]


def scenery_variants(scenery_cfg: dict) -> list[str]:
    configured = scenery_cfg.get("tree_variants") or scenery_cfg.get("tree_models")
    if configured:
        found = available_variants(tuple(configured))
        if found:
            return found
    return available_variants(DEFAULT_SCENERY_VARIANTS)


def native_height(variant: str, fallback: float = 5.7) -> float:
    return float(NATIVE_HEIGHTS.get(variant, fallback))


def uniform_scale(variant: str, target_height: float) -> float:
    return target_height / max(native_height(variant), 1e-3)


def _configure_opaque_surface(entity) -> None:
    """Disable Ursina's default alpha blending on opaque bark geometry."""
    if not entity.model:
        return
    from panda3d.core import TransparencyAttrib

    entity.model.setTransparency(TransparencyAttrib.M_none)


def _configure_foliage(entity) -> None:
    """
    Leaf atlases use alpha cutouts. Ursina enables M_dual transparency on every
    model, which makes cutout foliage disappear; use alpha test instead.
    """
    entity.double_sided = True
    if not entity.model:
        return
    from panda3d.core import AlphaTestAttrib, TransparencyAttrib

    entity.model.setTransparency(TransparencyAttrib.M_none)
    entity.model.setAttrib(AlphaTestAttrib.make(AlphaTestAttrib.MGreater, 0.35), 1)
    entity.model.setDepthWrite(True)


def mesh_base_offset(variant: str, scale: float, cfg: dict) -> float:
    if "model_base_y" in cfg:
        return -float(cfg["model_base_y"]) * scale
    return 0.0


def create_tree_entity(
    Entity,
    parent,
    variant: str,
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_y: float = 0.0,
    scale: float = 1.0,
    y_offset: float = 0.0,
    foliage_tint: tuple[float, float, float, float] | None = None,
):
    """Instantiate a textured Quaternius tree (bark + leaf meshes)."""
    bark_path = resolve_tree_part(variant, "bark")
    leaves_path = resolve_tree_part(variant, "leaves")
    textures = resolve_tree_textures(variant)
    if not bark_path or not leaves_path or not textures:
        raise FileNotFoundError(
            f"Tree parts not found for variant '{variant}'. "
            "Run: python download_assets.py"
        )
    bark_tex, leaf_tex = textures

    assembly = Entity(
        parent=parent,
        position=(position[0], position[1], position[2]),
        rotation_y=rotation_y,
        collider=None,
    )
    mesh_root = Entity(
        parent=assembly,
        position=(0.0, y_offset, 0.0),
        scale=(scale, scale, scale),
        collider=None,
    )
    trunk_mesh = Entity(
        parent=mesh_root,
        model=bark_path,
        collider=None,
        render_queue=0,
    )
    trunk_mesh.color = (1.0, 1.0, 1.0, 1.0)
    trunk_mesh.texture = bark_tex
    _configure_opaque_surface(trunk_mesh)

    foliage_mesh = Entity(
        parent=mesh_root,
        model=leaves_path,
        collider=None,
        render_queue=1,
    )
    foliage_mesh.texture = leaf_tex
    foliage_mesh.color = foliage_tint or DEFAULT_FOLIAGE_TINT
    # Run after texture: Ursina's model_setter re-applies M_dual transparency.
    _configure_foliage(foliage_mesh)
    return assembly, trunk_mesh, foliage_mesh


def create_scenery_tree(
    Entity,
    parent,
    x: float,
    z: float,
    variant: str,
    target_height: float,
    rng: random.Random,
    scenery_cfg: dict | None = None,
):
    scenery_cfg = scenery_cfg or {}
    height = target_height * rng.uniform(0.85, 1.12)
    scale = uniform_scale(variant, height)
    yaw = rng.uniform(0.0, 360.0)
    y_off = mesh_base_offset(variant, scale, scenery_cfg)
    tint = foliage_tint_from_cfg(scenery_cfg)
    root, trunk, foliage = create_tree_entity(
        Entity,
        parent,
        variant,
        position=(x, 0.0, z),
        rotation_y=yaw,
        scale=scale,
        y_offset=y_off,
        foliage_tint=tint,
    )
    return root, trunk, foliage
