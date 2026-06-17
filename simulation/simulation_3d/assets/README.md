# 3D Assets for Ursina Agricultural Simulation

Bundled models used by the Ursina backend (`--backend ursina`). All assets below
are **CC0** (public domain) unless noted.

## Robot (ground vehicle)

| File | Source | License |
|------|--------|---------|
| `models/robot/tractor.obj` | [Kenney Car Kit](https://kenney.nl/assets/car-kit) via [OpenGameArt](https://opengameart.org/content/car-kit) | CC0 |
| `models/robot/Textures/colormap.png` | Kenney Car Kit | CC0 |
| `models/robot/tractor_shovel.obj` | Kenney Car Kit | CC0 |
| `models/robot/delivery.obj` | Kenney Car Kit | CC0 |

| `models/robot/mars_rover.obj` | [Mars Rover](https://opengameart.org/content/mars-rover) by Tuomo Hijakka | CC0 |

Default robot type: **tractor** (set in `config/scene_config.json` or `--robot-type`).  
Type profiles (model path, texture, yaw offset) live in `core/visuals/robot_model.py`.

## Trees (goal markers and pasture scenery)

| File | Source | License |
|------|--------|---------|
| `models/trees/quaternius/Tree_*.obj` | [LowPoly Textured Trees](https://opengameart.org/content/lowpoly-textured-trees) by Quaternius | CC0 |
| `models/trees/quaternius/Birch_*.obj` | Quaternius stylized trees pack | CC0 |
| `models/trees/quaternius/Pine_*.obj` | Quaternius stylized trees pack | CC0 |
| `models/trees/Textures/*.png` | Bark and leaf atlases for the tree OBJ materials | CC0 |

Goal trees and background pasture trees use these textured models (branch-style
canopies instead of primitive spheres). Variants are listed in `goals.tree_variants`
and `scenery.tree_variants` inside `config/scene_config.json`.

## Corn plants (optional legacy assets)

| File | Source | License |
|------|--------|---------|
| `models/corn/corn.obj` | [Simple Corn](https://opengameart.org/content/simple-corn-obj) by Djsedj | CC0 |
| `models/corn/Corn_4.obj` | [Nature Crops Pack](https://opengameart.org/content/lowpoly-crops-pack) by Quaternius | CC0 |

Default crop model: **corn.obj** (Ursina-compatible). `Corn_4.obj` is kept as an
optional higher-detail mesh but may require triangulation in Blender before Ursina
can load it.

## Re-download full packs

```bash
cd simulation_3d
python scripts/download_assets.py
```

This fetches the OpenGameArt archives and copies the canonical files into
`assets/models/`.

## Configuration

Model paths and scale are set in `config/scene_config.json`. Robot type is selected with `"robots": { "type": "tractor" }` or `--robot-type`.
