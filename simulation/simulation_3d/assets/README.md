# 3D Assets for Ursina Agricultural Simulation

Bundled models used by the Ursina backend (`--backend ursina`). All assets below
are **CC0** (public domain) unless noted.

## Robot (ground vehicle)

| File | Source | License |
|------|--------|---------|
| `models/robot/tractor.obj` | [Kenney Car Kit](https://kenney.nl/assets/car-kit) via [OpenGameArt](https://opengameart.org/content/car-kit) | CC0 |
| `models/robot/Textures/colormap.png` | Kenney Car Kit | CC0 |
| `models/robot/mars_rover.obj` | [Mars Rover](https://opengameart.org/content/mars-rover) by Tuomo Hijakka | CC0 |

Default robot model: **tractor** (agricultural ground vehicle).  
Fallback: **mars_rover** if tractor files are missing.

## Corn plants

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

Model paths and scale are set in `config/scene_config.json`:

```json
"robots": { "model": "models/robot/tractor.obj", "model_length": 2.0, ... },
"corn":   { "model": "models/corn/Corn_4.obj", "model_height": 1.15, ... }
```
