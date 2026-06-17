# MultiBotNav 3D Agricultural Simulation

3D visualization of trained wheeled-robot navigation policies in an **agricultural
field** setting. The physics, observations, and CrossQ checkpoints are identical
to the 2D Pygame simulator (`new_simulations/wheeled_trained_for_cursor.py`) —
only the renderer changes.

Environment layouts are loaded from the same JSON file used during training:

```
exp_sets/wheeled/wheeled_configs.json
```

---

## Directory layout

```
simulation_3d/
├── run_simulation.py          # Main entry point (CLI)
├── requirements.txt
├── config/
│   └── scene_config.json      # 3D heights, colors, model paths
├── assets/                    # CC0 robot + tree OBJ models
│   └── models/
├── scripts/
│   └── download_assets.py     # Re-fetch models from OpenGameArt
├── core/                      # Shared physics + policy loading
│   ├── meshing.py             # Extruded polygon meshes
│   └── visuals/               # Asset loading + field layout
└── backends/
    └── ursina_agri.py         # Ursina 3D renderer
```

---

## Setup

```bash
cd simulation_3d
pip install -r requirements.txt
```

**Trained models:**

```
trained_models/wheeled/best_model_env{N}_stage2_robust_wind/best_model.zip
```

---

## Quick start

```bash
cd simulation_3d
python run_simulation.py --env-key env1 --random-policy
```

With a trained checkpoint:

```bash
python run_simulation.py --env-key env10 --weights "..\trained_models\wheeled\best_model_env10_stage2_robust_wind\best_model.zip"
```

Extruded obstacle polygons, a **tiled grass pasture**, **CC0 Quaternius trees**
(goal markers and pasture backdrop), and **CC0 tractor robots** (Kenney) driven by
bicycle-model physics. Visual settings are in `config/scene_config.json`; model
files live in `assets/models/`.

To (re)download models (robots, trees, etc.): `python scripts/download_assets.py`

Optional: `--scene-config path/to/scene_config.json` to override defaults.

| Flag | Default | Description |
|------|---------|-------------|
| `--env-key` | `env10` | Map key (`env1` … `env10`) |
| `--num-robots` | inferred | Robot count (`2`–`5`) |
| `--weights` | auto | Path to `best_model.zip` |
| `--random-policy` | off | Sample random actions |
| `--scene-config` | default | Override `scene_config.json` |
| `--robot-type` | from config | `tractor`, `delivery`, `tractor_shovel`, `rover` |
| `--fps` | `30` | Render frame rate |

---

## Coordinate system

Training 2D (origin bottom-left) maps to Ursina scene coordinates:

| Training | 3D scene |
|----------|----------|
| x → right | x (east) |
| y → up | z (north) |
| — | y (up) |

Field centered at origin: `scene_x = train_x - width/2`, `scene_z = train_y - height/2`.
