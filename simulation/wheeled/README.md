# MultiBotNav 3D Agricultural Simulation

3D visualization of trained wheeled-robot navigation policies in an **agricultural
field** setting. The physics, observations, and CrossQ checkpoints are identical
to the 2D Pygame simulator (`new_simulations/wheeled_trained_for_cursor.py`) — only
the renderer changes.

Environment layouts are loaded from the same JSON file used during training:

```
exp_sets/wheeled/wheeled_configs.json
```

---

## Directory layout

```
simulation_3d/
├── run_simulation.py          # Entry point (CLI)
├── download_assets.py         # Optional: fetch CC0 models from OpenGameArt
├── requirements.txt
├── config/
│   └── scene_config.json      # 3D heights, colors, model paths
├── assets/models/             # Bundled robot + tree OBJ models (CC0)
└── core/
    ├── multi_wheeled.py       # Bicycle-model physics environment
    ├── policy.py              # CrossQ checkpoint loading
    ├── ursina_scene.py        # Ursina renderer + simulation loop
    └── visuals/               # Field, trees, robots, landscape
```

---

## Setup

```bash
cd simulation_3d
pip install -r requirements.txt
```

If tree textures or robot models are missing, run once:

```bash
python download_assets.py
```

**Trained models:**

```
trained_models/wheeled/best_model_env{N}_stage2_robust_wind/best_model.zip
```

---

## Quick start

```bash
python run_simulation.py --env-key env1 --random-policy
```

With a trained checkpoint:

```bash
python run_simulation.py --env-key env10 --weights "..\trained_models\wheeled\best_model_env10_stage2_robust_wind\best_model.zip"
```

The scene includes extruded obstacle polygons, a tiled grass pasture, CC0 Quaternius
trees (goal markers and backdrop), and CC0 Kenney tractor robots. Visual settings
are in `config/scene_config.json`.

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

---

## Assets (CC0)

| Models | Source |
|--------|--------|
| `assets/models/robot/` | [Kenney Car Kit](https://kenney.nl/assets/car-kit), [Mars Rover](https://opengameart.org/content/mars-rover) |
| `assets/models/trees/quaternius/` | [Quaternius Textured Trees](https://opengameart.org/content/lowpoly-textured-trees) |

Robot type profiles live in `core/visuals/robot_model.py`. Tree variants are set in
`config/scene_config.json` under `goals` and `scenery`.
