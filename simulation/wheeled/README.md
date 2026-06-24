# MultiBotNav 3D Wheeled Simulation

3D Ursina visualization of trained wheeled-robot navigation policies in an
agricultural field setting. Physics and observations match the 2D wheeled trainer;
only the renderer differs.

Environment layouts are loaded from the repo training config:

```
../../exp_sets/wheeled/wheeled_configs.json
```

(relative to this folder: `simulation/wheeled`)

---

## Directory layout

```
MultiBotNav/
├── exp_sets/wheeled/wheeled_configs.json
├── trained_models/wheeled/best_model_env{N}_stage2_robust_wind/best_model.zip
└── simulation/wheeled/
    ├── run_simulation.py          # Entry point (CLI)
    ├── download_assets.py         # Optional: fetch CC0 models
    ├── requirements.txt
    ├── config/scene_config.json
    ├── assets/models/
    └── core/
        ├── multi_wheeled.py
        ├── policy.py
        ├── ursina_scene.py
        └── visuals/
```

---

## Setup

```bash
cd simulation/wheeled
pip install -r requirements.txt
```

If tree textures or robot models are missing:

```bash
python download_assets.py
```

**Default trained model path** (from `simulation/wheeled`):

```
../../trained_models/wheeled/best_model_env{N}_stage2_robust_wind/best_model.zip
```

---

## Quick start

```bash
cd simulation/wheeled
python run_simulation.py --env-key env1 --random-policy
```

With an explicit checkpoint:

```bash
python run_simulation.py --env-key env10 --weights "..\..\trained_models\wheeled\best_model_env10_stage2_robust_wind\best_model.zip"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--env-key` | `env10` | Map key (`env1` … `env10`) |
| `--num-robots` | inferred | Robot count (`2`–`5`) |
| `--weights` | `../../trained_models/wheeled/...` | Path to `best_model.zip` |
| `--json` | `../../exp_sets/wheeled/wheeled_configs.json` | Environment layout JSON |
| `--random-policy` | off | Sample random actions |
| `--scene-config` | default | Override `scene_config.json` |
| `--robot-type` | from config | `tractor`, `delivery`, `tractor_shovel`, `rover` |
| `--fps` | `30` | Render frame rate |

---

## Coordinate system

| Training (2D) | 3D scene (Ursina) |
|---------------|-------------------|
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

Robot profiles: `core/visuals/robot_model.py`. Tree variants: `config/scene_config.json`.
