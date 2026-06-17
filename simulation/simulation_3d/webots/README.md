# Webots Agricultural Simulation

Interactive 3D simulation with **free camera control** in the Webots GUI.
Physics and policies match the 2D Pygame trainer; only visualization runs in Webots.

## Prerequisites

1. [Webots](https://cyberbotics.com/) **R2023b or newer**
2. Python environment with project dependencies (`gymnasium`, `sb3-contrib`, `shapely`, …)

## Quick start

From `simulation_3d/`:

```bash
# Random policy (no checkpoint required)
python run_simulation.py --backend webots --env-key env1 --random-policy

# Trained CrossQ policy
python run_simulation.py --backend webots --env-key env10 --weights "..\trained_models\wheeled\best_model_env10_stage2_robust_wind\best_model.zip"
```

Or use the Webots launcher directly:

```bash
python webots/launch_webots.py --env-key env1 --random-policy
```

This will:

1. Generate `webots/worlds/ag_field_env1.wbt` from JSON
2. Write `webots/controllers/wheeled_nav/sim_config.json`
3. Write `runtime.ini` pointing to your current Python
4. Open Webots (if installed) or print the world path for manual open

Press **Play (▶)** in Webots to start the controller.

## Camera controls (Webots 3D view)

| Input | Action |
|-------|--------|
| Left mouse drag | Rotate camera |
| Right mouse drag | Pan |
| Mouse wheel | Zoom |
| `Ctrl` + arrows | Move viewpoint |

You can orbit around the full agricultural field at any angle.

## Manual open (if Webots is not on PATH)

1. Run `python webots/generate_world.py --env-key env1 --random-policy`
2. Open Webots → **File → Open World**
3. Select `simulation_3d/webots/worlds/ag_field_env1.wbt`
4. Press **Play (▶)**

Set `WEBOTS_HOME` to your install folder, e.g.:

```
C:\Program Files\Webots
```

## World contents

- Soil field with crop-row stripes and wooden fence
- JSON obstacles as barns / hay bales / hedgerows
- Spray-zone goals with flag markers (turn grey when visited)
- Tractor-style robots at JSON spawn poses
- Wind-direction arrow
- Supervisor robot running `wheeled_nav` controller

## Configuration

Settings are stored in `controllers/wheeled_nav/sim_config.json` when the world is generated:

```json
{
  "env_key": "env1",
  "num_robots": null,
  "weights_path": null,
  "random_policy": true,
  "max_steps": 1000
}
```

Environment variables override the config at runtime:

- `MULTIBOTNAV_ENV_KEY`
- `MULTIBOTNAV_NUM_ROBOTS`
- `MULTIBOTNAV_WEIGHTS`

## Project layout (Webots expects this structure)

```
webots/
├── worlds/           ← .wbt files
├── controllers/
│   └── wheeled_nav/  ← wheeled_nav.py + sim_config.json + runtime.ini
├── generate_world.py
└── launch_webots.py
```

Open worlds from this `webots/` folder so Webots resolves the `wheeled_nav` controller path correctly.

## Troubleshooting

### Black screen + `Forced termination` in console

Webots kills the controller if it does not call `supervisor.step()` within ~1 second.
Heavy imports (TensorFlow / Stable-Baselines3) used to run before that call.

**Fix (already applied in current `wheeled_nav.py`):** the controller calls
`supervisor.step()` immediately, then loads ML libraries.

If problems persist:

1. Regenerate config and `runtime.ini`:
   ```bash
   python webots/launch_webots.py --env-key env1 --random-policy
   ```
   (If Webots is not found, the script still updates `runtime.ini` and the world.)

2. Check the controller log:
   ```
   webots/controllers/wheeled_nav/wheeled_nav.log
   ```

3. Open the world from the `webots/` directory (not by double-clicking the `.wbt`
   from Explorer) so the `wheeled_nav` controller path resolves correctly.

4. In Webots: **View → Optional Rendering → Disable all** should be **off**.

5. Press **Play (▶)** after the world loads — the scene stays black until simulation runs.
