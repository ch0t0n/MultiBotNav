"""Isaac Sim integration bridge (state export + setup instructions)."""

from __future__ import annotations

import json
import os
import sys

SIM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_isaac_instructions(env_key: str, num_robots: int, weights: str | None):
    script = os.path.join(SIM_ROOT, "isaac", "isaac_sim_extension.py")
    print("\n=== NVIDIA Isaac Sim instructions ===")
    print("Isaac Sim requires Omniverse + a CUDA GPU. This repo provides a bridge")
    print("that exports the same 2D physics state for visualization in Isaac Sim.")
    print()
    print("1. Install Isaac Sim 4.x and enable the Python environment.")
    print("2. Copy the extension template:")
    print(f"   {script}")
    print("   into your Isaac Sim extensions folder.")
    print("3. Run the headless exporter to validate the policy loop:")
    print(
        f"   python run_simulation.py --backend headless --env-key {env_key} "
        f"--num-robots {num_robots}"
    )
    print("4. In Isaac Sim, load scene: simulation_3d/isaac/scenes/ag_field.usd")
    print("   (generate with: python isaac/generate_usd_stub.py)")
    print("5. Start the MultiBotNav extension; it reads state from:")
    print("   simulation_3d/isaac/runtime/state.json")
    if weights:
        print(f"6. Set MULTIBOTNAV_WEIGHTS={weights}")
    print("=====================================\n")


def export_state(env, info: dict, path: str | None = None):
    """Write robot poses and goals for Isaac Sim to consume."""
    path = path or os.path.join(SIM_ROOT, "isaac", "runtime", "state.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "world_width": float(env.world_width),
        "world_height": float(env.world_height),
        "robots": [
            {
                "x": float(r[0]),
                "y": float(r[1]),
                "theta": float(r[2]),
                "v": float(r[3]),
                "delta": float(r[4]),
            }
            for r in env.robots
        ],
        "goals": [
            {"x": float(g[0]), "y": float(g[1]), "visited": bool(v)}
            for g, v in zip(env.goal_positions, env.goal_visited)
        ],
        "obstacles": [
            [[float(p[0]), float(p[1])] for p in poly]
            for poly in env.obstacles
        ],
        "info": {
            "step_count": int(info.get("step_count", 0)),
            "goals_visited": int(info.get("goals_visited", 0)),
            "path_length": float(info.get("path_length", 0.0)),
            "term_cond": str(info.get("term_cond", "")),
            "wind_mag": float(info.get("wind_mag", 0.0)),
            "wind_dir": float(info.get("wind_dir", 0.0)),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
