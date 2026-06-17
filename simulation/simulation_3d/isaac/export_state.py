"""Export simulation state while running headless (for Isaac Sim sync)."""

from __future__ import annotations

import argparse
import os
import sys

SIM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SIM_ROOT not in sys.path:
    sys.path.insert(0, SIM_ROOT)

from backends.isaac_bridge import export_state  # noqa: E402
from core.paths import default_weights_path  # noqa: E402
from core.policy import prepare_env  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-key", default="env10")
    parser.add_argument("--num-robots", type=int, default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    weights = args.weights or default_weights_path(args.env_key)
    env, model, _ = prepare_env(
        env_key=args.env_key,
        num_robots=args.num_robots,
        weights_path=weights if os.path.isfile(weights) else None,
    )

    obs, _ = env.reset()
    out = os.path.join(SIM_ROOT, "isaac", "runtime", "state.json")

    for _ in range(args.steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, _, terminated, truncated, info = env.step(action)
        export_state(env, info, out)
        if terminated or truncated:
            obs, _ = env.reset()

    print(f"Exported state to {out}")


if __name__ == "__main__":
    main()
