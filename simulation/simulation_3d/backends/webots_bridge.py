"""Launch Webots agricultural simulation with interactive 3D camera."""

from __future__ import annotations

import os
import subprocess
import sys

SIM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEBOTS_LAUNCH = os.path.join(SIM_ROOT, "webots", "launch_webots.py")


def launch_webots(
    env_key: str,
    num_robots: int | None,
    weights: str | None,
    json_path: str | None = None,
    random_policy: bool = False,
    max_steps: int = 1000,
) -> int:
    cmd = [
        sys.executable,
        WEBOTS_LAUNCH,
        "--env-key",
        env_key,
        "--max-steps",
        str(max_steps),
    ]
    if num_robots is not None:
        cmd.extend(["--num-robots", str(num_robots)])
    if weights:
        cmd.extend(["--weights", weights])
    if json_path:
        cmd.extend(["--json", json_path])
    if random_policy:
        cmd.append("--random-policy")

    return subprocess.call(cmd, cwd=SIM_ROOT)


def run_webots_simulation(
    env_key: str,
    num_robots: int | None,
    weights: str | None,
    json_path: str | None = None,
    random_policy: bool = False,
    max_steps: int = 1000,
) -> int:
    """Generate world + config and open Webots (or print manual instructions)."""
    return launch_webots(
        env_key=env_key,
        num_robots=num_robots,
        weights=weights,
        json_path=json_path,
        random_policy=random_policy,
        max_steps=max_steps,
    )
