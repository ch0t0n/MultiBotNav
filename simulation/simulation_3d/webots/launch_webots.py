"""Generate world + config and launch Webots."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

WEBOTS_ROOT = os.path.dirname(os.path.abspath(__file__))
SIM_ROOT = os.path.dirname(WEBOTS_ROOT)
if SIM_ROOT not in sys.path:
    sys.path.insert(0, SIM_ROOT)
if WEBOTS_ROOT not in sys.path:
    sys.path.insert(0, WEBOTS_ROOT)

from core.paths import default_weights_path, default_wheeled_json  # noqa: E402
from generate_world import generate_world_file  # noqa: E402


def find_webots_executable() -> str | None:
    candidates = []
    webots_home = os.environ.get("WEBOTS_HOME")
    if webots_home:
        candidates.extend(
            [
                os.path.join(webots_home, "msys64", "webots.exe"),
                os.path.join(webots_home, "webots.exe"),
                os.path.join(webots_home, "webots"),
            ]
        )
    candidates.extend(
        [
            r"C:\Program Files\Webots\msys64\webots.exe",
            r"C:\Program Files\Webots\webots.exe",
            shutil.which("webots"),
        ]
    )
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def write_runtime_ini(python_exe: str, sim_root: str) -> str:
    controller_dir = os.path.join(WEBOTS_ROOT, "controllers", "wheeled_nav")
    ini_path = os.path.join(controller_dir, "runtime.ini")
    python_dir = os.path.dirname(python_exe)
    # Forward slashes avoid INI backslash escape issues on Windows paths.
    sim_root_ini = sim_root.replace("\\", "/")
    python_exe_ini = python_exe.replace("\\", "/")
    python_dir_ini = python_dir.replace("\\", "/")
    content = (
        "[environment]\n"
        f"PYTHONPATH = {sim_root_ini}\n"
        "\n"
        "[python]\n"
        f"COMMAND = {python_exe_ini}\n"
        f"PATH = {python_dir_ini}\n"
    )
    with open(ini_path, "w", encoding="utf-8") as f:
        f.write(content)
    return ini_path


def launch(
    env_key: str,
    weights: str | None = None,
    num_robots: int | None = None,
    json_path: str | None = None,
    random_policy: bool = False,
    max_steps: int = 1000,
    webots_exe: str | None = None,
) -> int:
    json_path = json_path or default_wheeled_json()
    if not random_policy and not weights:
        candidate = default_weights_path(env_key)
        weights = candidate if os.path.isfile(candidate) else None

    world_path = generate_world_file(
        env_key=env_key,
        json_path=json_path,
        num_robots=num_robots,
        weights_path=weights,
        random_policy=random_policy or weights is None,
        max_steps=max_steps,
    )
    write_runtime_ini(sys.executable, SIM_ROOT)

    webots_exe = webots_exe or find_webots_executable()
    if webots_exe is None:
        print("\nWebots executable not found.")
        print("Install Webots R2023b+ from https://cyberbotics.com/")
        print("Then either add webots to PATH or set WEBOTS_HOME.")
        print(f"\nWorld file ready to open manually:\n  {world_path}\n")
        print("In Webots: File → Open World → select the file above.")
        print("Press Play (▶) to start the trained-policy controller.")
        print("\nCamera controls in Webots:")
        print("  Left-drag   — rotate view")
        print("  Right-drag  — pan")
        print("  Scroll      — zoom")
        return 1

    print(f"Launching Webots: {webots_exe}")
    print(f"World: {world_path}")
    if weights:
        print(f"Policy: {weights}")
    else:
        print("Policy: random")

    env = os.environ.copy()
    env["PYTHONPATH"] = SIM_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["MULTIBOTNAV_ENV_KEY"] = env_key
    if num_robots is not None:
        env["MULTIBOTNAV_NUM_ROBOTS"] = str(num_robots)
    if weights:
        env["MULTIBOTNAV_WEIGHTS"] = weights

    return subprocess.call(
        [webots_exe, "--mode=realtime", world_path],
        cwd=WEBOTS_ROOT,
        env=env,
    )


def main():
    parser = argparse.ArgumentParser(description="Launch Webots agricultural simulation.")
    parser.add_argument("--env-key", default="env1")
    parser.add_argument("--num-robots", type=int, default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--random-policy", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--webots-exe", default=None)
    args = parser.parse_args()

    sys.exit(
        launch(
            env_key=args.env_key,
            weights=args.weights,
            num_robots=args.num_robots,
            json_path=args.json,
            random_policy=args.random_policy,
            max_steps=args.max_steps,
            webots_exe=args.webots_exe,
        )
    )


if __name__ == "__main__":
    main()
