"""
Observation-gap analysis (sim-to-real) for multi-robot navigation.

Supports both robot platforms in this repository:
  - UAV      (MultiUAV,     obs_mode="full", 4N+1)
  - Wheeled  (MultiWheeled, obs_mode="full", 5N+1)

INSTRUCTIONS:
  1. Run batch evaluation (no visual simulator required):
         python sim2real.py
         python sim2real.py --robot_type uav
         python sim2real.py --robot_type wheeled
  2. Optional UAV CoppeliaSim visual inspection (one episode at a time):
         python sim2real.py --robot_type uav --render_coppelia
     Requires CoppeliaSim with simulation/uav/coppeliasim_envs/uav_common_env.ttt open.
  3. After running, copy the printed IQM values into tab:obs_gap in writings/0_main.tex
     (UAV and wheeled blocks as applicable).

Policy: CrossQ main_default  (obs_mode="full")
Env:    variation 1, N=3

UAV obs_mode="full" layout  (4N + 1):
    [0    : 2N]   robot positions  (x, y per robot)
    [2N   : 4N]   robot velocities (vx, vy per robot)
    [4N]          visited_decimal  (binary coverage state)

Wheeled obs_mode="full" layout  (5N + 1):
    [0    : 2N]   robot positions  (x, y per robot)
    [2N   : 5N]   kinematics       (θ, v, δ per robot)
    [5N]          goal_decimal     (binary goal-visit state)

Perturbation → obs slice targeted:
    GPS noise        → positions  [0 : 2N]
    Wind latency     → UAV: velocities [2N : 4N]  (5-step-old values in "all combined")
                       Wheeled: kinematics [2N : 5N]  (θ, v, δ stale by N steps)
    Coverage dropout → UAV: visited [4N]  |  Wheeled: goal [5N]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque

import gymnasium as gym
import numpy as np
from sb3_contrib import CrossQ

sys.path.insert(0, os.path.join("."))
from src.env import MultiUAV, MultiWheeled
from src.utils import read_uav_json, read_wheeled_json

# ────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────
UAV_JSON_PATH = os.path.join("exp_sets", "uav", "cont_sets.json")
WHEELED_JSON_PATH = os.path.join("exp_sets", "wheeled", "wheeled_configs.json")

ENV_VARIATION = 1
NUM_ROBOTS = 3
N_EVAL_EPISODES = 50
HEIGHT = 0.4

DEFAULT_MODEL_PATHS = {
    "uav": [
        os.path.join(
            "logs", "main_default",
            "CrossQ_uav_N3_env1_seed42",
            "best_model", "best_model.zip",
        ),
        os.path.join(
            "trained_models", "uav", "May13_16_v9_uav_seed42",
            "best_model_env1", "best_model.zip",
        ),
    ],
    "wheeled": [
        os.path.join(
            "logs", "main_default",
            "CrossQ_wheeled_N3_env1_seed42",
            "best_model_stage2", "best_model.zip",
        ),
        os.path.join(
            "trained_models", "wheeled",
            "best_model_env1_stage2_robust_wind", "best_model.zip",
        ),
    ],
}

# Perturbation magnitudes (world units differ: UAV ~100 m field, wheeled ~500 px field)
PERTURBATION_CONFIG = {
    "uav": dict(
        gps_noise=5.0,
        gps_label="GPS noise (sigma=5.0)",
        kin_latency=3,
        kin_label="Wind estimate latency (3 steps)",
        kin_latency_combined=5,
        drop_prob=0.10,
        drop_label="Coverage dropout (10%)",
    ),
    "wheeled": dict(
        gps_noise=25.0,
        gps_label="GPS noise (sigma=25.0)",
        kin_latency=3,
        kin_label="Kinematic latency (3 steps)",
        kin_latency_combined=5,
        drop_prob=0.10,
        drop_label="Goal-map dropout (10%)",
    ),
}


# ────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────
def compute_iqm(rewards):
    rewards = np.array(rewards, dtype=np.float32)
    q25, q75 = np.percentile(rewards, [25, 75])
    mask = (rewards >= q25) & (rewards <= q75)
    return float(np.mean(rewards[mask])) if mask.any() else float(np.mean(rewards))


def infer_num_robots_from_model(model, robot_type: str) -> int:
    obs_dim = int(model.observation_space.shape[0])
    if robot_type == "uav":
        if (obs_dim - 1) % 4 != 0:
            raise ValueError(
                f"Cannot infer UAV num_robots from obs dim {obs_dim}; expected 4N+1."
            )
        return (obs_dim - 1) // 4

    if (obs_dim - 1) % 5 != 0:
        raise ValueError(
            f"Cannot infer wheeled num_robots from obs dim {obs_dim}; expected 5N+1."
        )
    return (obs_dim - 1) // 5


def resolve_model_path(robot_type: str, model_path: str | None) -> str:
    if model_path:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        return model_path

    for candidate in DEFAULT_MODEL_PATHS[robot_type]:
        if os.path.isfile(candidate):
            return candidate

    tried = "\n  ".join(DEFAULT_MODEL_PATHS[robot_type])
    raise FileNotFoundError(
        f"No {robot_type} model found. Tried:\n  {tried}\n"
        "Train with Step 1 (INSTRUCTIONS.MD) or pass --model_path."
    )


def build_conditions(robot_type: str) -> list[tuple[str, float, int, float]]:
    cfg = PERTURBATION_CONFIG[robot_type]
    return [
        ("No perturbation (baseline)", 0.0, 0, 0.00),
        (cfg["gps_label"], cfg["gps_noise"], 0, 0.00),
        (cfg["kin_label"], 0.0, cfg["kin_latency"], 0.00),
        (cfg["drop_label"], 0.0, 0, cfg["drop_prob"]),
        (
            "All combined",
            cfg["gps_noise"],
            cfg["kin_latency_combined"],
            cfg["drop_prob"],
        ),
    ]


def format_results_table(
    robot_type: str,
    results: dict[str, float],
    conditions: list[tuple[str, float, int, float]],
) -> str:
    baseline_label = conditions[0][0]
    baseline = results[baseline_label]

    alone_labels = {conditions[0][0], conditions[-1][0]}

    lines = []
    lines.append("=" * 60)
    lines.append(
        f"TABLE ({robot_type.upper()}) — copy into tab:obs_gap in writings/0_main.tex"
    )
    lines.append("=" * 60)
    lines.append(f"  {'Condition':<44} {'IQM':>8}  {'dIQM (%)':>10}  Alone")
    lines.append(f"  {'-' * 44} {'-' * 8}  {'-' * 10}  -----")
    for label, iqm in results.items():
        delta = ((iqm - baseline) / abs(baseline) * 100) if baseline != 0 else 0.0
        sign = "+" if delta > 0 else ""
        alone = "Y" if label not in alone_labels else "---"
        lines.append(f"  {label:<44} {iqm:>8.2f}  {sign}{delta:>8.1f}%   {alone}")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────
# Observation perturbers
# ────────────────────────────────────────────────────────────────
class UavObsPerturber:
    """obs_mode='full' for MultiUAV (4N + 1)."""

    def __init__(
        self,
        N: int,
        gps_noise: float = 0.0,
        wind_latency_steps: int = 0,
        coverage_drop_prob: float = 0.0,
    ):
        self.N = N
        self.gps_noise = gps_noise
        self.wind_latency_steps = wind_latency_steps
        self.coverage_drop_prob = coverage_drop_prob

        self._pos_s, self._pos_e = 0, 2 * N
        self._vel_s, self._vel_e = 2 * N, 4 * N
        self._cov_s, self._cov_e = 4 * N, 4 * N + 1

        _buf_len = wind_latency_steps if wind_latency_steps > 0 else 1
        self._vel_buf: deque = deque(
            [np.zeros(2 * N, dtype=np.float32) for _ in range(_buf_len)],
            maxlen=_buf_len,
        )

    def reset(self):
        for i in range(len(self._vel_buf)):
            self._vel_buf[i] = np.zeros(2 * self.N, dtype=np.float32)

    def apply(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.copy()

        if self.gps_noise > 0.0:
            obs[self._pos_s : self._pos_e] += np.random.normal(
                0.0, self.gps_noise, size=(self._pos_e - self._pos_s,)
            ).astype(np.float32)

        if self.wind_latency_steps > 0:
            stale_vel = self._vel_buf[0].copy()
            self._vel_buf.append(obs[self._vel_s : self._vel_e].copy())
            obs[self._vel_s : self._vel_e] = stale_vel

        if self.coverage_drop_prob > 0.0 and np.random.rand() < self.coverage_drop_prob:
            obs[self._cov_s : self._cov_e] = 0.0

        return obs


class WheeledObsPerturber:
    """obs_mode='full' for MultiWheeled (5N + 1)."""

    def __init__(
        self,
        N: int,
        gps_noise: float = 0.0,
        kin_latency_steps: int = 0,
        goal_drop_prob: float = 0.0,
    ):
        self.N = N
        self.gps_noise = gps_noise
        self.kin_latency_steps = kin_latency_steps
        self.goal_drop_prob = goal_drop_prob

        self._pos_s, self._pos_e = 0, 2 * N
        self._kin_s, self._kin_e = 2 * N, 5 * N
        self._goal_s, self._goal_e = 5 * N, 5 * N + 1

        _buf_len = kin_latency_steps if kin_latency_steps > 0 else 1
        self._kin_buf: deque = deque(
            [np.zeros(3 * N, dtype=np.float32) for _ in range(_buf_len)],
            maxlen=_buf_len,
        )

    def reset(self):
        for i in range(len(self._kin_buf)):
            self._kin_buf[i] = np.zeros(3 * self.N, dtype=np.float32)

    def apply(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.copy()

        if self.gps_noise > 0.0:
            obs[self._pos_s : self._pos_e] += np.random.normal(
                0.0, self.gps_noise, size=(self._pos_e - self._pos_s,)
            ).astype(np.float32)

        if self.kin_latency_steps > 0:
            stale_kin = self._kin_buf[0].copy()
            self._kin_buf.append(obs[self._kin_s : self._kin_e].copy())
            obs[self._kin_s : self._kin_e] = stale_kin

        if self.goal_drop_prob > 0.0 and np.random.rand() < self.goal_drop_prob:
            obs[self._goal_s : self._goal_e] = 0.0

        return obs


# ────────────────────────────────────────────────────────────────
# CoppeliaSim drone simulator (UAV visual mode only)
# ────────────────────────────────────────────────────────────────
class DroneSimulator:
    def __init__(self, gym_env, sim, scaling_factor=8, height=0.4, num_robots=3):
        self.sim = sim
        self.scaling_factor = scaling_factor
        self.height = height
        self.num_robots = num_robots
        self.polygon = gym_env.unwrapped.poly_vertices
        scaled = [(x / scaling_factor, y / scaling_factor) for (x, y) in self.polygon]
        self.rounded_polygon = scaled + [scaled[0]]
        self.target_locations = list(gym_env.unwrapped.initial_inf_locations)
        self.nozzle_scripts = []
        self.all_drones = []
        self.initial_positions = {}
        self.spawned_targets = []
        self.field_drawing = None

        for i in range(num_robots):
            h = sim.getObject(f"/Quadcopter[{i}]/Script", {"noError": True})
            if h != -1:
                self.nozzle_scripts.append(h)

        i = 0
        while True:
            h = sim.getObject(f"/Quadcopter[{i}]", {"noError": True})
            if h == -1:
                break
            self.all_drones.append(h)
            self.initial_positions[h] = sim.getObjectPosition(h, -1)
            i += 1

    def start(self):
        self.sim.startSimulation()

    def stop(self):
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            self.sim.step()
        self._clear_field()
        self._clear_targets()
        for drone, pos in self.initial_positions.items():
            self.sim.setObjectPosition(drone, -1, pos)
            self.sim.setModelProperty(drone, 0)

    def draw_field(self):
        self.field_drawing = self.sim.addDrawingObject(
            self.sim.drawing_lines, 5, 0, -1, 9999, [255, 255, 255])
        for i in range(len(self.rounded_polygon) - 1):
            p1, p2 = self.rounded_polygon[i], self.rounded_polygon[i + 1]
            self.sim.addDrawingObjectItem(
                self.field_drawing,
                [p1[0], p1[1], 0.1, p2[0], p2[1], 0.1])

    def set_agent_positions(self, info):
        for i, drone in enumerate(self.all_drones):
            if i < self.num_robots:
                pos = info[f"robot{i}"]
                self.sim.setObjectPosition(
                    drone, -1,
                    [p / self.scaling_factor for p in pos] + [self.height])
                self.sim.setObjectInt32Param(
                    drone, self.sim.objintparam_visibility_layer, 1)
            else:
                self.sim.setModelProperty(
                    drone,
                    self.sim.modelproperty_not_visible
                    | self.sim.modelproperty_not_collidable
                    | self.sim.modelproperty_not_detectable
                    | self.sim.modelproperty_not_dynamic)

    def set_target_locations(self):
        self._clear_targets()
        target_template = self.sim.getObject("/target_marker", {"noError": True})
        if target_template == -1:
            return
        for loc in self.target_locations:
            new_pos = [xi / self.scaling_factor for xi in loc] + [0]
            new_target = self.sim.copyPasteObjects([target_template])[0]
            self.sim.setObjectPosition(new_target, -1, new_pos)
            self.spawned_targets.append(new_target)

    def move_agents(self, info, action):
        for i in range(self.num_robots):
            target = self.sim.getObject(f"/target[{i}]")
            pos = info[f"robot{i}"]
            self.sim.setObjectPosition(
                target, -1,
                [p / self.scaling_factor for p in pos] + [self.height])

    def _clear_field(self):
        if self.field_drawing is not None:
            self.sim.removeDrawingObject(self.field_drawing)
            self.field_drawing = None

    def _clear_targets(self):
        for obj in self.spawned_targets:
            if self.sim.isHandle(obj):
                self.sim.removeObject(obj)
        self.spawned_targets = []


# ────────────────────────────────────────────────────────────────
# Single-episode runner
# ────────────────────────────────────────────────────────────────
def run_episode(env, model, perturber, drone_sim=None):
    obs, info = env.reset()
    perturber.reset()

    if drone_sim:
        drone_sim.set_agent_positions(info)
        drone_sim.set_target_locations()
        drone_sim.start()

    total_reward = 0.0
    while True:
        perturbed_obs = perturber.apply(obs)
        action, _ = model.predict(perturbed_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if drone_sim:
            drone_sim.move_agents(info, action)

        if terminated or truncated:
            break

    if drone_sim:
        drone_sim.stop()

    return total_reward


def make_perturber(robot_type: str, N: int, gps: float, kin_lat: int, drop: float):
    if robot_type == "uav":
        return UavObsPerturber(
            N=N,
            gps_noise=gps,
            wind_latency_steps=kin_lat,
            coverage_drop_prob=drop,
        )
    return WheeledObsPerturber(
        N=N,
        gps_noise=gps,
        kin_latency_steps=kin_lat,
        goal_drop_prob=drop,
    )


def build_env(robot_type: str, render_coppelia: bool, num_robots: int):
    N = num_robots

    if robot_type == "uav":
        json_dict = read_uav_json(UAV_JSON_PATH)
        field_info = json_dict[f"set{ENV_VARIATION}"]
        env_id = "MultiUAVSim2Real-v0"
        if env_id not in gym.envs.registry:
            gym.register(id=env_id, entry_point=MultiUAV, max_episode_steps=1000)

        env_kwargs = dict(
            field_info=field_info,
            num_robots=N,
            max_steps=1000,
            render_mode="human" if render_coppelia else None,
            obs_mode="full",
            dr_mode="none",
            uncertainty_mode="full",
            reward_ablation="full",
        )
        expected_obs_dim = 4 * N + 1
    else:
        wheeled_dict = read_wheeled_json(WHEELED_JSON_PATH)
        env_config = wheeled_dict[f"set{ENV_VARIATION}"]
        env_id = "MultiWheeledSim2Real-v0"
        if env_id not in gym.envs.registry:
            gym.register(id=env_id, entry_point=MultiWheeled, max_episode_steps=1000)

        env_kwargs = dict(
            env_params=env_config,
            num_robots=N,
            max_steps=1000,
            render_mode=None,
            obs_mode="full",
            uncertainty_mode="wind_only",
            dr_mode="wind",
            reward_ablation="full",
        )
        expected_obs_dim = 5 * N + 1

    env = gym.make(env_id, **env_kwargs)
    actual_obs_dim = env.observation_space.shape[0]
    assert actual_obs_dim == expected_obs_dim, (
        f"{robot_type}: obs dim mismatch — expected {expected_obs_dim}, got {actual_obs_dim}"
    )
    print(f"[{robot_type}] obs_dim = {actual_obs_dim}  OK")
    return env


def run_platform_analysis(
    robot_type: str,
    model_path: str | None,
    render_coppelia: bool,
    num_robots: int | None = None,
) -> dict[str, float]:
    resolved_model = resolve_model_path(robot_type, model_path)
    model = CrossQ.load(resolved_model)
    N = num_robots if num_robots is not None else infer_num_robots_from_model(model, robot_type)
    print(f"[{robot_type}] Loaded model from {resolved_model}  (N={N})")

    env = build_env(robot_type, render_coppelia and robot_type == "uav", N)

    drone_sim = None
    if robot_type == "uav" and render_coppelia:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient

        client = RemoteAPIClient()
        sim_obj = client.getObject("sim")
        sim_obj.setInt32Param(sim_obj.intparam_idle_fps, 0)
        drone_sim = DroneSimulator(
            env, sim_obj, scaling_factor=8, height=HEIGHT, num_robots=N)
        drone_sim.draw_field()

    conditions = build_conditions(robot_type)
    results: dict[str, float] = {}

    for label, gps, kin_lat, drop in conditions:
        print(f"\n{'=' * 60}")
        print(f"[{robot_type.upper()}] Condition : {label}")
        print(
            f"  gps_noise={gps}  kin_latency_steps={kin_lat}  "
            f"drop_prob={drop}"
        )
        print(f"  Running {N_EVAL_EPISODES} episodes ...")

        perturber = make_perturber(robot_type, N, gps, kin_lat, drop)
        episode_rewards = []
        for ep in range(N_EVAL_EPISODES):
            r = run_episode(env, model, perturber, drone_sim=drone_sim)
            episode_rewards.append(r)
            print(f"  ep {ep + 1:3d}/{N_EVAL_EPISODES}  reward={r:.2f}")

        iqm = compute_iqm(episode_rewards)
        results[label] = iqm
        print(f"  -> IQM = {iqm:.2f}")

    table = format_results_table(robot_type, results, conditions)
    print(f"\n{table}")

    out_path = f"sim2real_{robot_type}.out"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\n[{robot_type}] Results saved to {out_path}")

    env.close()
    return results


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Sim-to-real observation-gap analysis (UAV and/or wheeled)."
    )
    p.add_argument(
        "--robot_type",
        choices=["uav", "wheeled", "both"],
        default="both",
        help="Which platform(s) to evaluate (default: both).",
    )
    p.add_argument(
        "--model_path",
        default=None,
        help="Override checkpoint path (applies to the selected platform only).",
    )
    p.add_argument(
        "--uav_model_path",
        default=None,
        help="UAV checkpoint override (used when --robot_type is uav or both).",
    )
    p.add_argument(
        "--wheeled_model_path",
        default=None,
        help="Wheeled checkpoint override (used when --robot_type is wheeled or both).",
    )
    p.add_argument(
        "--render_coppelia",
        action="store_true",
        help="Visual UAV run in CoppeliaSim (UAV only; one episode at a time).",
    )
    p.add_argument(
        "--num_robots",
        type=int,
        default=None,
        help="Override robot count (default: infer from checkpoint obs dim).",
    )
    p.add_argument(
        "--n_eval_episodes",
        type=int,
        default=N_EVAL_EPISODES,
        help=f"Episodes per perturbation condition (default: {N_EVAL_EPISODES}).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    global N_EVAL_EPISODES
    N_EVAL_EPISODES = args.n_eval_episodes

    if args.render_coppelia and args.robot_type == "wheeled":
        print("Warning: --render_coppelia applies to UAV only; ignoring for wheeled.")

    platforms = []
    if args.robot_type in ("uav", "both"):
        platforms.append("uav")
    if args.robot_type in ("wheeled", "both"):
        platforms.append("wheeled")

    for robot_type in platforms:
        if robot_type == "uav":
            model_path = args.uav_model_path or (
                args.model_path if args.robot_type == "uav" else None
            )
        else:
            model_path = args.wheeled_model_path or (
                args.model_path if args.robot_type == "wheeled" else None
            )

        run_platform_analysis(
            robot_type,
            model_path=model_path,
            render_coppelia=args.render_coppelia,
            num_robots=args.num_robots,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
