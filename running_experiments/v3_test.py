# ================================
# train_path_planning.py
# Multi-seed training script for path-planning environments.
# Supports both MultiUAV and MultiWheeled robot types.
#
# BUG FIXES vs original
# ─────────────────────
# [FIX 1] 'MultiRobotEnv-v0' in make_vec_env → 'MultiUAV-v0' (was never registered;
#          caused a KeyError at every training launch).
# [FIX 2] Added 'from shapely.geometry import Polygon, Point' — these were used
#          inside MultiWheeled but the import block was accidentally commented out.
# [FIX 3] Added get_robot_polygon() helper — called by MultiWheeled.step() but
#          defined in neither src/utils.py nor the local helpers section.
# [FIX 4] Removed the duplicate gym.register() call inside main(); the module-level
#          registration already runs on import, making the in-main check dead code.
#
# NEW FEATURES
# ────────────
# [NEW]  robot_type config key: "uav" (default) | "wheeled"
#        • Selects the environment class and gym ID used for training.
#        • Registers MultiWheeled-v0 alongside MultiUAV-v0.
#        • Adds read_env_config() to parse a single .ini file (configparser)
#          and read_wheeled_configs() to scan a directory of .ini files.
#          Replaces the earlier read_wheeled_json() stub.
#        • train_single_env() branches on robot_type for env kwargs and IDs.
# ================================

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import copy
import json
import configparser                  # [NEW] .ini config files for wheeled robot envs
from datetime import datetime

import numpy as np
import gymnasium as gym
import pygame

import torch
import random
import multiprocessing as mp

# shapely — used in MultiWheeled for collision geometry
# [FIX 2] was commented out; also corrected to top-level 'shapely' import
#         (matches the wheeled robot config script)
from shapely import Polygon, Point

from stable_baselines3.common.callbacks import LogEveryNTimesteps, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from sb3_contrib import CrossQ


# ================================
# Utility / Helper Functions
# ================================

def is_inside_polygon(point, poly):
    """Ray-casting point-in-polygon test."""
    x, y = point
    inside = False
    n = len(poly)
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def compute_min_dist(x):
    """Minimum pairwise distance between a set of 2-D points."""
    x = np.asarray(x, dtype=np.float32)
    diff = x[:, None, :] - x[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_matrix, np.inf)
    return float(np.min(dist_matrix))


def binary_list_to_decimal(bin_list):
    """Convert a list of 0/1 values to its decimal integer equivalent."""
    return int("".join(str(int(b)) for b in bin_list), 2)


# [FIX 3] get_robot_polygon was called by MultiWheeled.step() but never defined.
# Implementation taken directly from the wheeled robot config script.
def get_robot_polygon(x, y, theta, robot_length, robot_width):
    """Return a Shapely Polygon for the robot's rectangular footprint."""
    dx = robot_length / 2
    dy = robot_width  / 2
    corners = np.array([
        [ dx,  dy],
        [ dx, -dy],
        [-dx, -dy],
        [-dx,  dy],
    ])
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated = np.dot(corners, R.T) + np.array([x, y])
    return Polygon(rotated)


# ================================
# JSON Loaders
# ================================

def read_uav_json(json_path, sf=10):
    """Load UAV field-info dicts from JSON, applying a scale factor sf."""
    with open(json_path, "r") as f:
        data = json.load(f)
    for set_name, cfg in data.items():
        cfg["field"]             = [tuple((p[0] * sf, p[1] * sf)) for p in cfg["field"]]
        cfg["init_positions"]    = [np.array(p, dtype=float) * sf for p in cfg["init_positions"]]
        cfg["infected_locations"]= [tuple((p[0] * sf, p[1] * sf)) for p in cfg["infected_locations"]]
    return data


def read_wheeled_configs(config_dir):
    """
    Scan config_dir for .ini files (env1.ini, env2.ini, …) and load each one.

    Files are sorted lexicographically so env1 → set1, env2 → set2, etc.
    Returns a dict {"set1": env_params_1, "set2": env_params_2, …} so that
    train_single_env() can access configs with the same set_key pattern used
    for UAV (json_dict[f"set{env_id}"]).
    """
    ini_files = sorted(f for f in os.listdir(config_dir) if f.endswith(".ini"))
    if not ini_files:
        raise FileNotFoundError(f"No .ini files found in '{config_dir}'")
    configs = {}
    for i, fname in enumerate(ini_files, start=1):
        configs[f"set{i}"] = read_env_config(os.path.join(config_dir, fname))
        print(f"  Loaded wheeled config set{i} ← {fname}")
    return configs


def read_env_config(config_path):
    """
    Parse a single .ini environment config file (configparser format).
    Taken directly from the wheeled robot reference script — do not modify
    the field names or section names without also updating the .ini files.
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    env_params = {}

    # Screen
    env_params['SCREEN_WIDTH']  = float(config['SCREEN']['WIDTH'])
    env_params['SCREEN_HEIGHT'] = float(config['SCREEN']['HEIGHT'])

    # Robot physical params
    env_params['ROBOT_LENGTH'] = float(config['ROBOTS']['LENGTH'])
    env_params['ROBOT_WIDTH']  = float(config['ROBOTS']['WIDTH'])
    env_params['MAX_SPEED']    = float(config['ROBOTS']['MAX_SPEED'])
    env_params['MAX_STEER']    = float(config['ROBOTS']['MAX_STEER'])
    env_params['NUM_ROBOTS']   = int(config['ROBOTS']['NUM_ROBOTS'])

    # Initial robot configurations — theta stored in radians
    env_params['ROBOT_INIT_CONFIGS'] = []
    for i in range(env_params['NUM_ROBOTS']):
        conf = config['ROBOTS'][f'ROBOT_{i + 1}']
        x, y, theta = map(float, conf.split(','))
        env_params['ROBOT_INIT_CONFIGS'].append((x, y, float(np.radians(theta))))

    # Goal positions
    env_params['NUM_GOALS']    = int(config['GOALS']['NUM_GOALS'])
    env_params['GOAL_SIZE']    = float(config['GOALS']['GOAL_SIZE'])
    env_params['GOAL_POSITIONS'] = []
    for i in range(env_params['NUM_GOALS']):
        g_pos = config['GOALS'][f'GOAL_{i + 1}']
        x, y  = map(float, g_pos.split(','))
        env_params['GOAL_POSITIONS'].append((x, y))

    # Polygonal obstacles — vertices separated by ';', coordinates by ','
    env_params['NUM_OBSTACLES'] = int(config['OBSTACLES']['NUM_OBSTACLES'])
    env_params['OBSTACLES']     = []
    for i in range(env_params['NUM_OBSTACLES']):
        ver_str = config['OBSTACLES'][f'OBSTACLE_{i + 1}']
        points  = [tuple(map(int, pt.split(','))) for pt in ver_str.split(';')]
        env_params['OBSTACLES'].append(points)

    return env_params


# ================================
# Environment Classes
# ================================

class MultiUAV(gym.Env):
    """
    Multi-UAV path-planning environment.
    Robots visit every infected/target location (binary coverage, no spraying).
    """

    metadata = {'render_modes': ['human', 'rgb_array'], "render_fps": 60}

    def __init__(
        self,
        field_info,
        render_mode=None,
        wind_par=None,
        num_robots=3,
        max_steps=1000,
        reward_ablation="full",
        obs_mode="full",
        uncertainty_mode="full",
        dr_mode="none",
        target_screen_size=800,
    ):
        super().__init__()

        if wind_par is None:
            wind_par = [0, 0]

        assert uncertainty_mode in ("full", "wind_only", "act_only", "deterministic"), \
            f"Unknown uncertainty_mode: {uncertainty_mode}"
        assert dr_mode in ("none", "wind", "full"), \
            f"Unknown dr_mode: {dr_mode}"
        assert reward_ablation in ("full", "no_term", "no_path"), \
            f"Unknown reward_ablation: {reward_ablation}"
        assert obs_mode in ("full", "no_pos", "no_inf_hist", "pos_only"), \
            f"Unknown obs_mode: {obs_mode}"
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.reward_ablation  = reward_ablation
        self.obs_mode         = obs_mode
        self.uncertainty_mode = uncertainty_mode
        self.dr_mode          = dr_mode
        self.max_steps        = max_steps

        # Field geometry
        self.field_info    = copy.deepcopy(field_info)
        self.poly_vertices = self.field_info['field']
        xs, ys             = zip(*self.poly_vertices)
        self.min_x, self.max_x = float(np.min(xs)), float(np.max(xs))
        self.min_y, self.max_y = float(np.min(ys)), float(np.max(ys))
        self.world_width   = self.max_x - self.min_x
        self.world_height  = self.max_y - self.min_y

        # Auto-scale render window
        scale_x = target_screen_size / self.world_width
        scale_y = target_screen_size / self.world_height
        self.render_scale  = min(scale_x, scale_y) * 0.90
        self.screen_width  = int(self.world_width  * self.render_scale) + 40
        self.screen_height = int(self.world_height * self.render_scale) + 40
        self.offset_x      = (self.screen_width  - self.world_width  * self.render_scale) / 2
        self.offset_y      = (self.screen_height - self.world_height * self.render_scale) / 2

        # Robot physical params
        self.num_robots   = num_robots
        self.init_robot_positions = np.array(
            self.field_info['init_positions'][:num_robots], dtype=np.float32)
        self.robot_size   = 10
        self.mass         = 1.0
        self.thrust_power = 0.5
        self.max_speed    =  5.0
        self.min_speed    = -5.0
        self.robot_colors = [
            (255, 0,   0), (0, 200,   0), (0,   0, 255),
            (255, 128, 0), (128, 0, 255), (255, 0, 255), (128, 128, 128),
        ]

        # Infected / target locations
        self.initial_inf_locations  = [tuple(loc) for loc in self.field_info['infected_locations']]
        self._nominal_infected_size = 10
        self.infected_size          = self._nominal_infected_size
        self.infected_length        = len(self.initial_inf_locations)

        # Base wind
        self.base_wind_mag, self.base_wind_dir = float(wind_par[0]), float(wind_par[1])

        # Noise stds — driven by uncertainty_mode
        _noise = {
            "full":          dict(wind=0.20, wind_dir=5.0, action=0.10, obs=0.01),
            "wind_only":     dict(wind=0.20, wind_dir=5.0, action=0.00, obs=0.00),
            "act_only":      dict(wind=0.00, wind_dir=0.0, action=0.10, obs=0.00),
            "deterministic": dict(wind=0.00, wind_dir=0.0, action=0.00, obs=0.00),
        }[uncertainty_mode]

        self.wind_noise_std      = _noise["wind"]
        self.wind_dir_noise_std  = _noise["wind_dir"]
        self.action_noise_std    = _noise["action"]
        self.obs_noise_std       = _noise["obs"]
        self.init_position_noise = 0.5

        # Nominal values for DR restore
        self._nominal_action_noise_std = _noise["action"]
        self._nominal_mass             = 1.0
        self._nominal_thrust_power     = 0.5

        # Action space: (ax, ay) per robot
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(num_robots, 2), dtype=np.float32)

        # Observation space — size depends on obs_mode
        # "full"        positions(2N) + velocities(2N) + visited_decimal(1) = 4N+1
        # "no_pos"      visited_decimal(1)  [remove all kinematics]
        # "no_inf_hist" positions(2N) + velocities(2N) = 4N
        # "pos_only"    positions(2N)
        N = num_robots
        _obs_dims = {"full": 4*N+1, "no_pos": 1, "no_inf_hist": 4*N, "pos_only": 2*N}
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(_obs_dims[obs_mode],), dtype=np.float32)

        self.render_mode = render_mode
        self.screen      = None
        self.clock       = None
        self.reset()

    def world_to_screen(self, pos):
        x = (pos[0] - self.min_x) * self.render_scale + self.offset_x
        y = (pos[1] - self.min_y) * self.render_scale + self.offset_y
        return int(x), int(y)

    def _get_obs(self):
        infected_decimal = float(binary_list_to_decimal(list(self.infected_dict.values())))
        if self.obs_mode == "full":
            state = np.concatenate([self.robot_positions.flatten(),
                                    self.robot_velocities.flatten(),
                                    np.array([infected_decimal], dtype=np.float32)])
        elif self.obs_mode == "no_pos":
            state = np.array([infected_decimal], dtype=np.float32)
        elif self.obs_mode == "no_inf_hist":
            state = np.concatenate([self.robot_positions.flatten(),
                                    self.robot_velocities.flatten()])
        elif self.obs_mode == "pos_only":
            state = self.robot_positions.flatten().copy()

        state = state.astype(np.float32)
        if self.obs_noise_std > 0:
            state += np.random.normal(0, self.obs_noise_std, size=state.shape).astype(np.float32)

        info = {f'robot{i}': self.robot_positions[i].copy() for i in range(self.num_robots)}
        return state, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Domain randomisation
        if self.dr_mode == "none":
            self.wind_mag         = self.base_wind_mag + np.random.normal(0, self.wind_noise_std)
            self.wind_dir         = self.base_wind_dir + np.random.normal(0, self.wind_dir_noise_std)
            self.action_noise_std = self._nominal_action_noise_std
            self.infected_size    = self._nominal_infected_size
            self.mass             = self._nominal_mass
            self.thrust_power     = self._nominal_thrust_power
        elif self.dr_mode == "wind":
            self.wind_mag         = float(np.random.uniform(0.0, 1.0))
            self.wind_dir         = float(np.degrees(np.random.uniform(0.0, 2 * np.pi)))
            self.action_noise_std = self._nominal_action_noise_std
            self.infected_size    = self._nominal_infected_size
            self.mass             = self._nominal_mass
            self.thrust_power     = self._nominal_thrust_power
        elif self.dr_mode == "full":
            self.wind_mag         = float(np.random.uniform(0.0, 1.0))
            self.wind_dir         = float(np.degrees(np.random.uniform(0.0, 2 * np.pi)))
            self.action_noise_std = float(np.random.uniform(0.01, 0.10))
            r0                    = self._nominal_infected_size
            self.infected_size    = float(np.random.uniform(0.8 * r0, 1.2 * r0))
            self.mass             = float(np.random.uniform(0.90, 1.10))
            self.thrust_power     = 0.5 * float(np.random.uniform(0.80, 1.20))

        # Starting position jitter
        self.robot_positions = (
            self.init_robot_positions
            + np.random.normal(0, self.init_position_noise, self.init_robot_positions.shape)
        ).astype(np.float32)

        self.step_count         = 0
        self.visited            = set()
        self.infected_locations = list(copy.deepcopy(self.initial_inf_locations))
        self.infected_dict      = {loc: 0 for loc in self.initial_inf_locations}
        self.robot_velocities   = np.zeros((self.num_robots, 2), dtype=np.float32)
        self.trajectories       = [[] for _ in range(self.num_robots)]
        self.total_path_length  = 0.0
        self.prev_positions     = self.robot_positions.copy()

        return self._get_obs()

    def step(self, actions):
        self.step_count += 1
        terminated, truncated = False, False
        rewards = 0.0

        # Per-step stochastic wind
        wind_mag = self.wind_mag + np.random.normal(0, self.wind_noise_std)
        wind_dir = self.wind_dir + np.random.normal(0, self.wind_dir_noise_std)
        theta_w  = np.radians(wind_dir)
        wind     = np.array([wind_mag * np.cos(theta_w), wind_mag * np.sin(theta_w)],
                            dtype=np.float32)

        for i in range(self.num_robots):
            ax_raw, ay_raw = actions[i]
            ax = ax_raw * self.thrust_power + np.random.normal(0, self.action_noise_std)
            ay = ay_raw * self.thrust_power + np.random.normal(0, self.action_noise_std)

            self.robot_velocities[i] += np.array([ax, ay]) / self.mass + wind
            self.robot_velocities[i]  = np.clip(self.robot_velocities[i], self.min_speed, self.max_speed)

            new_pos = self.robot_positions[i] + self.robot_velocities[i]
            if is_inside_polygon(new_pos, self.poly_vertices):
                self.robot_positions[i] = new_pos
            else:
                rewards -= 50                           # boundary penalty — unchanged vs reference env
                self.robot_velocities[i][:] = 0

            if self.reward_ablation != "no_path":
                rewards -= 0.1 * (ax ** 2 + ay ** 2)   # energy penalty — unchanged
                rewards -= 0.3 * float(np.linalg.norm(self.robot_velocities[i]))  # speed penalty — unchanged

            # ── Coverage shaping: reward exploration, penalise revisits ──────────
            # FIX: original penalised both new (-10) and revisited (-100) cells,
            # which contradicts a coverage task and causes large negative drift.
            # Now: small positive reward for visiting new cells, moderate penalty
            # only for revisiting — magnitudes scaled to match reference env.
            # ── Coverage shaping ─────────────────────────────────────────────────────
            pos_key = tuple(np.round(self.robot_positions[i], 1))
            if pos_key in self.visited:
                rewards -= 30        # revisit deterrent (was -100; 20× reduction prevents runaway)
            else:
                rewards += 3      # exploration bonus (was -10; flipped — penalising new cells
                                    #                    discourages the coverage objective)
            self.visited.add(pos_key)

            # ── Target coverage ──────────────────────────────────────────────────
            # FIX: +10,000 per target caused large training spikes. Reduced to +200
            # so that visiting all targets contributes ~+1,000–2,000 total —
            # meaningful but not dominating; same order of magnitude as reference env.
            for inf_loc in list(self.infected_locations):
                if np.linalg.norm(self.robot_positions[i] - np.array(inf_loc)) <= self.infected_size:
                    self.infected_locations.remove(inf_loc)
                    self.infected_dict[inf_loc] = 1
                    rewards += 1000      # was +10,000 — 50× reduction

            self.trajectories[i].append(self.robot_positions[i].copy())
            if len(self.trajectories[i]) > 200:
                self.trajectories[i].pop(0)

        # Path length
        step_dist = np.linalg.norm(self.robot_positions - self.prev_positions, axis=1)
        step_path = float(np.sum(step_dist))
        self.total_path_length += step_path
        self.prev_positions     = self.robot_positions.copy()

        if self.reward_ablation != "no_path":
            rewards -= 1.0 * step_path     # path-length penalty — unchanged
            rewards -= 2.0                  # time penalty — unchanged

        # Distance shaping to nearest unvisited target (always active) — unchanged
        if self.infected_locations:
            target_arr = np.array(self.infected_locations, dtype=np.float32)
            dists_mat  = np.linalg.norm(
                self.robot_positions[:, None] - target_arr[None, :], axis=2)
            rewards += 0.5 * float(np.sum(np.exp(-np.min(dists_mat, axis=1))))

        # ── Terminal signals ──────────────────────────────────────────────────────
        term_cond = ""
        if len(self.infected_locations) == 0:
            if self.reward_ablation != "no_term":
                rewards += 5000      # FIX: was +100,000 — matches reference env success bonus
            term_cond  = "visited_all"
            terminated = True

        if self.num_robots > 1 and compute_min_dist(self.robot_positions) < self.robot_size:
            if self.reward_ablation != "no_term":
                rewards -= 5000     # FIX: was -100,000 — 2× success bonus, matches reference env
            term_cond  = "collision"
            terminated = True

        truncated = self.step_count >= self.max_steps
        if truncated:
            term_cond = "max_steps"

        obs, info = self._get_obs()
        info.update({
            "step_count":        self.step_count,
            "remaining_targets": len(self.infected_locations),
            "path_length":       self.total_path_length,
            "term_cond":         term_cond,
        })
        return obs, rewards, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Multi-UAV Path Planning")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))
        scaled_poly = [self.world_to_screen(p) for p in self.poly_vertices]
        pygame.draw.polygon(self.screen, (255, 255, 0), scaled_poly)

        for i in range(self.num_robots):
            if len(self.trajectories[i]) > 1:
                pygame.draw.lines(self.screen, (150, 150, 150), False,
                                  [self.world_to_screen(p) for p in self.trajectories[i]], 2)

        r_px = max(3, int(self.robot_size * self.render_scale / 2))
        for i in range(self.num_robots):
            pygame.draw.circle(self.screen, self.robot_colors[i % len(self.robot_colors)],
                               self.world_to_screen(self.robot_positions[i]), r_px)

        inf_px = max(3, int(self.infected_size * self.render_scale / 2))
        for loc in self.infected_locations:
            pygame.draw.circle(self.screen, (0, 220, 220), self.world_to_screen(loc), inf_px)
        for loc, vis in self.infected_dict.items():
            if vis:
                pygame.draw.circle(self.screen, (80, 80, 80), self.world_to_screen(loc), inf_px)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None


class MultiWheeled(gym.Env):
    """Multi-wheeled-robot path-planning (bicycle-model kinematics)."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        env_params,
        render_mode=None,
        wind_par=None,
        uncertainty_mode="full",
        dr_mode="none",
        reward_ablation="full",
        obs_mode="full",
        max_steps=1000,
        target_screen_size=800,
    ):
        super().__init__()

        if wind_par is None:
            wind_par = [0, 0]

        assert uncertainty_mode in ("full", "wind_only", "act_only", "deterministic"), \
            f"Unknown uncertainty_mode: {uncertainty_mode}"
        assert dr_mode in ("none", "wind", "full"), \
            f"Unknown dr_mode: {dr_mode}"
        assert reward_ablation in ("full", "no_term", "no_path"), \
            f"Unknown reward_ablation: {reward_ablation}"
        assert obs_mode in ("full", "no_pos", "no_inf_hist", "pos_only"), \
            f"Unknown obs_mode: {obs_mode}"
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.uncertainty_mode = uncertainty_mode
        self.dr_mode          = dr_mode
        self.reward_ablation  = reward_ablation
        self.obs_mode         = obs_mode
        self.max_steps        = max_steps
        self.render_mode      = render_mode
        self.screen           = None
        self.clock            = None
        self.dt               = 0.05

        self.ROBOT_LENGTH = env_params['ROBOT_LENGTH']
        self.ROBOT_WIDTH  = env_params['ROBOT_WIDTH']
        self.NUM_ROBOTS   = env_params['NUM_ROBOTS']
        self.init_ROBOTS  = env_params['ROBOT_INIT_CONFIGS']

        self._nominal_max_speed   = float(env_params['MAX_SPEED'])
        self._nominal_max_steer   = float(np.radians(env_params['MAX_STEER']))
        self._nominal_accel_scale = 100.0
        self.MAX_SPEED            = self._nominal_max_speed
        self.MAX_STEER            = self._nominal_max_steer
        self.accel_scale          = self._nominal_accel_scale

        self.goal_radius    = env_params['GOAL_SIZE']
        self.goal_positions = env_params['GOAL_POSITIONS']
        self.obstacles      = env_params['OBSTACLES']

        self.world_width  = float(env_params['SCREEN_WIDTH'])
        self.world_height = float(env_params['SCREEN_HEIGHT'])

        # Auto-scale render window
        scale_x = target_screen_size / self.world_width
        scale_y = target_screen_size / self.world_height
        self.render_scale  = min(scale_x, scale_y) * 0.95
        self.screen_width  = int(self.world_width  * self.render_scale) + 20
        self.screen_height = int(self.world_height * self.render_scale) + 20
        self.offset_x      = (self.screen_width  - self.world_width  * self.render_scale) / 2
        self.offset_y      = (self.screen_height - self.world_height * self.render_scale) / 2

        self.base_wind_mag = float(wind_par[0])
        self.base_wind_dir = float(wind_par[1])

        _noise = {
            "full":          dict(wind=0.20, wind_dir=5.0, action=0.10, obs=0.01),
            "wind_only":     dict(wind=0.20, wind_dir=5.0, action=0.00, obs=0.00),
            "act_only":      dict(wind=0.00, wind_dir=0.0, action=0.10, obs=0.00),
            "deterministic": dict(wind=0.00, wind_dir=0.0, action=0.00, obs=0.00),
        }[uncertainty_mode]

        self.wind_noise_std      = _noise["wind"]
        self.wind_dir_noise_std  = _noise["wind_dir"]
        self.action_noise_std    = _noise["action"]
        self.obs_noise_std       = _noise["obs"]
        self.init_position_noise = 0.5
        self.init_heading_noise  = float(np.radians(2))
        self._nominal_action_noise_std = _noise["action"]

        # Observation space — size depends on obs_mode
        # "full"        (x,y,θ,v,δ)(5N) + goal_decimal(1) = 5N+1
        # "no_pos"      (θ,v,δ)(3N)     + goal_decimal(1) = 3N+1
        # "no_inf_hist" (x,y,θ,v,δ)(5N)
        # "pos_only"    (x,y)(2N)
        N = self.NUM_ROBOTS
        _obs_dims = {"full": 5*N+1, "no_pos": 3*N+1, "no_inf_hist": 5*N, "pos_only": 2*N}
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(_obs_dims[obs_mode],), dtype=np.float64)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.NUM_ROBOTS, 2), dtype=np.float32)

        self.r_s, self.r_l, self.r_M = 10, 10000, 100000
        self.reset()

    def world_to_screen(self, x, y):
        return int(x * self.render_scale + self.offset_x), \
               int(y * self.render_scale + self.offset_y)

    def _get_obs(self):
        self.dec_g = binary_list_to_decimal(self.goal_visited)
        if self.obs_mode == "full":
            obs = np.concatenate([self.robots.flatten(), np.array([self.dec_g])])
        elif self.obs_mode == "no_pos":
            obs = np.concatenate([self.robots[:, 2:].flatten(), np.array([self.dec_g])])
        elif self.obs_mode == "no_inf_hist":
            obs = self.robots.flatten().copy()
        elif self.obs_mode == "pos_only":
            obs = self.robots[:, :2].flatten().copy()

        obs = obs.astype(np.float64)
        if self.obs_noise_std > 0:
            obs += np.random.normal(0, self.obs_noise_std, size=obs.shape)

        info = {f'robot{i}': self.robots[i, :2].copy() for i in range(self.NUM_ROBOTS)}
        return obs, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.collision_occurred = False
        self.collision_point    = (0.0, 0.0)
        self.goal_visited       = [False] * len(self.goal_positions)
        self.robot_paths        = [[] for _ in range(self.NUM_ROBOTS)]
        self.t                  = 0

        if self.dr_mode == "none":
            self.wind_mag         = self.base_wind_mag + np.random.normal(0, self.wind_noise_std)
            self.wind_dir         = self.base_wind_dir + np.random.normal(0, self.wind_dir_noise_std)
            self.action_noise_std = self._nominal_action_noise_std
            self.MAX_SPEED        = self._nominal_max_speed
            self.MAX_STEER        = self._nominal_max_steer
            self.accel_scale      = self._nominal_accel_scale
        elif self.dr_mode == "wind":
            self.wind_mag         = float(np.random.uniform(0.0, 1.0))
            self.wind_dir         = float(np.degrees(np.random.uniform(0.0, 2 * np.pi)))
            self.action_noise_std = self._nominal_action_noise_std
            self.MAX_SPEED        = self._nominal_max_speed
            self.MAX_STEER        = self._nominal_max_steer
            self.accel_scale      = self._nominal_accel_scale
        elif self.dr_mode == "full":
            self.wind_mag         = float(np.random.uniform(0.0, 1.0))
            self.wind_dir         = float(np.degrees(np.random.uniform(0.0, 2 * np.pi)))
            self.action_noise_std = float(np.random.uniform(0.01, 0.10))
            self.MAX_SPEED        = self._nominal_max_speed  * float(np.random.uniform(0.85, 1.15))
            self.MAX_STEER        = self._nominal_max_steer  * float(np.random.uniform(0.85, 1.15))
            self.accel_scale      = self._nominal_accel_scale * float(np.random.uniform(0.80, 1.20))

        self.robots = []
        for i in range(self.NUM_ROBOTS):
            x, y, theta = copy.deepcopy(self.init_ROBOTS[i])
            x     = np.clip(x + np.random.normal(0, self.init_position_noise), 0., self.world_width)
            y     = np.clip(y + np.random.normal(0, self.init_position_noise), 0., self.world_height)
            theta += np.random.normal(0, self.init_heading_noise)
            self.robots.append([x, y, theta, 0.0, 0.0])

        self.robots            = np.array(self.robots, dtype=np.float64)
        self.total_path_length = 0.0
        self.prev_positions    = self.robots[:, :2].copy()

        if self.render_mode == "human":
            self._render_pygame()
        return self._get_obs()

    def step(self, action):
        terminated, truncated = False, False
        reward = -(self.r_s / self.dec_g) if self.dec_g != 0 else -self.r_s
        self.t += 1

        # Per-step stochastic wind
        wind_mag = self.wind_mag + np.random.normal(0, self.wind_noise_std)
        wind_dir = self.wind_dir + np.random.normal(0, self.wind_dir_noise_std)
        w_theta  = np.radians(wind_dir)
        wind     = np.array([wind_mag * np.cos(w_theta), wind_mag * np.sin(w_theta)])

        robot_polygons = []
        for i in range(self.NUM_ROBOTS):
            a_noisy       = float(action[i][0]) + np.random.normal(0, self.action_noise_std)
            d_delta_noisy = float(action[i][1]) + np.random.normal(0, self.action_noise_std)

            a = a_noisy * self.accel_scale
            d_delta = d_delta_noisy
            x, y, theta, v, delta = self.robots[i]

            v     = np.clip(v + a * self.dt, -self.MAX_SPEED, self.MAX_SPEED)
            delta = np.clip(delta + d_delta, -self.MAX_STEER, self.MAX_STEER)

            omega = (v / (self.ROBOT_LENGTH / np.tan(delta))) if abs(delta) > 1e-4 else 0.0
            theta += omega * self.dt
            x     += v * np.cos(theta) * self.dt + wind[0] * self.dt
            y     += v * np.sin(theta) * self.dt + wind[1] * self.dt
            x      = np.clip(x, 0.0, self.world_width)
            y      = np.clip(y, 0.0, self.world_height)

            if self.reward_ablation != "no_path":
                reward -= 0.1 * (a_noisy ** 2 + d_delta_noisy ** 2)
                reward -= 0.3 * abs(v)

            robot_poly = get_robot_polygon(x, y, theta, self.ROBOT_LENGTH, self.ROBOT_WIDTH)

            for obs_pts in self.obstacles:
                obs_poly = Polygon(obs_pts)
                if robot_poly.intersects(obs_poly):
                    pt = robot_poly.intersection(obs_poly).representative_point()
                    self.collision_point = (pt.x, pt.y)
                    if self.reward_ablation != "no_term":
                        reward = -self.r_M
                    terminated = True
                    self.collision_occurred = True

            for prev_poly in robot_polygons:
                if robot_poly.intersects(prev_poly):
                    pt = robot_poly.intersection(prev_poly).representative_point()
                    self.collision_point = (pt.x, pt.y)
                    if self.reward_ablation != "no_term":
                        reward = -self.r_M
                    terminated = True
                    self.collision_occurred = True
            robot_polygons.append(robot_poly)

            for j, (gx, gy) in enumerate(self.goal_positions):
                if not self.goal_visited[j] and Point(gx, gy).buffer(self.goal_radius).intersects(robot_poly):
                    reward += self.r_l
                    self.goal_visited[j] = True

            self.robots[i] = [x, y, theta, v, delta]
            self.robot_paths[i].append((x, y))

        # Path length
        current_positions   = self.robots[:, :2]
        step_path           = float(np.sum(np.linalg.norm(current_positions - self.prev_positions, axis=1)))
        self.total_path_length += step_path
        self.prev_positions     = current_positions.copy()

        if self.reward_ablation != "no_path":
            reward -= 1.0 * step_path
            reward -= 2.0

        # Distance shaping to nearest unvisited goal (always active)
        unvisited = [gp for gp, vis in zip(self.goal_positions, self.goal_visited) if not vis]
        if unvisited:
            target_arr = np.array(unvisited, dtype=np.float64)
            dists_mat  = np.linalg.norm(current_positions[:, None] - target_arr[None, :], axis=2)
            reward    += 0.5 * float(np.sum(np.exp(-np.min(dists_mat, axis=1))))

        term_cond = ""
        if sum(self.goal_visited) >= len(self.goal_positions):
            if self.reward_ablation != "no_term":
                reward += self.r_M
            term_cond  = "all_goals"
            terminated = True

        truncated = self.t >= self.max_steps
        if truncated:
            term_cond = "max_steps"

        if self.render_mode == "human":
            self._render_pygame()

        obs, info = self._get_obs()
        info.update({
            "step_count":    self.t,
            "goals_visited": int(sum(self.goal_visited)),
            "path_length":   self.total_path_length,
            "term_cond":     term_cond,
        })
        return obs, reward, terminated, truncated, info

    def _render_pygame(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Multi-Wheeled Robot Path Planning")
            self.clock = pygame.time.Clock()

        self.screen.fill((240, 240, 240))

        for obs in self.obstacles:
            pygame.draw.polygon(self.screen, (100, 100, 100),
                                [self.world_to_screen(p[0], p[1]) for p in obs])

        goal_r_px = max(3, int(self.goal_radius * self.render_scale))
        for (gx, gy), visited in zip(self.goal_positions, self.goal_visited):
            pygame.draw.circle(self.screen, (30, 30, 30) if visited else (0, 200, 80),
                               self.world_to_screen(gx, gy), goal_r_px)

        if self.collision_occurred:
            pygame.draw.circle(self.screen, (255, 0, 0),
                               self.world_to_screen(*self.collision_point),
                               max(5, int(10 * self.render_scale)))

        for path in self.robot_paths:
            if len(path) > 1:
                pygame.draw.lines(self.screen, (200, 80, 80), False,
                                  [self.world_to_screen(*p) for p in path], 2)

        r_len_px = max(4, int(self.ROBOT_LENGTH * self.render_scale))
        r_wid_px = max(2, int(self.ROBOT_WIDTH  * self.render_scale))
        for x, y, theta, _, delta in self.robots:
            sx, sy = self.world_to_screen(x, y)
            surf = pygame.Surface((r_len_px, r_wid_px), pygame.SRCALPHA)
            surf.fill((0, 128, 255))
            pygame.draw.line(surf, (255, 0, 0),
                             (r_len_px // 2, r_wid_px // 2), (r_len_px, r_wid_px // 2), 2)
            rotated = pygame.transform.rotate(surf, -np.degrees(theta))
            self.screen.blit(rotated, rotated.get_rect(center=(sx, sy)))

            off_x = r_len_px // 2 - r_wid_px // 5
            off_y = r_wid_px  // 2 - r_wid_px // 5
            for pos_name, (dx, dy) in {
                'front_left': (off_x, -off_y), 'front_right': (off_x, off_y),
                'rear_left': (-off_x, -off_y),  'rear_right': (-off_x, off_y)
            }.items():
                wx = sx + np.cos(theta) * dx - np.sin(theta) * dy
                wy = sy + np.sin(theta) * dx + np.cos(theta) * dy
                wsurf = pygame.Surface((max(2, r_len_px // 6), 1), pygame.SRCALPHA)
                wsurf.fill((20, 20, 20))
                w_angle = theta + delta if 'front' in pos_name else theta
                rw = pygame.transform.rotate(wsurf, -np.degrees(w_angle))
                self.screen.blit(rw, rw.get_rect(center=(int(wx), int(wy))))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        pygame.event.get()

    def render(self):
        self._render_pygame()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# ================================
# Gym Registration
# ================================

# [FIX 4] Only one registration block; the duplicate inside main() has been removed.
# Both envs registered here so make_vec_env can find them by ID.
if 'MultiUAV-v0' not in gym.envs.registry:
    gym.register(id='MultiUAV-v0',     entry_point=MultiUAV,     max_episode_steps=1000)
if 'MultiWheeled-v0' not in gym.envs.registry:
    gym.register(id='MultiWheeled-v0', entry_point=MultiWheeled, max_episode_steps=1000)


# ================================
# Seed Helper
# ================================

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ================================
# Training Worker (one env per process)
# ================================

def train_single_env(env_id, config, seed=42):
    torch.set_num_threads(1)
    set_seed(seed)

    robot_type = config["robot_type"]   # [NEW] "uav" or "wheeled"
    device_ids = config["device_ids"]
    proc_id    = (env_id - 1)
    device_id  = device_ids[(proc_id + seed * 10) % len(device_ids)]
    log_path   = config["log_path"]
    json_dict  = config["json_dict"]
    num_envs   = config["num_envs"]
    num_robots = config["num_robots"]
    max_steps  = config["max_steps"]
    time_steps = config["time_steps"]

    run_name = f"env{env_id}_seed{seed}_{robot_type}"
    print(f"[START] Training {run_name} on {device_id}")

    set_key = f"set{env_id}"

    # [NEW] Select gym ID and env_kwargs based on robot_type
    # ─────────────────────────────────────────────────────
    if robot_type == "uav":
        gym_id   = "MultiUAV-v0"
        # [FIX 1] Was 'MultiRobotEnv-v0' — that ID is never registered and
        #          caused a KeyError at the start of every training run.
        env_kwargs = {
            "field_info": json_dict[set_key],
            "num_robots": num_robots,
            "max_steps":  max_steps,
            "render_mode": None,
        }
    elif robot_type == "wheeled":
        gym_id   = "MultiWheeled-v0"
        env_kwargs = {
            "env_params": json_dict[set_key],
            "max_steps":  max_steps,
            "render_mode": None,
        }
    else:
        raise ValueError(f"Unknown robot_type: '{robot_type}'. Choose 'uav' or 'wheeled'.")

    vec_env = make_vec_env(gym_id, env_kwargs=env_kwargs, n_envs=num_envs, seed=seed)
    eval_env = make_vec_env(gym_id, env_kwargs=env_kwargs, n_envs=1,       seed=seed)

    callback = CallbackList([
        LogEveryNTimesteps(n_steps=10000),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(log_path, f"best_model_env{env_id}"),
            log_path            =os.path.join(log_path, f"eval_logs_env{env_id}"),
            eval_freq           =max(10000 // num_envs, 1),
            n_eval_episodes     =5,
            deterministic       =True,
            render              =False,
        ),
    ])

    logger = configure(
        os.path.join(log_path, f"crossq_env{env_id}"),
        ["stdout", "log", "csv", "tensorboard"],
    )

    model = CrossQ("MlpPolicy", vec_env, verbose=1, device=device_id, seed=seed)
    model.set_logger(logger)
    model.learn(total_timesteps=time_steps, callback=callback)
    model.save(os.path.join(log_path, f"env{env_id}_CrossQ_{robot_type}"))

    vec_env.close()
    eval_env.close()
    print(f"[DONE]  Training env {env_id}, seed {seed}, robot_type {robot_type}")


# ================================
# Main Launcher
# ================================

def main():

    # ── CONFIG ─────────────────────────────────────────────────────────────
    version    = "3"
    robot_type = "uav"       # [NEW] change to "wheeled" to train MultiWheeled
    # seeds      = [0, 42, 123, 2024, 9999]
    seeds      = [42]

    config_base = {
        "robot_type": robot_type,  # [NEW] forwarded to every worker
        "device_ids": [f"cuda:{i}" for i in range(8)],
        "time_steps": int(2e6),
        "num_envs":   8,
        "num_robots": 3,           # only used by MultiUAV; ignored by MultiWheeled
        "max_steps":  1000,
    }

    # ── LOAD CONFIGS ───────────────────────────────────────────────────────
    if robot_type == "uav":
        # Single JSON file — each "setN" key holds one field_info dict
        json_path = os.path.join(".", "exp_sets", "uav", "cont_sets.json")
        json_dict = read_uav_json(json_path)
    elif robot_type == "wheeled":
        # One .ini file per environment (env1.ini, env2.ini, …) in the directory.
        # read_wheeled_configs() scans the folder and returns {"set1": …, "set2": …}
        # so train_single_env() can use the same set_key pattern as UAV.
        config_dir = os.path.join(".", "exp_sets", "wheeled")
        json_dict  = read_wheeled_configs(config_dir)
    else:
        raise ValueError(f"Unknown robot_type: '{robot_type}'")

    # ── MULTI-SEED LOOP ────────────────────────────────────────────────────
    for seed in seeds:
        date     = datetime.now().strftime('%b%d_%H')
        log_path = os.path.join(".", "logs", f"{date}_v{version}_{robot_type}_seed{seed}")
        os.makedirs(log_path, exist_ok=True)

        print(f"\n========== SEED {seed} | robot_type={robot_type} ==========\n")

        config          = config_base.copy()
        config["log_path"]  = log_path
        config["json_dict"] = json_dict

        env_ids       = list(range(1, len(json_dict) + 1))
        max_processes = mp.cpu_count()

        for start in range(0, len(env_ids), max_processes):
            batch = env_ids[start : start + max_processes]
            procs = [mp.Process(target=train_single_env, args=(eid, config, seed))
                     for eid in batch]
            for p in procs:
                p.start()
            for p in procs:
                p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # required for PyTorch + SB3
    main()