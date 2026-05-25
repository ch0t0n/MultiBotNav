# This is the new environment for multi-robot navigation developed in May 11, 2026

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import copy
import json
from datetime import datetime
import numpy as np
import gymnasium as gym
import pygame
import torch
import random
import multiprocessing as mp
from shapely import Polygon, Point      # shapely — used in MultiWheeled for collision geometry
from stable_baselines3.common.callbacks import LogEveryNTimesteps, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from sb3_contrib import CrossQ

# ================================
# Utility / Helper Functions
# ================================
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

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

def scale_polygon_about_centroid(poly_pts, scale):
    """Scale polygon vertices about centroid by `scale` (0<scale<=1 shrinks)."""
    pts = np.asarray(poly_pts, dtype=np.float64)
    centroid = np.mean(pts, axis=0)
    scaled = centroid + scale * (pts - centroid)
    return [tuple(p) for p in scaled.tolist()]

def read_uav_json(json_path, sf=1):
    """Load UAV field-info dicts from JSON, applying a scale factor sf."""
    with open(json_path, "r") as f:
        data = json.load(f)
    for set_name, cfg in data.items():
        cfg["field"]             = [tuple((p[0] * sf, p[1] * sf)) for p in cfg["field"]]
        cfg["init_positions"]    = [np.array(p, dtype=float) * sf for p in cfg["init_positions"]]
        cfg["infected_locations"]= [tuple((p[0] * sf, p[1] * sf)) for p in cfg["infected_locations"]]
    return data

def read_wheeled_json(json_path):
    """
    Load all wheeled-robot environment configs from a single JSON file.

    The JSON file maps environment keys ("env1", "env2", …) to config dicts
    with the following structure::

        {
          "screen":    {"width": 500, "height": 500},
          "robots":    {"length": 20, "width": 15, "max_speed": 100,
                        "max_steer": 45, "num_robots": 5,
                        "configs": [[x, y, theta_deg], ...]},
          "goals":     {"num_goals": 4, "goal_size": 10,
                        "positions": [[x, y], ...]},
          "obstacles": [[[x, y], ...], ...]
        }

    Returns a dict {"set1": env_params_1, "set2": env_params_2, …} whose
    values are flat dicts with the exact keys expected by MultiWheeled
    (SCREEN_WIDTH, ROBOT_INIT_CONFIGS with radians, GOAL_POSITIONS, etc.).
    """
    # Shrink obstacles in env2-env9 so robots can traverse cluttered maps more easily.
    obstacle_scale_env2_9 = 0.30

    with open(json_path, "r") as f:
        raw = json.load(f)

    configs = {}
    for i, (key, cfg) in enumerate(sorted(raw.items(),
                                          key=lambda kv: int(kv[0].replace("env", ""))),
                                   start=1):
        r   = cfg["robots"]
        g   = cfg["goals"]
        obs = cfg["obstacles"]
        if key.startswith("env"):
            env_idx = int(key.replace("env", ""))
            if 2 <= env_idx <= 9:
                obs = [scale_polygon_about_centroid(poly, obstacle_scale_env2_9) for poly in obs]
        env_params = {
            "SCREEN_WIDTH":      float(cfg["screen"]["width"]),
            "SCREEN_HEIGHT":     float(cfg["screen"]["height"]),
            "ROBOT_LENGTH":      float(r["length"]),
            "ROBOT_WIDTH":       float(r["width"]),
            "MAX_SPEED":         float(r["max_speed"]),
            "MAX_STEER":         float(r["max_steer"]),
            "NUM_ROBOTS":        int(r["num_robots"]),
            # theta stored in degrees in JSON; MultiWheeled expects radians
            "ROBOT_INIT_CONFIGS": [
                (float(c[0]), float(c[1]), float(np.radians(c[2])))
                for c in r["configs"]
            ],
            "NUM_GOALS":         int(g["num_goals"]),
            "GOAL_SIZE":         float(g["goal_size"]),
            "GOAL_POSITIONS":    [tuple(p) for p in g["positions"]],
            "NUM_OBSTACLES":     len(obs),
            "OBSTACLES":         [[tuple(v) for v in poly] for poly in obs],
        }
        configs[f"set{i}"] = env_params
        print(f"  Loaded wheeled config set{i} ← {key}  "
              f"({env_params['NUM_ROBOTS']} robots, "
              f"{int(env_params['SCREEN_WIDTH'])}×{int(env_params['SCREEN_HEIGHT'])})")

    return configs

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
        target_screen_size=800,
        # ── experiment control ──────────────────────────────────────
        reward_ablation="full",
        obs_mode="full",
        uncertainty_mode="full",
        dr_mode="none"
    ):
        super().__init__()

        # ── Validate experiment parameters ──────────────────────────
        assert uncertainty_mode in ("full", "wind_only", "act_only", "deterministic"), \
            f"Unknown uncertainty_mode: {uncertainty_mode}"
        assert dr_mode in ("none", "wind", "full"), \
            f"Unknown dr_mode: {dr_mode}"
        assert reward_ablation in ("full", "no_term", "no_path"), \
            f"Unknown reward_ablation: {reward_ablation}"
        assert obs_mode in ("full", "no_pos", "no_vis_hist", "pos_only"), \
            f"Unknown obs_mode: {obs_mode}"
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        
        # Store experiment flags
        self.reward_ablation  = reward_ablation
        self.obs_mode         = obs_mode
        self.uncertainty_mode = uncertainty_mode
        self.dr_mode          = dr_mode
        self.max_steps        = max_steps

        # ── Field info ──────────────────────────────────────────────
        self.field_info    = copy.deepcopy(field_info)          # Deep copy to prevent mutating original field data
        self.poly_vertices = self.field_info['field']           # Extract polygon vertices defining the boundary
        xs, ys             = zip(*self.poly_vertices)           # Unzip coordinates into separate X and Y tuples
        self.min_x, self.max_x = float(np.min(xs)), float(np.max(xs)) # Find min/max X for bounding box
        self.min_y, self.max_y = float(np.min(ys)), float(np.max(ys)) # Find min/max Y for bounding box
        self.world_width   = self.max_x - self.min_x            # Calculate total width of the environment
        self.world_height  = self.max_y - self.min_y            # Calculate total height of the environment

        # ── Rendering setup ─────────────────────────────────────────
        # Auto-scale render window to fit the target screen size
        scale_x = target_screen_size / self.world_width
        scale_y = target_screen_size / self.world_height
        self.render_scale  = min(scale_x, scale_y) * 0.90       # Scale down slightly (90%) to leave margins
        self.screen_width  = int(self.world_width  * self.render_scale) + 40  # Screen width with padding
        self.screen_height = int(self.world_height * self.render_scale) + 40  # Screen height with padding
        self.offset_x      = (self.screen_width  - self.world_width  * self.render_scale) / 2 # Center X offset
        self.offset_y      = (self.screen_height - self.world_height * self.render_scale) / 2 # Center Y offset

        # ── Robot params ────────────────────────────────────────────
        self.num_robots   = num_robots                          # Total number of UAVs
        self.init_robot_positions = np.array(
            self.field_info['init_positions'][:num_robots], dtype=np.float32) # Fetch initial positions
        self.robot_size   = 1.0                                 # Base robot visual/collision size
        self.mass         = 1.0                                 # UAV mass (overridden by DR full)
        self.thrust_power = 0.5                                 # Action scaling multiplier (overridden by DR full)
        self.max_speed    =  5.0                                # Maximum allowable speed
        self.min_speed    = -5.0                                # Minimum allowable speed
        self.robot_colors = [                                   # Color palette for differentiating robots
            (255, 0,   0), (0, 200,   0), (0,   0, 255),
            (255, 128, 0), (128, 0, 255), (255, 0, 255), (128, 128, 128),
        ]

        # ── Infection / Target params ───────────────────────────────
        self.initial_inf_locations  = [tuple(loc) for loc in self.field_info['infected_locations']] # Target coordinates
        self._nominal_infected_size = 1.5                       # Base radius for successful visitation
        self.infected_size          = self._nominal_infected_size
        self.infected_length        = len(self.initial_inf_locations) # Total number of targets

        # ── Base wind (mean) magnitude and direction ────────────────
        if wind_par is None: # Load wind parameters if not provided
            wind_par = [0, 0]
        self.base_wind_mag, self.base_wind_dir = float(wind_par[0]), float(wind_par[1])

        # ── Noise stds — set by uncertainty_mode ────────────────────
        # These are the *nominal* values; DR "full" may override at the start of each episode.
        _noise = {
            "full":          dict(wind=0.20, wind_dir=5.0, action=0.10, obs=0.01),
            "wind_only":     dict(wind=0.20, wind_dir=5.0, action=0.00, obs=0.00),
            "act_only":      dict(wind=0.00, wind_dir=0.0, action=0.10, obs=0.00),
            "deterministic": dict(wind=0.00, wind_dir=0.0, action=0.00, obs=0.00),
        }[uncertainty_mode]

        self.wind_noise_std      = _noise["wind"]               # Wind magnitude volatility
        self.wind_dir_noise_std  = _noise["wind_dir"]           # Wind direction volatility
        self.action_noise_std    = _noise["action"]             # Volatility applied to UAV control inputs
        self.obs_noise_std       = _noise["obs"]                # Sensor noise added to state observations
        self.init_position_noise = 0.05                          # Jitter added to spawn coordinates

        # ── Nominal values for DR restore ───────────────────────────
        self._nominal_action_noise_std = _noise["action"]
        self._nominal_mass             = 1.0
        self._nominal_thrust_power     = 0.5

        # ── Action space: (ax, ay) per robot ────────────────────────
        # Each robot's action has the following:
        #   1. a_x = Force component (or thrust) along x-axis
        #   2. a_y = Force component (or thrust) along y-axis
        # Therefore, total actions = 2 * num_robots
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(num_robots, 2), dtype=np.float32)

        # ── Observation space — depends on obs_mode ─────────────────
        # "full"        positions(2N) + velocities(2N) + visited_decimal(1) = 4N+1
        # "no_pos"      visited_decimal(1)  [remove all kinematics]
        # "no_vis_hist" positions(2N) + velocities(2N) = 4N
        # "pos_only"    positions(2N)
        N = num_robots
        _obs_dims = {"full": 4*N+1, "no_pos": 1, "no_vis_hist": 4*N, "pos_only": 2*N}
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(_obs_dims[obs_mode],), dtype=np.float32)

        self.render_mode = render_mode
        self.screen      = None
        self.clock       = None
        self.reset() # Initialise state

    # ── coordinate conversion ────────────────────────────────────────
    def world_to_screen(self, pos):
        """Converts physical world coordinates to PyGame screen coordinates."""
        x = (pos[0] - self.min_x) * self.render_scale + self.offset_x
        y = (pos[1] - self.min_y) * self.render_scale + self.offset_y
        return int(x), int(y)

    # ── observation builder ──────────────────────────────────────────
    def _get_obs(self):
        """Constructs the state array based on the selected observation mode."""
        # Convert the binary list of visited locations into a single float for the neural network
        infected_decimal = float(binary_list_to_decimal(list(self.infected_dict.values())))
        
        if self.obs_mode == "full":
            state = np.concatenate([self.robot_positions.flatten(),
                                    self.robot_velocities.flatten(),
                                    np.array([infected_decimal], dtype=np.float32)])
        elif self.obs_mode == "no_pos":
            state = np.array([infected_decimal], dtype=np.float32)
        elif self.obs_mode == "no_vis_hist":
            state = np.concatenate([self.robot_positions.flatten(),
                                    self.robot_velocities.flatten()])
        elif self.obs_mode == "pos_only":
            state = self.robot_positions.flatten().copy()

        state = state.astype(np.float32)
        
        # Add observation noise if specified
        if self.obs_noise_std > 0:
            state += np.random.normal(0, self.obs_noise_std, size=state.shape).astype(np.float32)

        # Package cleanly formatted info dictionary
        info = {f'robot{i}': self.robot_positions[i].copy() for i in range(self.num_robots)}
        return state, info

    # ── reset ────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode."""
        super().reset(seed=seed)

        # ── Domain randomization — re-sample base params each episode ─
        if self.dr_mode == "none":
            # Standard: add small episode-level noise to the nominal wind
            self.wind_mag         = self.base_wind_mag + np.random.normal(0, self.wind_noise_std)
            self.wind_dir         = self.base_wind_dir + np.random.normal(0, self.wind_dir_noise_std)
            # Restore nominal physical params (may have been overridden last episode)
            self.action_noise_std = self._nominal_action_noise_std
            self.infected_size    = self._nominal_infected_size
            self.mass             = self._nominal_mass
            self.thrust_power     = self._nominal_thrust_power
            
        elif self.dr_mode == "wind":
            # Randomise wind speed and direction uniformly
            self.wind_mag         = float(np.random.uniform(0.0, 1.0))
            self.wind_dir         = float(np.degrees(np.random.uniform(0.0, 2 * np.pi)))
            self.action_noise_std = self._nominal_action_noise_std
            self.infected_size    = self._nominal_infected_size
            self.mass             = self._nominal_mass
            self.thrust_power     = self._nominal_thrust_power
            
        elif self.dr_mode == "full":
            # Randomise all physical parameters significantly to improve policy robustness
            self.wind_mag         = float(np.random.uniform(0.0, 1.0))
            self.wind_dir         = float(np.degrees(np.random.uniform(0.0, 2 * np.pi)))
            self.action_noise_std = float(np.random.uniform(0.01, 0.10))
            r0                    = self._nominal_infected_size
            self.infected_size    = float(np.random.uniform(0.8 * r0, 1.2 * r0))
            self.mass             = float(np.random.uniform(0.90, 1.10))
            self.thrust_power     = 0.5 * float(np.random.uniform(0.80, 1.20))

        # ── Randomise starting positions ─────────────────────────────
        self.robot_positions = (
            self.init_robot_positions
            + np.random.normal(0, self.init_position_noise, self.init_robot_positions.shape)
        ).astype(np.float32)

        # ── Initialise dynamic state ─────────────────────────────────
        self.step_count         = 0
        self.visited            = set()                                 # Cells currently visited to track exploration
        self.infected_locations = list(copy.deepcopy(self.initial_inf_locations)) # Remaining targets
        self.infected_dict      = {loc: 0 for loc in self.initial_inf_locations}  # 0=unvisited, 1=visited
        self.robot_velocities   = np.zeros((self.num_robots, 2), dtype=np.float32)
        
        # ── Episode counters ─────────────────────────────────────────
        self.trajectories       = [[] for _ in range(self.num_robots)]  # Breadcrumbs for rendering
        self.total_path_length  = 0.0                                   # Accumulator for distance travelled
        self.prev_positions     = self.robot_positions.copy()           # Memory of previous step for path diff

        return self._get_obs()

    # ── step ─────────────────────────────────────────────────────────
    def step(self, actions):
        """Advances the simulation by one timestep."""
        self.step_count += 1
        terminated, truncated = False, False
        rewards = 0.0

        # ── Stochastic wind for this step ────────────────────────────
        # Wind changes dynamically every frame based on noise standard deviation
        wind_mag = self.wind_mag + np.random.normal(0, self.wind_noise_std)
        wind_dir = self.wind_dir + np.random.normal(0, self.wind_dir_noise_std)
        theta_w  = np.radians(wind_dir)
        wind     = np.array([wind_mag * np.cos(theta_w), wind_mag * np.sin(theta_w)],
                            dtype=np.float32)

        for i in range(self.num_robots):                        # For each robot
            ax_raw, ay_raw = actions[i]                         # Get raw neural net outputs

            # Action noise + scaling
            ax = ax_raw * self.thrust_power + np.random.normal(0, self.action_noise_std)
            ay = ay_raw * self.thrust_power + np.random.normal(0, self.action_noise_std)

            # Velocity update (Newtonian dynamics F=ma -> a=F/m)
            self.robot_velocities[i] += np.array([ax, ay]) / self.mass + wind
            self.robot_velocities[i]  = np.clip(self.robot_velocities[i], self.min_speed, self.max_speed)

            # Position update
            new_pos = self.robot_positions[i] + self.robot_velocities[i] # Predict next position
            if is_inside_polygon(new_pos, self.poly_vertices):           # Check if new position is inside the field
                self.robot_positions[i] = new_pos                        # Move to the new location
            else:
                rewards -= 50                                            # Boundary penalty — unchanged vs reference env
                self.robot_velocities[i][:] = 0                          # Set robot velocities to zero (crash into wall)

            # ── Per-robot movement penalties ─────────────────
            if self.reward_ablation != "no_path":
                rewards -= 0.1 * (ax ** 2 + ay ** 2)                     # Energy penalty (encourages efficient control)
                rewards -= 0.3 * float(np.linalg.norm(self.robot_velocities[i]))  # Speed penalty (Penalize high speeds)

            # ── Coverage shaping ─────────────────────────────────────────────────────
            # Encourage exploring new areas by mapping continuous space to a grid (round to 1 decimal)
            pos_key = tuple(np.round(self.robot_positions[i], 1))
            if pos_key in self.visited:
                rewards -= 30        # Revisit deterrent (prevents loitering in one spot)
            else:
                rewards += 3         # Exploration bonus (encourages sweeping the field)
            self.visited.add(pos_key)

            # ── Target coverage ──────────────────────────────────────────────────
            # Check if the robot successfully flew over an active target
            for inf_loc in list(self.infected_locations):
                if np.linalg.norm(self.robot_positions[i] - np.array(inf_loc)) <= self.infected_size:
                    self.infected_locations.remove(inf_loc)              # Mark target as cleared
                    self.infected_dict[inf_loc] = 1                      # Update dictionary for network observation
                    rewards += 1000                                      # Big reward for successfully visiting a target

            # Trajectory buffer (rendering only)
            self.trajectories[i].append(self.robot_positions[i].copy())
            if len(self.trajectories[i]) > 200:                          # Limit memory of breadcrumbs to 200 ticks
                self.trajectories[i].pop(0)

        # ── Path length computation ──────────────────────────────────
        step_dist = np.linalg.norm(self.robot_positions - self.prev_positions, axis=1) # True travel distance
        step_path = float(np.sum(step_dist))
        self.total_path_length += step_path
        self.prev_positions     = self.robot_positions.copy()

        # ── Global time and path penalties ───────────────────────────
        if self.reward_ablation != "no_path":
            rewards -= 1.0 * step_path     # Path-length penalty (encourages shortest route)
            rewards -= 2.0                 # Time penalty (encourages fast completion)

        # Distance shaping to nearest unvisited target (always active) — unchanged
        # Provides dense gradient pointing robots towards remaining targets
        if self.infected_locations:
            target_arr = np.array(self.infected_locations, dtype=np.float32)
            dists_mat  = np.linalg.norm(
                self.robot_positions[:, None] - target_arr[None, :], axis=2)
            rewards += 0.5 * float(np.sum(np.exp(-np.min(dists_mat, axis=1))))

        # ── Terminal signals ──────────────────────────────────────────────────────
        term_cond = ""
        
        # Win condition: All targets visited
        if len(self.infected_locations) == 0:
            if self.reward_ablation != "no_term":
                rewards += 5000      # Matches reference env success bonus
            term_cond  = "visited_all"
            terminated = True

        # Lose condition: Robots crashed into each other
        if self.num_robots > 1 and compute_min_dist(self.robot_positions) < self.robot_size:
            if self.reward_ablation != "no_term":
                rewards -= 5000     # Crash penalty
            term_cond  = "collision"
            terminated = True

        # Truncation: Hit max time steps
        truncated = self.step_count >= self.max_steps
        if truncated:
            term_cond = "max_steps"

        # Generate observations and metadata
        obs, info = self._get_obs()
        info.update({
            "step_count":        self.step_count,
            "remaining_targets": len(self.infected_locations),
            "path_length":       self.total_path_length,
            "term_cond":         term_cond,
        })
        return obs, rewards, terminated, truncated, info

    def render(self):
        """Displays the environment graphically using PyGame."""
        if self.render_mode != "human":
            return
            
        # Initialize PyGame window safely on first call
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Multi-UAV Path Planning")
            self.clock = pygame.time.Clock()

        # Clear background to white
        self.screen.fill((255, 255, 255))
        
        # Draw the boundary polygon
        scaled_poly = [self.world_to_screen(p) for p in self.poly_vertices]
        pygame.draw.polygon(self.screen, (255, 255, 0), scaled_poly)

        # Draw trajectory tails behind each robot
        for i in range(self.num_robots):
            if len(self.trajectories[i]) > 1:
                pygame.draw.lines(self.screen, (150, 150, 150), False,
                                  [self.world_to_screen(p) for p in self.trajectories[i]], 2)

        # ── Rendering Scaling Fix ────────────────────────────────────────────────
        # Using a much larger baseline multiplier to match Code 1's visibility intent.
        # Ensure robots and targets aren't shrunken by dynamic screen scaling.
        r_px = int(self.robot_size * self.render_scale)
        
        # Draw Robots
        for i in range(self.num_robots):
            pygame.draw.circle(self.screen, self.robot_colors[i % len(self.robot_colors)],
                               self.world_to_screen(self.robot_positions[i]), r_px)

        inf_px = int(self.infected_size * self.render_scale) / 1.5
        
        # Draw unvisited targets (Cyan)
        for loc in self.infected_locations:
            pygame.draw.circle(self.screen, (0, 220, 220), self.world_to_screen(loc), inf_px)
            
        # Draw visited targets (Dark Gray)
        for loc, vis in self.infected_dict.items():
            if vis:
                pygame.draw.circle(self.screen, (80, 80, 80), self.world_to_screen(loc), inf_px)

        # Swap display buffers and tick framerate
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """Cleanly shuts down the PyGame window."""
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
        max_steps=1000,
        target_screen_size=800,
        # ── experiment control ──────────────────────────────────────
        uncertainty_mode="full",
        dr_mode="none",
        reward_ablation="full",
        obs_mode="full"        
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
        assert obs_mode in ("full", "no_pos", "no_vis_hist", "pos_only"), \
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
        # "no_vis_hist" (x,y,θ,v,δ)(5N)
        # "pos_only"    (x,y)(2N)
        N = self.NUM_ROBOTS
        _obs_dims = {"full": 5*N+1, "no_pos": 3*N+1, "no_vis_hist": 5*N, "pos_only": 2*N}
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(_obs_dims[obs_mode],), dtype=np.float64)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.NUM_ROBOTS, 2), dtype=np.float32)

        # Reward scales tuned for dense learning in cluttered maps.
        # Keep magnitudes moderate so value targets stay stable.
        self.r_time         = 1.0
        self.r_goal         = 300.0
        self.r_success      = 2000.0
        self.r_collision    = 1200.0
        self.r_progress     = 8.0
        self.r_action_coeff = 0.02
        self.r_speed_coeff  = 0.01
        self.r_path_coeff   = 0.02
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
        elif self.obs_mode == "no_vis_hist":
            obs = self.robots.flatten().copy()
        elif self.obs_mode == "pos_only":
            obs = self.robots[:, :2].flatten().copy()

        obs = obs.astype(np.float64)
        if self.obs_noise_std > 0:
            obs += np.random.normal(0, self.obs_noise_std, size=obs.shape)

        info = {f'robot{i}': self.robots[i, :2].copy() for i in range(self.NUM_ROBOTS)}
        return obs, info

    def _nearest_unvisited_goal_dists(self, positions):
        """Return nearest unvisited-goal distance per robot."""
        unvisited = [gp for gp, vis in zip(self.goal_positions, self.goal_visited) if not vis]
        if not unvisited:
            return np.zeros(self.NUM_ROBOTS, dtype=np.float64)
        target_arr = np.array(unvisited, dtype=np.float64)
        dists_mat  = np.linalg.norm(positions[:, None] - target_arr[None, :], axis=2)
        return np.min(dists_mat, axis=1)

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
        self.prev_nearest_goal_dists = self._nearest_unvisited_goal_dists(self.prev_positions)

        if self.render_mode == "human":
            self._render_pygame()
        return self._get_obs()

    def step(self, action):
        terminated, truncated = False, False
        term_cond = ""
        # Recompute dec_g fresh at the start of each step
        self.dec_g = binary_list_to_decimal(self.goal_visited)
        # Constant time pressure (do not depend on binary goal encoding).
        reward = -self.r_time
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
                reward -= self.r_action_coeff * (a_noisy ** 2 + d_delta_noisy ** 2)
                reward -= self.r_speed_coeff * abs(v)

            robot_poly = get_robot_polygon(x, y, theta, self.ROBOT_LENGTH, self.ROBOT_WIDTH)

            for obs_pts in self.obstacles:
                obs_poly = Polygon(obs_pts)
                if robot_poly.intersects(obs_poly):
                    pt = robot_poly.intersection(obs_poly).representative_point()
                    self.collision_point = (pt.x, pt.y)
                    if self.reward_ablation != "no_term":
                        reward = -self.r_collision
                    terminated = True                    
                    self.collision_occurred = True
                    term_cond = "collision"
                    obs, info = self._get_obs()
                    info.update({
                        "step_count":    self.t,
                        "goals_visited": int(sum(self.goal_visited)),
                        "path_length":   self.total_path_length,
                        "term_cond":     term_cond,
                    })
                    return obs, reward, terminated, truncated, info

            for prev_poly in robot_polygons:
                if robot_poly.intersects(prev_poly):
                    pt = robot_poly.intersection(prev_poly).representative_point()
                    self.collision_point = (pt.x, pt.y)
                    if self.reward_ablation != "no_term":
                        reward = -self.r_collision
                    terminated = True
                    self.collision_occurred = True
                    term_cond = "collision"
                    obs, info = self._get_obs()
                    info.update({
                        "step_count":    self.t,
                        "goals_visited": int(sum(self.goal_visited)),
                        "path_length":   self.total_path_length,
                        "term_cond":     term_cond,
                    })
                    return obs, reward, terminated, truncated, info
            robot_polygons.append(robot_poly)

            for j, (gx, gy) in enumerate(self.goal_positions):
                if not self.goal_visited[j] and Point(gx, gy).buffer(self.goal_radius).intersects(robot_poly):
                    reward += self.r_goal
                    self.goal_visited[j] = True

            self.robots[i] = [x, y, theta, v, delta]
            self.robot_paths[i].append((x, y))

        # Path length
        current_positions   = self.robots[:, :2]
        step_path           = float(np.sum(np.linalg.norm(current_positions - self.prev_positions, axis=1)))
        self.total_path_length += step_path
        self.prev_positions     = current_positions.copy()

        if self.reward_ablation != "no_path":
            reward -= self.r_path_coeff * step_path

        # Potential-based shaping: reward progress towards nearest unvisited goal.
        curr_nearest_dists = self._nearest_unvisited_goal_dists(current_positions)
        progress = np.clip(self.prev_nearest_goal_dists - curr_nearest_dists, -5.0, 5.0)
        reward  += self.r_progress * float(np.sum(progress))
        self.prev_nearest_goal_dists = curr_nearest_dists

        if sum(self.goal_visited) >= len(self.goal_positions):
            if self.reward_ablation != "no_term":
                reward += self.r_success
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
    stage_specs = []
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
        stage_specs = [("main", env_kwargs, time_steps)]
    elif robot_type == "wheeled":
        gym_id   = "MultiWheeled-v0"
        # Curriculum inside one run:
        # 1) deterministic/no-DR for discovery
        # 2) wind uncertainty + DR for robustness
        stage1_steps = int(0.70 * time_steps)
        stage2_steps = max(time_steps - stage1_steps, 0)
        env_kwargs_stage1 = {
            "env_params": json_dict[set_key],
            "max_steps":  max_steps,
            "render_mode": None,
            "uncertainty_mode": "deterministic",
            "dr_mode": "none",
        }
        stage_specs.append(("stage1_deterministic", env_kwargs_stage1, stage1_steps))
        if stage2_steps > 0:
            env_kwargs_stage2 = {
                "env_params": json_dict[set_key],
                "max_steps":  max_steps,
                "render_mode": None,
                "uncertainty_mode": "wind_only",
                "dr_mode": "wind",
            }
            stage_specs.append(("stage2_robust_wind", env_kwargs_stage2, stage2_steps))
    else:
        raise ValueError(f"Unknown robot_type: '{robot_type}'. Choose 'uav' or 'wheeled'.")

    initial_stage_name, initial_kwargs, initial_steps = stage_specs[0]
    vec_env = make_vec_env(gym_id, env_kwargs=initial_kwargs, n_envs=num_envs, seed=seed, monitor_dir=None)

    logger = configure(
        os.path.join(log_path, f"crossq_env{env_id}"),
        ["stdout", "log", "csv", "tensorboard"],
    )

    model = CrossQ("MlpPolicy", vec_env, verbose=1, device=device_id, seed=seed)
    model.set_logger(logger)
    if initial_steps > 0:
        eval_env = make_vec_env(gym_id, env_kwargs=initial_kwargs, n_envs=1, seed=seed, monitor_dir=None)
        callback = CallbackList([
            LogEveryNTimesteps(n_steps=10000),
            EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(log_path, f"best_model_env{env_id}_{initial_stage_name}"),
                log_path            =os.path.join(log_path, f"eval_logs_env{env_id}_{initial_stage_name}"),
                eval_freq           =max(10000 // num_envs, 1),
                n_eval_episodes     =10,
                deterministic       =True,
                render              =False,
            ),
        ])
        model.learn(total_timesteps=initial_steps, callback=callback)
        eval_env.close()

    # Continue training in later stages without resetting model timestep counter.
    for stage_name, env_kwargs, stage_steps in stage_specs[1:]:
        if stage_steps <= 0:
            continue
        vec_env.close()
        vec_env = make_vec_env(gym_id, env_kwargs=env_kwargs, n_envs=num_envs, seed=seed, monitor_dir=None)
        model.set_env(vec_env)
        eval_env = make_vec_env(gym_id, env_kwargs=env_kwargs, n_envs=1, seed=seed, monitor_dir=None)
        callback = CallbackList([
            LogEveryNTimesteps(n_steps=10000),
            EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(log_path, f"best_model_env{env_id}_{stage_name}"),
                log_path            =os.path.join(log_path, f"eval_logs_env{env_id}_{stage_name}"),
                eval_freq           =max(10000 // num_envs, 1),
                n_eval_episodes     =10,
                deterministic       =True,
                render              =False,
            ),
        ])
        model.learn(total_timesteps=stage_steps, callback=callback, reset_num_timesteps=False)
        eval_env.close()

    model.save(os.path.join(log_path, f"env{env_id}_CrossQ_{robot_type}"))

    vec_env.close()
    print(f"[DONE]  Training env {env_id}, seed {seed}, robot_type {robot_type}")


# ================================
# Main Launcher
# ================================
def main():
    # ── CONFIG ─────────────────────────────────────────────────────────────
    version    = "1"
    robot_type = "wheeled"       # [NEW] change to "wheeled" to train MultiWheeled
    # seeds      = [0, 42, 123, 2024, 9999]
    seeds      = [42]

    config_base = {
        "robot_type": robot_type,  # [NEW] forwarded to every worker
        "device_ids": [f"cuda:{i}" for i in range(8)],
        "time_steps": int(3e6),
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
        # Single JSON file consolidating all 10 environments (500×500, 5 robots each).
        # read_wheeled_json() returns {"set1": env_params_1, …} so train_single_env()
        # uses the identical set_key pattern as UAV.
        json_path = os.path.join(".", "exp_sets", "wheeled", "wheeled_configs.json")
        json_dict = read_wheeled_json(json_path)
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