# ================================
# This is the version 1.0 (multi-seed variation) of the final spraying env confirmed in April 20, 2026
# Author: Jahid Chowdhury Choton (email: choton@ksu.edu)
# ================================

import os  # OS operations (paths, folders)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import json  # JSON file handling
from datetime import datetime  # timestamping logs
import numpy as np  # numerical operations
import gymnasium as gym  # RL environment framework
import pygame  # rendering

import torch
import random
import multiprocessing as mp
from stable_baselines3.common.callbacks import LogEveryNTimesteps, EvalCallback, CallbackList  # logging callback
from stable_baselines3.common.env_util import make_vec_env  # vectorized env
from stable_baselines3.common.logger import configure  # logger
from sb3_contrib import CrossQ  # RL algorithm

# ================================
# Utility Functions (Optimized)
# ================================

def is_inside_polygon(point, poly):  # check if a point is inside polygon
    x, y = point  # unpack point
    inside = False  # initialize flag
    n = len(poly)  # number of vertices
    p1x, p1y = poly[0]  # first vertex
    for i in range(n + 1):  # iterate through edges
        p2x, p2y = poly[i % n]  # next vertex (wrap around)
        # check if point is within vertical bounds of edge
        if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:  # avoid division by zero
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x  # intersection
            if p1x == p2x or x <= xinters:  # crossing check
                inside = not inside  # toggle
        p1x, p1y = p2x, p2y  # move to next edge
    return inside  # return result


def compute_min_dist(x):  # compute minimum pairwise distance
    x = np.asarray(x, dtype=np.float32)  # ensure array
    diff = x[:, None, :] - x[None, :, :]  # pairwise differences
    dist_matrix = np.linalg.norm(diff, axis=-1)  # compute distances
    np.fill_diagonal(dist_matrix, np.inf)  # ignore self-distance
    return float(np.min(dist_matrix))  # return min distance


def load_experiment_dict_json(json_path):  # load experiment configs
    with open(json_path, "r") as f:  # open file
        data = json.load(f)  # load JSON
    for _, cfg in data.items():  # iterate over configs
        cfg["field"] = [tuple(p) for p in cfg["field"]]  # convert to tuples
        cfg["init_positions"] = np.array(cfg["init_positions"], dtype=float)  # positions array
        cfg["infected_locations"] = [tuple(p) for p in cfg["infected_locations"]]  # infection
    return data  # return processed dict


# ================================
# Environment
# ================================

class MultiRobotEnv(gym.Env):
    metadata = {'render_modes': ['human', 'print', 'rgb_array'], "render_fps": 60}
    def __init__(
        self,
        field_info,
        render_mode=None,
        wind_par=[0, 0],
        num_robots=3,
        render_scale=5,
        max_steps=1000,
        # ── experiment control ──────────────────────────────────────
        reward_ablation="full",
        obs_mode="full",
        uncertainty_mode="full",
        dr_mode="none",
    ):
        super().__init__()

        # Validate experiment parameters
        assert reward_ablation in ("full", "no_term" , "no_spr" , "no_path"), \
            f"Unknown reward_ablation: {reward_ablation}"
        assert obs_mode in ("full" , "no_pos" , "no_inf_hist" , "pos_only"), \
            f"Unknown obs_mode: {obs_mode}"
        assert uncertainty_mode in ("full" , "wind_only" , "act_only" , "deterministic"), \
            f"Unknown uncertainty_mode: {uncertainty_mode}"
        assert dr_mode in ("none", "wind", "full"), \
            f"Unknown dr_mode: {dr_mode}"
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # Store experiment flags
        self.reward_ablation  = reward_ablation
        self.obs_mode         = obs_mode
        self.uncertainty_mode = uncertainty_mode
        self.dr_mode          = dr_mode

        # ── Field info ──────────────────────────────────────────────
        self.field_info = field_info
        self.poly_vertices = field_info['field']
        self.xs, self.ys = zip(*self.poly_vertices)
        self.min_x, self.max_x = np.min(self.xs), np.max(self.xs)
        self.min_y, self.max_y = np.min(self.ys), np.max(self.ys)
        self.world_width  = self.max_x - self.min_x
        self.world_height = self.max_y - self.min_y

        # ── Rendering ───────────────────────────────────────────────
        self.render_scale = render_scale
        self.screen_width, self.screen_height = 800, 800
        self.offset_x = (self.screen_width  - self.world_width  * self.render_scale) / 2
        self.offset_y = (self.screen_height - self.world_height * self.render_scale) / 2
        self.max_steps = max_steps        
        self.render_mode = render_mode
        self.screen = None
        self.clock  = None

        # ── Robot params ────────────────────────────────────────────
        self.num_robots = num_robots
        self.init_robot_positions = field_info['init_positions'][:num_robots]
        self.init_robot_capacities = np.array(
            field_info['robot_capacities'][:num_robots], dtype=np.float32)
        self.robot_size   = 2    # collision radius
        self.mass         = 1.0  # UAV mass (overridden by DR full)
        self.thrust_power = 0.5  # action scaling (overridden by DR full)
        self.max_speed    = 5
        self.min_speed    = -5
        self.robot_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Lime / Green
            (255, 0, 255),  # Magenta / Fuchsia
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Violet / Purple
            (255, 0, 255)   # Magenta / Fuchsia
        ]

        # ── Infection params ────────────────────────────────────────
        self._nominal_infected_radius = 5   # nominal spray radius (DR may override per-episode)
        self.infected_radius = self._nominal_infected_radius
        self.spray_sigma     = self.infected_radius / 2
        self.init_infected_positions = np.array(
            [[x, y] for x, y, _ in field_info['infected_locations']])
        self.init_infected_levels = np.array(
            [lvl for _, _, lvl in field_info['infected_locations']], dtype=np.float32)
        self.max_infection_level = np.max(self.init_infected_levels)

        # ── Base wind (mean) magnitude and direction
        self.base_wind_mag, self.base_wind_dir = wind_par

        # ── Noise stds — set by uncertainty_mode ────────────────────
        # These are the *nominal* values; DR "full" may override
        # action_noise_std at the start of each episode.
        _noise = {
            "full":          dict(wind=0.20, wind_dir=5.0, action=0.10, spray=0.05, obs=0.01),
            "wind_only":     dict(wind=0.20, wind_dir=5.0, action=0.00, spray=0.00, obs=0.00),
            "act_only":      dict(wind=0.00, wind_dir=0.0, action=0.10, spray=0.00, obs=0.00),
            "deterministic": dict(wind=0.00, wind_dir=0.0, action=0.00, spray=0.00, obs=0.00),
        }[uncertainty_mode]

        self.wind_noise_std         = _noise["wind"]
        self.wind_dir_noise_std     = _noise["wind_dir"]
        self.action_noise_std       = _noise["action"]
        self.spray_noise_std        = _noise["spray"]
        self.obs_noise_std          = _noise["obs"]
        self.init_position_noise    = 0.5

        # ── Nominal noise stds (DR "full" samples around these) ─────
        self._nominal_action_noise_std = _noise["action"]
        self._nominal_mass             = 1.0
        self._nominal_thrust_power     = 0.5

        # Each robot's action have the following:
        #   1. a_x = Force component (or thrust) along x-axis
        #   2. a_y = Force component (or thrust) along y-axis
        #   3. sigma = spray rate
        # Therefore, total actions = 3*num_robots

        # ── Action space: (ax, ay, spray) per robot ─────────────────
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(num_robots, 3), dtype=np.float32)

        # Each robot's observation have the following for base:
        #    1. (x, y) co-ordinates of each robot
        #    2. (v_x, v_y) velocities along x and y axis of each robot
        #    3. current spraying capacity (C_p(t))
        #    4. current infection levels of M locations
        # Therefore, total observations = 2*num_robots (positions) + 2*num_robots (velocities) + 
        # num_robots (spraying capacities) + M (num_inf_locations) = 5*num_robots + M

        # ── Observation space — depends on obs_mode ──────────────────
        M = len(self.init_infected_levels)
        N = num_robots
        _obs_dims = {
            "full":          5*N + M,       # original
            "no_pos":        N + M,         # capacities(N) + inf_levels(M)
            "no_inf_hist":   5*N,     # 5N 
            "pos_only":      2*N,           # positions only
        }
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(_obs_dims[obs_mode],), dtype=np.float32)

        self.reset()  # initialise state

    # ── coordinate conversion ────────────────────────────────────────
    def world_to_screen(self, pos):
        x = (pos[0] - self.min_x) * self.render_scale + self.offset_x
        y = (pos[1] - self.min_y) * self.render_scale + self.offset_y
        return int(x), int(y)

    # ── observation builder ──────────────────────────────────────────
    def _get_obs(self):
        info = {
            f'robot{i}': {
                'position': self.robot_positions[i],
                'capacity': self.robot_capacities[i],
            }
            for i in range(self.num_robots)
        }
        infection_norm = self.infected_levels / self.max_infection_level

        if self.obs_mode == "full":
            state = np.concatenate([self.robot_positions.flatten(),
            self.robot_velocities.flatten(),
            self.robot_capacities,
            infection_norm])
        elif self.obs_mode == "no_pos":
            state = np.concatenate([self.robot_capacities, infection_norm])
        elif self.obs_mode == "no_inf_hist":
            state = np.concatenate([self.robot_positions.flatten(),
            self.robot_velocities.flatten(),
            self.robot_capacities])
        elif self.obs_mode == "pos_only":
            state = self.robot_positions.flatten().copy()

        if self.obs_noise_std > 0:
            state = state + np.random.normal(0, self.obs_noise_std, size=state.shape)

        return state.astype(np.float32), info

    # ── reset ────────────────────────────────────────────────────────
    def reset(self, seed=None, options={}):
        super().reset(seed=seed)

        # ── Domain randomization — re-sample base params each episode ─
        if self.dr_mode == "none":
            # Standard: add small episode-level noise to the nominal wind
            self.wind_mag = self.base_wind_mag + np.random.normal(0, self.wind_noise_std)
            self.wind_dir = self.base_wind_dir + np.random.normal(0, self.wind_dir_noise_std)
            # Restore nominal physical params (may have been overridden last episode)
            self.action_noise_std = self._nominal_action_noise_std
            self.infected_radius  = self._nominal_infected_radius
            self.spray_sigma      = self.infected_radius / 2
            self.mass             = self._nominal_mass
            self.thrust_power     = self._nominal_thrust_power

        elif self.dr_mode == "wind":
            # Randomise wind speed and direction uniformly
            self.wind_mag = float(np.random.uniform(0.0, 1.0))
            self.wind_dir = float(np.degrees(np.random.uniform(0.0, 2 * np.pi)))
            self.action_noise_std = self._nominal_action_noise_std
            self.infected_radius  = self._nominal_infected_radius
            self.spray_sigma      = self.infected_radius / 2
            self.mass             = self._nominal_mass
            self.thrust_power     = self._nominal_thrust_power

        elif self.dr_mode == "full":
            # Randomise all physical parameters
            self.wind_mag         = float(np.random.uniform(0.0, 1.0))
            self.wind_dir         = float(np.degrees(np.random.uniform(0.0, 2 * np.pi)))
            self.action_noise_std = float(np.random.uniform(0.01, 0.10))
            r0 = self._nominal_infected_radius
            self.infected_radius  = float(np.random.uniform(0.8 * r0, 1.2 * r0))
            self.spray_sigma      = self.infected_radius / 2
            self.mass             = float(np.random.uniform(0.9, 1.1))
            self.thrust_power     = 0.5 * float(np.random.uniform(0.8, 1.2))

        # ── Randomise starting positions ─────────────────────────────
        self.robot_positions = self.init_robot_positions + np.random.normal(
            0, self.init_position_noise, self.init_robot_positions.shape)

        # ── Initialise dynamic state ─────────────────────────────────
        self.robot_velocities = np.zeros((self.num_robots, 2), dtype=np.float32)
        self.robot_capacities = self.init_robot_capacities.copy()
        self.infected_positions = self.init_infected_positions.copy()
        self.infected_levels    = self.init_infected_levels.copy()

        # ── Episode counters ─────────────────────────────────────────
        self.step_count       = 0
        self.total_path_length = 0.0
        self.prev_positions   = self.robot_positions.copy()
        self.trajectories     = [[] for _ in range(self.num_robots)] # Only for rendering

        return self._get_obs()

    # ── step ─────────────────────────────────────────────────────────
    def step(self, actions):
        self.step_count += 1
        reward = 0
        terminated, truncated = False, False

        # ── Stochastic wind for this step ────────────────────────────
        wind_mag = self.wind_mag + np.random.normal(0, self.wind_noise_std)
        wind_dir = self.wind_dir + np.random.normal(0, self.wind_dir_noise_std)
        theta = np.radians(wind_dir)
        wind  = np.array([wind_mag * np.cos(theta), wind_mag * np.sin(theta)])

        total_sprayed = 0                       # Track total infection reduced this step (for reward)
        for i in range(self.num_robots):        # For each robot
            ax, ay, spray = actions[i]          # Get actual actions: thrust_x, thrust_y, sigma (spray amount)

            # Action noise + scaling
            ax = ax * self.thrust_power + np.random.normal(0, self.action_noise_std)
            ay = ay * self.thrust_power + np.random.normal(0, self.action_noise_std)

            # Velocity update (Newtonian dynamics)
            self.robot_velocities[i] += np.array([ax, ay]) / self.mass + wind
            self.robot_velocities[i]  = np.clip(
                self.robot_velocities[i], self.min_speed, self.max_speed)

            # Position update
            new_pos = self.robot_positions[i] + self.robot_velocities[i] # Predict next position
            outside = not is_inside_polygon(new_pos, self.poly_vertices) # Check if new position is outside the field
            if outside:                                 # If new position is outside the field
                reward -= 50                            # Penalty for moving outside the field
                self.robot_velocities[i] = 0            # Set all robot velocities to zero
            else:
                self.robot_positions[i] = new_pos       # Else, move to the new location

            # Spray dynamics (vectorized + capacity constraint)
            if spray > 0 and self.robot_capacities[i] > 0:                              # Only spray if spray rate > 0 and remaining spraying capacities > 0
                dists = np.linalg.norm(
                    self.robot_positions[i] - self.infected_positions, axis=1)          # Distances from robot to infected locations
                mask  = (dists <= self.infected_radius) & (self.infected_levels > 0)    # Only affect nearby infected locations

                if np.any(mask):                                                                # If nearby infected locations exists
                    w = np.exp(-(dists[mask] ** 2) / (2 * self.spray_sigma ** 2))               # Gaussian decay: closer points get more spray effect
                    spray_effect = spray * w                                                    # Scale spray intensity by distance weighting
                    spray_effect += np.random.normal(0, self.spray_noise_std, size=w.shape)     # Add stochasticity to spraying process
                    spray_effect  = np.clip(spray_effect, 0, None)                              # Prevent negative spray
                    applied       = np.minimum(spray_effect, self.infected_levels[mask])        # Cannot disinfect more infection than exists
                    total         = np.sum(applied)                                             # Total spray to apply
                    if total > self.robot_capacities[i]:                        # If exceeding available spray capacity
                        applied *= self.robot_capacities[i] / (total + 1e-8)    # Scale down proportionally
                        total    = self.robot_capacities[i]                     # Total spray applied by this robot in this step
                    self.robot_capacities[i]    -= total                        # Reduce sprayed amount from capacity
                    self.infected_levels[mask]  -= applied                      # Reduce infection levels by applied spray amount
                    total_sprayed               += total                        # Accumulate global spraying amount for all robots
                else:
                    # Useless spray penalty (part of R_eff)
                    if self.reward_ablation != "no_spr":
                        reward -= 0.05 * spray

            # ── Per-robot movement penalties ─────────────────
            if self.reward_ablation != "no_path":
                reward -= 0.1 * (ax ** 2 + ay ** 2)                       # Energy penalty (encourages efficient control)
                reward -= 0.3 * np.linalg.norm(self.robot_velocities[i])  # Movement penalty (Penalize high speeds)

            # Trajectory buffer (rendering only)
            self.trajectories[i].append(self.robot_positions[i].copy())
            if len(self.trajectories[i]) > 200:
                self.trajectories[i].pop(0)

        # Path length computation (true traveled distance)
        step_dist = np.linalg.norm(self.robot_positions - self.prev_positions, axis=1)   # Distance traveled by each robot this step
        step_path = np.sum(step_dist)                                                    # Total distance traveled by all robots
        self.total_path_length += step_path                                              # Accumulate episode path length
        self.prev_positions = self.robot_positions.copy()                                # Update previous positions

        # ── R_spray: core spraying reward ────────────────────────────
        if self.reward_ablation != "no_spr":
            reward += 100.0 * total_sprayed     # Strong positive reward for removing infection

        # ── R_eff: global efficiency penalties ───────────────────────
        if self.reward_ablation != "no_path":
            reward -= 1.0 * step_path       # path length penalty
            reward -= 2.0                   # time penalty

        # Remaining infection penalty
        remaining = np.sum(self.infected_levels)
        if self.reward_ablation != "no_spr":
            # Distance shaping to nearest active infection
            active_mask = self.infected_levels > 0.01
            if np.any(active_mask):
                active_pos = self.infected_positions[active_mask]
                dists_mat  = np.linalg.norm(
                    self.robot_positions[:, None] - active_pos[None, :], axis=2)
                nearest    = np.min(dists_mat, axis=1)
                reward    += 0.5 * np.sum(np.exp(-nearest))            
            reward -= 0.5 * remaining

        if remaining <= 0.01:
            if self.reward_ablation != "no_term":
                reward += 500.0      # Large positive reward for successfull spraying
            terminated = True        # always terminate on succession (completion)

        # ── R_col: collision penalty ──────────────────────────────────
        if self.num_robots > 1:                                             # Check collision only for multi-robots
            if compute_min_dist(self.robot_positions) < self.robot_size:    # Robots too close
                if self.reward_ablation != "no_term":
                    reward -= 1000.0                                        # Large negative reward for collision
                terminated = True   # always terminate on collision (safety)

        # ── Build observation & info ──────────────────────────────────
        obs, info = self._get_obs()
        info.update({
            "total_sprayed": float(total_sprayed),          # Total infection removed this step
            "remaining_infection": float(remaining),        # Remaining infected amount
            "episode_length": self.step_count,              # Current timestep
            "path_length": float(self.total_path_length)    # Total distance traveled
        })
        truncated = self.step_count >= self.max_steps       # Episode ends if max steps reached

        return obs, reward, terminated, truncated, info

    # ── render ───────────────────────────────────────────────────────
    def render(self):
        if self.render_mode != "human": 
            return              # Only render if mode is set to "human" (skip rendering for training efficiency)

        if self.screen is None:
            pygame.init()                                                                   # Initialize all pygame modules
            pygame.display.init()                                                           # Initialize display module explicitly
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))  # Create window with predefined screen size
            pygame.display.set_caption("Multi-Robot Spraying Environment")                  # Window title
            self.clock = pygame.time.Clock()                                                # Clock to control rendering FPS

        self.screen.fill((255, 255, 255))       # Fill screen with white color

        # --- Draw field boundary ---
        scaled_poly = [self.world_to_screen(p) for p in self.poly_vertices] # Convert field vertices from world coordinates → screen coordinates
        pygame.draw.polygon(self.screen, (255, 255, 0), scaled_poly)

        # --- Draw robot trajectories (paths) ---
        for i in range(self.num_robots):
            if len(self.trajectories[i]) > 1:                                       # Only draw if at least two points exist (needed for a line)
                points = [self.world_to_screen(p) for p in self.trajectories[i]]    # Convert trajectory points to screen coordinates
                pygame.draw.lines(self.screen, (150, 150, 150), False, points, 2)   # Draw polyline showing path history

        # --- Draw robots (current positions) ---        
        for i in range(self.num_robots):
            pos = self.world_to_screen(self.robot_positions[i])                 # Convert robot position to screen coordinates
            pygame.draw.circle(self.screen,
                               self.robot_colors[i % len(self.robot_colors)],   # Assign color cyclically
                               pos, self.robot_size * self.render_scale / 2)    # Scale radius for visualization

        # --- Draw infected locations (dynamic color) ---
        for pos, level in zip(self.infected_positions, self.infected_levels):
            intensity = int(255 * min(level / 5.0, 1.0))                          # Map infection level → color intensity (visual feedback)
            pygame.draw.circle(self.screen, (0, 255-intensity, 255),              # Higher infection → darker blue and lower infection → lighter blue
                               self.world_to_screen(pos), self.render_scale + 2)  # Fixed radius for visibility
        
        # Update display
        pygame.display.flip()  # Push all drawings to the screen (double buffering)
        self.clock.tick(60)    # Limit rendering to 60 FPS for smooth visualization

    def close(self):
        if self.screen:
            pygame.quit()

if 'MultiRobotEnv-v0' not in gym.envs.registry:
    gym.register(id='MultiRobotEnv-v0', entry_point=MultiRobotEnv, max_episode_steps=1000)

# ================================
# SEED FUNCTION
# ================================
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================================
# TRAIN FUNCTION (ONE ENV PER PROCESS)
# ================================
def train_single_env(env_id, config, seed=42):
    torch.set_num_threads(1)
    set_seed(seed)    
    device_ids = config["device_ids"]
    proc_id = (env_id - 1)
    device_id = device_ids[(proc_id + seed * 10) % len(device_ids)]
    log_path = config["log_path"]
    json_dict = config["json_dict"]
    num_envs = config["num_envs"]
    num_robots = config["num_robots"]
    max_steps = config["max_steps"]
    time_steps = config["time_steps"]

    run_name = f"env{env_id}_seed{seed}"
    print(f"[START] Training {run_name} on {device_id}")

    vec_env = make_vec_env(
        'MultiRobotEnv-v0',
        env_kwargs={
            'field_info': json_dict[f"set{env_id}"],
            'num_robots': num_robots,
            'max_steps': max_steps,
            'render_mode': None
        },
        n_envs=num_envs,
        seed=seed
    )

    eval_env = make_vec_env(
        'MultiRobotEnv-v0',
        env_kwargs={
            'field_info': json_dict[f"set{env_id}"],
            'num_robots': num_robots,
            'max_steps': max_steps,
            'render_mode': None
        },
        n_envs=1,
        seed=seed
    )

    callback1 = LogEveryNTimesteps(n_steps=10000)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_path, f"best_model_env{env_id}"),
        log_path=os.path.join(log_path, f"eval_logs_env{env_id}"),
        eval_freq=max(10000 // num_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    callback = CallbackList([callback1, eval_callback])

    logger = configure(
        os.path.join(log_path, f"crossq_env{env_id}"),
        ["stdout", "log", "csv", "tensorboard"]
    )

    model = CrossQ(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device=device_id,
        seed=seed
    )

    model.set_logger(logger)
    model.learn(total_timesteps=time_steps, callback=callback)
    model.save(os.path.join(log_path, f"env{env_id}_CrossQ"))
    vec_env.close()
    eval_env.close()
    print(f"[DONE] Training env {env_id}, seed {seed}")


# ================================
# MULTIPROCESS LAUNCHER
# ================================
def main():

    # ---------------- CONFIG ----------------
    version = "1"
    seeds = [0, 42, 123, 2024, 9999]   # <-- CHANGE HERE

    config_base = {
        "device_ids": [f"cuda:{i}" for i in range(8)],
        "time_steps": int(2e6),
        "num_envs": 8,
        "num_robots": 3,
        "max_steps": 1000,
    }

    # ---------------- REGISTER ENV ----------------
    if 'MultiRobotEnv-v0' not in gym.envs.registry:
        gym.register(id='MultiRobotEnv-v0', entry_point=MultiRobotEnv, max_episode_steps=config_base['max_steps'])

    # ---------------- LOAD JSON ----------------
    json_path = os.path.join('.', 'exp_sets', 'stochastic_envs_v2.json')
    json_dict = load_experiment_dict_json(json_path)

    # ---------------- MULTI-SEED LOOP ----------------
    for seed in seeds:

        date = datetime.now().strftime('%b%d_%H')
        log_path = os.path.join(
            '.', 'logs',
            f"{date}_v{version}_seed{seed}"
        )
        os.makedirs(log_path, exist_ok=True)

        print(f"\n========== SEED {seed} ==========\n")

        # Merge config
        config = config_base.copy()
        config["log_path"] = log_path
        config["json_dict"] = json_dict

        env_ids = list(range(1, 11))
        max_processes = mp.cpu_count()

        for start in range(0, len(env_ids), max_processes):
            batch = env_ids[start:start + max_processes]

            processes = []
            for env_id in batch:
                p = mp.Process(
                    target=train_single_env,
                    args=(env_id, config, seed)
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # REQUIRED for PyTorch + SB3
    main()