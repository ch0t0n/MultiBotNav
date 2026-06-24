"""Headless MultiWheeled environment (bicycle kinematics, no pygame dependency)."""

from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
from shapely import Point, Polygon

from .geometry import binary_list_to_decimal, get_robot_polygon


class MultiWheeled(gym.Env):
    """Multi-wheeled-robot path-planning (bicycle-model kinematics)."""

    metadata = {"render_modes": [], "render_fps": 60}

    def __init__(
        self,
        env_params,
        wind_par=None,
        max_steps=1000,
        uncertainty_mode="full",
        dr_mode="none",
        reward_ablation="full",
        obs_mode="full",
    ):
        super().__init__()
        if wind_par is None:
            wind_par = [0, 0]

        assert uncertainty_mode in ("full", "wind_only", "act_only", "deterministic")
        assert dr_mode in ("none", "wind", "full")
        assert reward_ablation in ("full", "no_term", "no_path")
        assert obs_mode in ("full", "no_pos", "no_vis_hist", "pos_only")

        self.uncertainty_mode = uncertainty_mode
        self.dr_mode = dr_mode
        self.reward_ablation = reward_ablation
        self.obs_mode = obs_mode
        self.max_steps = max_steps
        self.dt = 0.05

        self.ROBOT_LENGTH = env_params["ROBOT_LENGTH"]
        self.ROBOT_WIDTH = env_params["ROBOT_WIDTH"]
        self.NUM_ROBOTS = env_params["NUM_ROBOTS"]
        self.init_ROBOTS = env_params["ROBOT_INIT_CONFIGS"]

        self._nominal_max_speed = float(env_params["MAX_SPEED"])
        self._nominal_max_steer = float(np.radians(env_params["MAX_STEER"]))
        self._nominal_accel_scale = 100.0
        self.MAX_SPEED = self._nominal_max_speed
        self.MAX_STEER = self._nominal_max_steer
        self.accel_scale = self._nominal_accel_scale

        self.goal_radius = env_params["GOAL_SIZE"]
        self.goal_positions = env_params["GOAL_POSITIONS"]
        self.obstacles = env_params["OBSTACLES"]

        self.world_width = float(env_params["SCREEN_WIDTH"])
        self.world_height = float(env_params["SCREEN_HEIGHT"])

        self.base_wind_mag = float(wind_par[0])
        self.base_wind_dir = float(wind_par[1])

        _noise = {
            "full": dict(wind=0.20, wind_dir=5.0, action=0.10, obs=0.01),
            "wind_only": dict(wind=0.20, wind_dir=5.0, action=0.00, obs=0.00),
            "act_only": dict(wind=0.00, wind_dir=0.0, action=0.10, obs=0.00),
            "deterministic": dict(wind=0.00, wind_dir=0.0, action=0.00, obs=0.00),
        }[uncertainty_mode]

        self.wind_noise_std = _noise["wind"]
        self.wind_dir_noise_std = _noise["wind_dir"]
        self.action_noise_std = _noise["action"]
        self.obs_noise_std = _noise["obs"]
        self.init_position_noise = 0.5
        self.init_heading_noise = float(np.radians(2))
        self._nominal_action_noise_std = _noise["action"]

        n = self.NUM_ROBOTS
        _obs_dims = {
            "full": 5 * n + 1,
            "no_pos": 3 * n + 1,
            "no_vis_hist": 5 * n,
            "pos_only": 2 * n,
        }
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(_obs_dims[obs_mode],),
            dtype=np.float64,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.NUM_ROBOTS, 2),
            dtype=np.float32,
        )

        self.r_time = 1.0
        self.r_goal = 300.0
        self.r_success = 2000.0
        self.r_collision = 1200.0
        self.r_progress = 8.0
        self.r_action_coeff = 0.02
        self.r_speed_coeff = 0.01
        self.r_path_coeff = 0.02
        self.reset()

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

        info = {f"robot{i}": self.robots[i, :2].copy() for i in range(self.NUM_ROBOTS)}
        return obs, info

    def _nearest_unvisited_goal_dists(self, positions):
        unvisited = [
            gp for gp, vis in zip(self.goal_positions, self.goal_visited) if not vis
        ]
        if not unvisited:
            return np.zeros(self.NUM_ROBOTS, dtype=np.float64)
        target_arr = np.array(unvisited, dtype=np.float64)
        dists_mat = np.linalg.norm(
            positions[:, None] - target_arr[None, :], axis=2
        )
        return np.min(dists_mat, axis=1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.collision_occurred = False
        self.collision_point = (0.0, 0.0)
        self.goal_visited = [False] * len(self.goal_positions)
        self.robot_paths = [[] for _ in range(self.NUM_ROBOTS)]
        self.t = 0

        if self.dr_mode == "none":
            self.wind_mag = self.base_wind_mag + np.random.normal(0, self.wind_noise_std)
            self.wind_dir = self.base_wind_dir + np.random.normal(
                0, self.wind_dir_noise_std
            )
            self.action_noise_std = self._nominal_action_noise_std
            self.MAX_SPEED = self._nominal_max_speed
            self.MAX_STEER = self._nominal_max_steer
            self.accel_scale = self._nominal_accel_scale
        elif self.dr_mode == "wind":
            self.wind_mag = float(np.random.uniform(0.0, 1.0))
            self.wind_dir = float(np.degrees(np.random.uniform(0.0, 2 * np.pi)))
            self.action_noise_std = self._nominal_action_noise_std
            self.MAX_SPEED = self._nominal_max_speed
            self.MAX_STEER = self._nominal_max_steer
            self.accel_scale = self._nominal_accel_scale
        elif self.dr_mode == "full":
            self.wind_mag = float(np.random.uniform(0.0, 1.0))
            self.wind_dir = float(np.degrees(np.random.uniform(0.0, 2 * np.pi)))
            self.action_noise_std = float(np.random.uniform(0.01, 0.10))
            self.MAX_SPEED = self._nominal_max_speed * float(
                np.random.uniform(0.85, 1.15)
            )
            self.MAX_STEER = self._nominal_max_steer * float(
                np.random.uniform(0.85, 1.15)
            )
            self.accel_scale = self._nominal_accel_scale * float(
                np.random.uniform(0.80, 1.20)
            )

        self.robots = []
        for i in range(self.NUM_ROBOTS):
            x, y, theta = copy.deepcopy(self.init_ROBOTS[i])
            x = np.clip(
                x + np.random.normal(0, self.init_position_noise), 0.0, self.world_width
            )
            y = np.clip(
                y + np.random.normal(0, self.init_position_noise), 0.0, self.world_height
            )
            theta += np.random.normal(0, self.init_heading_noise)
            self.robots.append([x, y, theta, 0.0, 0.0])

        self.robots = np.array(self.robots, dtype=np.float64)
        self.total_path_length = 0.0
        self.prev_positions = self.robots[:, :2].copy()
        self.prev_nearest_goal_dists = self._nearest_unvisited_goal_dists(
            self.prev_positions
        )
        return self._get_obs()

    def step(self, action):
        terminated, truncated = False, False
        term_cond = ""
        self.dec_g = binary_list_to_decimal(self.goal_visited)
        reward = -self.r_time
        self.t += 1

        wind_mag = self.wind_mag + np.random.normal(0, self.wind_noise_std)
        wind_dir = self.wind_dir + np.random.normal(0, self.wind_dir_noise_std)
        w_theta = np.radians(wind_dir)
        wind = np.array(
            [wind_mag * np.cos(w_theta), wind_mag * np.sin(w_theta)]
        )

        robot_polygons = []
        for i in range(self.NUM_ROBOTS):
            a_noisy = float(action[i][0]) + np.random.normal(0, self.action_noise_std)
            d_delta_noisy = float(action[i][1]) + np.random.normal(
                0, self.action_noise_std
            )

            a = a_noisy * self.accel_scale
            d_delta = d_delta_noisy
            x, y, theta, v, delta = self.robots[i]

            v = np.clip(v + a * self.dt, -self.MAX_SPEED, self.MAX_SPEED)
            delta = np.clip(delta + d_delta, -self.MAX_STEER, self.MAX_STEER)

            omega = (
                (v / (self.ROBOT_LENGTH / np.tan(delta)))
                if abs(delta) > 1e-4
                else 0.0
            )
            theta += omega * self.dt
            x += v * np.cos(theta) * self.dt + wind[0] * self.dt
            y += v * np.sin(theta) * self.dt + wind[1] * self.dt
            x = np.clip(x, 0.0, self.world_width)
            y = np.clip(y, 0.0, self.world_height)

            if self.reward_ablation != "no_path":
                reward -= self.r_action_coeff * (a_noisy**2 + d_delta_noisy**2)
                reward -= self.r_speed_coeff * abs(v)

            robot_poly = get_robot_polygon(
                x, y, theta, self.ROBOT_LENGTH, self.ROBOT_WIDTH
            )

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
                    info.update(self._episode_info(term_cond))
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
                    info.update(self._episode_info(term_cond))
                    return obs, reward, terminated, truncated, info
            robot_polygons.append(robot_poly)

            for j, (gx, gy) in enumerate(self.goal_positions):
                if (
                    not self.goal_visited[j]
                    and Point(gx, gy).buffer(self.goal_radius).intersects(robot_poly)
                ):
                    reward += self.r_goal
                    self.goal_visited[j] = True

            self.robots[i] = [x, y, theta, v, delta]
            self.robot_paths[i].append((x, y))

        current_positions = self.robots[:, :2]
        step_path = float(
            np.sum(np.linalg.norm(current_positions - self.prev_positions, axis=1))
        )
        self.total_path_length += step_path
        self.prev_positions = current_positions.copy()

        if self.reward_ablation != "no_path":
            reward -= self.r_path_coeff * step_path

        curr_nearest_dists = self._nearest_unvisited_goal_dists(current_positions)
        progress = np.clip(self.prev_nearest_goal_dists - curr_nearest_dists, -5.0, 5.0)
        reward += self.r_progress * float(np.sum(progress))
        self.prev_nearest_goal_dists = curr_nearest_dists

        if sum(self.goal_visited) >= len(self.goal_positions):
            if self.reward_ablation != "no_term":
                reward += self.r_success
            term_cond = "all_goals"
            terminated = True

        truncated = self.t >= self.max_steps
        if truncated:
            term_cond = "max_steps"

        obs, info = self._get_obs()
        info.update(self._episode_info(term_cond))
        return obs, reward, terminated, truncated, info

    def _episode_info(self, term_cond: str) -> dict:
        return {
            "step_count": self.t,
            "goals_visited": int(sum(self.goal_visited)),
            "path_length": self.total_path_length,
            "term_cond": term_cond,
            "wind_mag": self.wind_mag,
            "wind_dir": self.wind_dir,
        }

    def close(self):
        pass
