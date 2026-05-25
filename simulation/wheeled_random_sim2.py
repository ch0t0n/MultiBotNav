# Hyperparameters and paths
max_steps = 1000

import os
import json
import numpy as np
import copy
import pygame
from shapely import Polygon, Point
import gymnasium as gym

# Path to the consolidated JSON config and the environment key to load
wheeled_json_path = os.path.join('exp_sets', 'wheeled', 'wheeled_configs.json')
env_key           = 'env10'   # change to 'env2' … 'env10' to simulate other environments


def load_env_from_json(json_path, key='env1'):
    """
    Load a single wheeled-robot environment config from the consolidated JSON.

    The JSON maps environment keys ("env1" … "env10") to config dicts with
    the structure produced by the generation script (screen / robots / goals /
    obstacles sections, robot headings stored in degrees).  Returns a flat
    env_params dict with the exact keys expected by MultiWheeled:
        SCREEN_WIDTH, SCREEN_HEIGHT, ROBOT_LENGTH, ROBOT_WIDTH,
        MAX_SPEED, MAX_STEER, NUM_ROBOTS,
        ROBOT_INIT_CONFIGS  ← list of (x, y, theta_rad),
        NUM_GOALS, GOAL_SIZE, GOAL_POSITIONS,
        NUM_OBSTACLES, OBSTACLES.
    """
    with open(json_path, 'r') as f:
        raw = json.load(f)

    if key not in raw:
        raise KeyError(f"Environment key '{key}' not found in {json_path}. "
                       f"Available keys: {list(raw.keys())}")

    cfg = raw[key]
    r   = cfg['robots']
    g   = cfg['goals']
    obs = cfg['obstacles']

    return {
        'SCREEN_WIDTH':       float(cfg['screen']['width']),
        'SCREEN_HEIGHT':      float(cfg['screen']['height']),
        'ROBOT_LENGTH':       float(r['length']),
        'ROBOT_WIDTH':        float(r['width']),
        'MAX_SPEED':          float(r['max_speed']),
        'MAX_STEER':          float(r['max_steer']),
        'NUM_ROBOTS':         int(r['num_robots']),
        # JSON stores theta in degrees; MultiWheeled expects radians
        'ROBOT_INIT_CONFIGS': [(float(c[0]), float(c[1]), float(np.radians(c[2])))
                               for c in r['configs']],
        'NUM_GOALS':          int(g['num_goals']),
        'GOAL_SIZE':          float(g['goal_size']),
        'GOAL_POSITIONS':     [tuple(p) for p in g['positions']],
        'NUM_OBSTACLES':      len(obs),
        'OBSTACLES':          [[tuple(v) for v in poly] for poly in obs],
    }


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


# convert binary list to decimal
def binary_list_to_decimal(bin_list):
    """Convert a list of 0/1 values to its decimal integer equivalent."""
    return int("".join(str(int(b)) for b in bin_list), 2)


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
        elif self.obs_mode == "no_vis_hist":
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
        term_cond = ""
        # Recompute dec_g fresh at the start of each step
        self.dec_g = binary_list_to_decimal(self.goal_visited)
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
                        reward = -self.r_M
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
if __name__ == "__main__":
    if 'MultiWheeled-v0' not in gym.envs.registry:
        gym.register(id='MultiWheeled-v0', entry_point=MultiWheeled, max_episode_steps=1000)

    env_params = load_env_from_json(wheeled_json_path, key=env_key)

    # Make the environment
    env = gym.make('MultiWheeled-v0', env_params=env_params, render_mode='human', max_steps=max_steps)
    env.unwrapped.metadata['render_fps'] = 30
    obs, info = env.reset()
    env.render()
    pygame.image.save(env.unwrapped.screen, f"wheeled_init_{env_key}.jpg")

    total_rewards = 0
    total_steps   = 0

    for i in range(10000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_rewards += reward
        total_steps   += 1

        env.render()

        # Drain the event queue so the OS doesn't mark the window as "not responding"
        # and so the user can close it with the X button.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                raise SystemExit("Window closed by user.")

        if terminated or truncated:
            print(f"Episode ended | steps: {total_steps} | reward: {total_rewards:.1f} "
                  f"| term_cond: {info.get('term_cond', '?')}")
            print(f"Observation: {obs}")

            pause_ms   = 1
            start_tick = pygame.time.get_ticks()
            while pygame.time.get_ticks() - start_tick < pause_ms:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        raise SystemExit("Window closed by user.")

            obs, _ = env.reset()
            total_rewards, total_steps = 0, 0

    env.close()