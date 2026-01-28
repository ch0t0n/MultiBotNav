import json
import itertools
import numpy as np
import copy
import pygame
import gymnasium as gym
from src.utils import compute_min_dist, is_inside_polygon

class MultiRobotEnv(gym.Env):
    metadata = {'render_modes': ['human', 'print', 'rgb_array'], "render_fps": 4}    
    def __init__(self, field_info, render_mode=None, wind_par=[0,0], num_robots=3, render_scale=10):
        super().__init__()

        # Screen, rendering (simulation) and field parameters
        self.field_info = copy.deepcopy(field_info)
        self.render_scale = render_scale # Scaling factor for rendering only 
        self.poly_vertices = self.field_info['field']  # Physics coordinates
        self.xs, self.ys = zip(*self.field_info['field']) # x and y values of the vertices of the polygonal field
        self.WIDTH, self.HEIGHT = max(self.xs) + 10, max(self.ys) + 10 # Boundary above the max values
        assert render_mode is None or render_mode in self.metadata["render_modes"] # Check valid render modes
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # Robot parameters
        self.num_robots = num_robots
        self.robot_size = 2
        self.mass = 1.0
        self.thrust_power = 0.5 # Force applied per action
        self.max_speed = 5
        self.min_speed = -5
        self.init_robot_positions = np.array(self.field_info['init_positions'])[:self.num_robots]
        self.init_robot_capacities = np.array(self.field_info['robot_capacities'][:self.num_robots], dtype=np.float32)
        self.max_capacity = np.max(self.init_robot_capacities)
        self.wind_f_a, self.wind_beta_a = wind_par # Wind parameters: magnitude and angle

        # Infected locations with levels
        self.init_infected_locations = [(np.array([x, y]), level) for x, y, level in self.field_info['infected_locations']]
        self.max_infection_level = max(lvl for _, lvl in self.init_infected_locations)
        self.infected_radius = 5
        self.spray_sigma = self.infected_radius / 2.0
        self.n_infected = len(self.init_infected_locations)
        
        # Action space (ax, ay, spray) and Observation space (x, y, vx, vy, inf_loc): position and velocity for each robot and infected location
        self.action_space = gym.spaces.Box(
            low=np.array([[-1.0, -1.0, 0.0]] * self.num_robots),
            high=np.array([[1.0, 1.0, 1.0]] * self.num_robots), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=np.concatenate((
                np.zeros(self.num_robots * 2),                # (x, y) co-ordinates of each robot
                np.full(self.num_robots * 2, self.min_speed), # (v_x, v_y) velocities along x and y axis of each robot
                np.zeros(self.num_robots), # robot spraying capacities
                np.zeros(self.n_infected)  # infection levels of each infected locations
            )),
            high=np.concatenate((
                np.tile([self.WIDTH, self.HEIGHT], self.num_robots), # (x, y) co-ordinates of each robot
                np.full(self.num_robots * 2, self.max_speed), # (v_x, v_y) velocities along x and y axis of each robot
                np.full(self.num_robots, self.max_capacity), # Full capacity for all robots
                np.ones(self.n_infected) # infection levels of each infected locations
            )), dtype=np.float32
        )
        
        self.reset() # Reset environment and start

    def _get_obs(self):
        info = {f'robot{i}': {'position': self.robot_positions[i], 'capacity': self.robot_capacities[i]}
                    for i in range(self.num_robots)}
        infection_levels = np.array([lvl / self.max_infection_level 
                                     for _, lvl in self.infected_locations], dtype=np.float32) # Normalized infection levels for infected locations
        state = np.concatenate((self.robot_positions.flatten(), self.robot_velocities.flatten(),
                                self.robot_capacities, infection_levels), dtype=np.float32)
        return state, info

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)
        self.robot_positions = copy.deepcopy(self.init_robot_positions) # Initial robot locations
        self.robot_velocities = np.zeros((self.num_robots, 2), dtype=np.float32) # Initial velocities of each robot (zero)
        self.robot_capacities = copy.deepcopy(self.init_robot_capacities) # Initial robot capacities
        self.infected_locations = copy.deepcopy(self.init_infected_locations) # Initial infected locations
        self.visited = set() # Keep track of visited states
        return self._get_obs()

    def step(self, actions):
        terminated, truncated = False, False
        rewards = 0
        for i in range(self.num_robots): # For every robot
            ax, ay, spray = actions[i] # Get the position and spray levels for the robot

            # ----- Gaussian area spray -----
            if spray > 0 and self.robot_capacities[i] > 0: # Check spraying level
                for idx, (loc, level) in enumerate(self.infected_locations): # Check each infected location
                    dist = np.linalg.norm(self.robot_positions[i] - loc) # Distance between robot and infected location
                    if dist <= self.infected_radius and level > 0: # If the distance is within infected radius
                        w = np.exp(-(dist ** 2) / (2 * self.spray_sigma ** 2)) # Gaussian spray value
                        applied = min(spray * w, self.robot_capacities[i], level) # Take the minimum of (spray_amount, robot_capacity, infection level)
                        self.robot_capacities[i] -= applied # Reduce the robot capacity
                        self.infected_locations[idx] = (loc, level - applied) # Updated the spraying level
                        rewards += 2000.0 * applied

            # Movement by robot dynamics
            ax, ay = ax * self.thrust_power, ay * self.thrust_power
            self.robot_velocities[i, 0] += ax / self.mass + self.wind_f_a * np.cos(np.radians(self.wind_beta_a))
            self.robot_velocities[i, 1] += ay / self.mass + self.wind_f_a * np.sin(np.radians(self.wind_beta_a))
            self.robot_velocities[i] = np.clip(self.robot_velocities[i], self.min_speed, self.max_speed)

            # Check new position and update visiteds states
            new_position = self.robot_positions[i] + self.robot_velocities[i]
            if not is_inside_polygon(new_position, self.poly_vertices): # If new position is outside the field
                rewards -= 10000 # Big negative reward for going outside the field
                self.robot_velocities[i][:] = 0
            else:
                self.robot_positions[i] = new_position
            pos_key = tuple(np.round(self.robot_positions[i], 1))
            if pos_key in self.visited: # If current location is visited previously
                rewards -= 100 # Small negative reward for visiting same state
            else:
                rewards -= 10 # Very small negative reward for visiting new state
            self.visited.add(pos_key)

        if all(level <= 0.01 for _, level in self.infected_locations): # If all infected locations are cleared
            rewards += 100000 # Very big reward for visiting all infected locations and terminate
            terminated = True

        if self.num_robots > 1:
            min_dist_between_robots = compute_min_dist(self.robot_positions)
            if min_dist_between_robots < self.robot_size: # Check if any collisions occured
                rewards = -100000 # Very big negative reward for collisions and terminate
                terminated = True

        obs, info = self._get_obs()
        return obs, rewards, terminated, truncated, info

    def render(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.WIDTH*self.render_scale, self.HEIGHT*self.render_scale))
            pygame.display.set_caption("Multi-robot RL Environment")
            if self.clock is None:
                self.clock = pygame.time.Clock()
                self.running = True

        self.screen.fill((255, 255, 255))
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 128, 0),
                  (128, 0, 255), (255, 0, 255), (128, 128, 128)]
        pix_size = 10

        # Draw polygon scaled
        scaled_poly = [(x*self.render_scale, y*self.render_scale) for x, y in self.poly_vertices]
        pygame.draw.polygon(surface=self.screen, color=(255, 255, 0), points=scaled_poly)

        # Draw visited points scaled
        for point in self.visited:
            scaled_point = (int(point[0]*self.render_scale), int(point[1]*self.render_scale))
            pygame.draw.circle(self.screen, pygame.Color(100, 100, 100, a=0.2), scaled_point, pix_size//2)

        # Draw robots scaled
        for i in range(self.num_robots):
            scaled_pos = (int(self.robot_positions[i][0]*self.render_scale), int(self.robot_positions[i][1]*self.render_scale))
            pygame.draw.circle(self.screen, colors[i], scaled_pos, pix_size//2)

        # Draw infected locations scaled
        for loc, level in self.infected_locations:
            intensity = int(255 * min(level/5.0, 1.0))
            scaled_loc = (int(loc[0]*self.render_scale), int(loc[1]*self.render_scale))
            pygame.draw.circle(self.screen, (0, intensity, 255), scaled_loc, 6)

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()