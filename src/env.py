# Import libraries
import copy
import pygame
import gymnasium as gym
from src.utils import binary_list_to_decimal, get_robot_polygon
import numpy as np
from shapely import Polygon, Point

class MultiRobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    def __init__(self, env_params, render_mode=None):
        super().__init__()        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.dt = 0.05 # timestep
        # self.ds = np.radians(5) # rate of change of steering angle per time step

        # Screen parameters
        self.WIDTH, self.HEIGHT = env_params['SCREEN_WIDTH'], env_params['SCREEN_HEIGHT']

        # Robot parameters
        self.ROBOT_LENGTH = env_params['ROBOT_LENGTH']
        self.ROBOT_WIDTH = env_params['ROBOT_WIDTH']
        self.MAX_SPEED = env_params['MAX_SPEED']
        self.MAX_STEER = np.radians(env_params['MAX_STEER'])

        # Initial configurations
        self.NUM_ROBOTS = env_params['NUM_ROBOTS']
        self.init_ROBOTS = env_params['ROBOT_INIT_CONFIGS']

        # Obstacles and goals
        self.goal_radius = env_params['GOAL_SIZE']
        self.goal_positions = env_params['GOAL_POSITIONS']
        self.obstacles = env_params['OBSTACLES']

        # Each robot's state: x, y, theta, v, delta
        obs_high = np.array([self.WIDTH, self.HEIGHT, np.pi, self.MAX_SPEED, self.MAX_STEER] * self.NUM_ROBOTS, dtype=np.float64)
        obs_high = np.concatenate((obs_high, np.array([2**(len(self.goal_positions))-1], dtype=np.float64))) # Adding the binary encoding of obstacles
        obs_low = np.array([0, 0, -np.pi, -self.MAX_SPEED, -self.MAX_STEER] * self.NUM_ROBOTS, dtype=np.float64)
        obs_low = np.concatenate((obs_low, np.array([0], dtype=np.float64)))
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)

        # Action: (accel, delta_change) for each robot
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.NUM_ROBOTS, 2), dtype=np.float32)

        # initialize reward values
        self.r_s, self.r_l, self.r_M = 10, 10000, 100000

        # Reset the environment and start
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.collision_occurred = False
        self.collision_point = (0,0)
        self.goal_visited = [False for _ in self.goal_positions]
        self.robot_paths = [[] for _ in range(self.NUM_ROBOTS)]

        # Initialize robot states
        v, delta = 0.0, 0.0 # Initial velocity and steering rate

        self.robots = [] # List to store the initial states of n robots
        self.configs = copy.deepcopy(self.init_ROBOTS) # Read the initial configurations

        for i in range(self.NUM_ROBOTS):
            x, y, theta = self.configs[i] # read config of each robot
            self.robots.append([x, y, theta, v, delta]) # state of each robot
        

        self.robots = np.array(self.robots, dtype=np.float64)
        self.t = 0

        if self.render_mode == "human":
            self._render_pygame()
        
        obs, info = self._get_obs()
        return obs, info

    def _get_obs(self):        
        self.dec_g = binary_list_to_decimal(self.goal_visited) # Convert the visited goals to decimal value
        obs = np.concatenate((self.robots.flatten(), np.array([self.dec_g]))) # Add the encodings of goals
        info = {f'robot{i}': self.robots[i][:2] for i in range(self.NUM_ROBOTS)} # Current x,y positions of each robot
        return obs, info

    def step(self, action):
        terminated, truncated = False, False        
        reward = -(self.r_s / self.dec_g) if (self.dec_g != 0) else -self.r_s  # base step penalty
        self.t += 1 # step_count
        robot_polygons = [] # to store the space occupied by each robot

        for i in range(self.NUM_ROBOTS): # For every robot
            a = float(action[i][0]) * 100.0 # Acceleration 
            d_delta = float(action[i][1]) #* self.ds # Rate of change in steering
            x, y, theta, v, delta = self.robots[i] # get the current state

            # Update state
            v = np.clip(v + a * self.dt, -self.MAX_SPEED, self.MAX_SPEED)
            delta = np.clip(delta + d_delta, -self.MAX_STEER, self.MAX_STEER)

            if abs(delta) > 1e-4: # avoid division by zero or numerical instability
                R = self.ROBOT_LENGTH / np.tan(delta)
                omega = v / R
            else:
                omega = 0.0 # go straight for a very small delta

            theta += omega * self.dt
            theta = (theta + np.pi) % (2 * np.pi) - np.pi
            x += v * np.cos(theta) * self.dt
            y += v * np.sin(theta) * self.dt

            # Keep inside bounds
            x = np.clip(x, 0, self.WIDTH)
            y = np.clip(y, 0, self.HEIGHT)            

            # Make the robot polygon object
            robot_poly = get_robot_polygon(x, y, theta, self.ROBOT_LENGTH, self.ROBOT_WIDTH)

            # Check for collisions with obstacles                       
            for obs_pts in self.obstacles: 
                obs_poly = Polygon(obs_pts)
                if robot_poly.intersects(obs_poly): # if collisions occur
                    inter = robot_poly.intersection(obs_poly)
                    int_point = inter.representative_point()
                    self.collision_point = (int_point.x, int_point.y)
                    reward = -self.r_M # very large negative reward
                    terminated = True
                    self.collision_occurred = True
            
            # Check for collisions between robots
            if robot_polygons: # If there exists other robot polygons
                for robot_i in robot_polygons: # For each robot polygon
                    if robot_poly.intersects(robot_i): # if collisions occur
                        inter = robot_poly.intersection(robot_i)
                        int_point = inter.representative_point()
                        self.collision_point = (int_point.x, int_point.y)
                        reward = -self.r_M
                        terminated = True
                        self.collision_occurred = True                      
            robot_polygons.append(robot_poly) # Add the current robot to the robot polygons

            # Check if a goal region is visited
            for j, (gx, gy) in enumerate(self.goal_positions): # Loop through each goal region
                if not self.goal_visited[j]:
                    goal_point = Point(gx,gy)
                    goal_point = goal_point.buffer(self.goal_radius) # define the goal region
                    if goal_point.intersects(robot_poly): # If the goal point is reached                        
                        reward += self.r_l # Get a large reward for visiting each goal region                        
                        self.goal_visited[j] = True
                        # print(f"GOAL {j} reached!!! Goals visited: {sum(self.goal_visited)}, reward: {reward}")
            
            # Record the robot path
            self.robots[i] = [x, y, theta, v, delta]
            self.robot_paths[i].append((x, y))

        # Check if all goal regions are visited
        if sum(self.goal_visited) >= len(self.goal_positions):
            reward += self.r_M
            # print(f"All goals reached!!! Goals visited: {self.goal_visited}, reward: {reward}")
            terminated = True

        if self.render_mode == "human":
            self._render_pygame()
        
        obs, info = self._get_obs() # Get the updated observations
        return obs, reward, terminated, truncated, info

    def _render_pygame(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Multi-Robot Environment")
            self.clock = pygame.time.Clock()

        self.screen.fill((240, 240, 240))

        # Draw obstacles
        for obs in self.obstacles:
            # pygame.draw.rect(self.screen, (100, 100, 100), obs)
            pygame.draw.polygon(self.screen, (100, 100, 100), obs)

        # Draw goal regions    
        for (gx, gy), visited in zip(self.goal_positions, self.goal_visited):
            color = (0, 255, 100) if not visited else (0, 0, 0)
            pygame.draw.circle(self.screen, color, (int(gx), int(gy)), self.goal_radius, 0)

        # Draw collision point if collision occured
        if self.collision_occurred:
            # draw_explosion(self.screen, self.collision_point)
            pygame.draw.circle(self.screen, color=(255,0,0), center=self.collision_point, radius=10)
            exp_sound = pygame.mixer.Sound(r'./assets/audio/boom.wav')
            exp_sound.play()

        # Draw robot paths
        for path in self.robot_paths:
            if len(path) > 1:
                pygame.draw.lines(self.screen, color=[255,0,0], closed=False, points=path, width=2)


        for x, y, theta, _, delta in self.robots:
            # --- Draw robot body ---
            robot_surf = pygame.Surface((self.ROBOT_LENGTH, self.ROBOT_WIDTH), pygame.SRCALPHA)
            robot_surf.fill((0, 128, 255))
            pygame.draw.line(robot_surf, (255, 0, 0), (self.ROBOT_LENGTH // 2, self.ROBOT_WIDTH // 2), (self.ROBOT_LENGTH, self.ROBOT_WIDTH // 2), 2)
            rotated_robot = pygame.transform.rotate(robot_surf, -np.degrees(theta))
            rect = rotated_robot.get_rect(center=(x, y))
            self.screen.blit(rotated_robot, rect)

            # --- Draw wheels ---
            offset_x = self.ROBOT_LENGTH // 2 - self.ROBOT_WIDTH / 5
            offset_y = self.ROBOT_WIDTH // 2 - self.ROBOT_WIDTH / 5
            wheel_offsets = {
                'front_left':  (offset_x, -offset_y),
                'front_right': (offset_x, offset_y),
                'rear_left':   (-offset_x, -offset_y),
                'rear_right':  (-offset_x, offset_y),
            }

            for pos, (dx, dy) in wheel_offsets.items():
                # Rotate wheel positions into global frame
                x_local = dx
                y_local = dy
                x_global = x + np.cos(theta)*x_local - np.sin(theta)*y_local
                y_global = y + np.sin(theta)*x_local + np.cos(theta)*y_local

                # Wheel surface
                wheel_surf = pygame.Surface((self.ROBOT_LENGTH/6, 1), pygame.SRCALPHA)
                wheel_surf.fill((20, 20, 20))

                # Determine wheel rotation
                if 'front' in pos:
                    wheel_angle = theta + delta
                else:
                    wheel_angle = theta

                rotated_wheel = pygame.transform.rotate(wheel_surf, -np.degrees(wheel_angle))
                wheel_rect = rotated_wheel.get_rect(center=(x_global, y_global))
                self.screen.blit(rotated_wheel, wheel_rect)        

        pygame.display.flip()
        self.clock.tick(60)
        pygame.event.get()

    def render(self):
        self._render_pygame()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
