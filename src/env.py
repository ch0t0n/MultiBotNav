import json
import itertools
import numpy as np
import copy
import pygame
import gymnasium as gym
from shapely import Polygon, Point
from src.utils import *

class MultiUAV(gym.Env):
    metadata = {'render_modes': ['human', 'print', 'rgb_array'], "render_fps": 4}
    def __init__(self, field_info, render_mode=None, wind_par=[0,0], num_robots=3):
        super().__init__()
        self.field_info = copy.deepcopy(field_info)
        # Screen dimensions
        self.edge_buffer = 10 # Boundary above the max values
        self.poly_vertices = self.field_info['field'] # Vertices of polygon
        self.xs, self.ys = zip(*self.field_info['field']) # x and y values of the vertices of the polygonal field
        self.WIDTH, self.HEIGHT = 1000, 1000 # Use this if we want to have fixed width and height  
        # self.WIDTH, self.HEIGHT = max(self.xs) + self.edge_buffer, max(self.ys) + self.edge_buffer        

        # Robot parameters
        self.num_robots = num_robots # Rendering error if more than 7
        self.init_robot_positions = np.array(self.field_info['init_positions'])[:self.num_robots]
        self.robot_size = 10
        self.mass = 1.0
        self.thrust_power = 0.5  # Force applied per action
        self.max_speed = 5  # Maximum speed    
        self.min_speed = -5 # Minimum speed
        self.min_positions = np.zeros(self.num_robots*2) # Minimum positions
        self.max_positions = np.array([[self.WIDTH, self.HEIGHT] for _ in range(self.num_robots)]) # Maximum positions
        self.min_velocities = np.array([[self.min_speed, self.min_speed] for _ in range(self.num_robots)]) # Min speed list
        self.max_velocities = np.array([[self.max_speed, self.max_speed] for _ in range(self.num_robots)]) # Max speed list
        self.wind_f_a, self.wind_beta_a = wind_par # Wind parameters: magnitude and angle

        # infected locations
        self.initial_inf_locations = self.field_info['infected_locations']
        self.infected_size = 10 # Radius of infected locations
        self.infected_length = len(self.field_info['infected_locations'])
        self.infected_state_length = 2**(self.infected_length) # 2**5, binary to decimal

        # Action space: thrust in x and y directions for each robot
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_robots, 2), dtype=np.float32)

        # Observation space: position and velocity (x, y, vx, vy) for each robot + infected location        
        self.observation_space = gym.spaces.Box( # The (visited) weed locations are tracked on the observation space
                    low = np.concatenate((self.min_positions.flatten(), self.min_velocities.flatten(), np.array([0]))), # Lowest positions and velocities
                    high = np.concatenate((self.max_positions.flatten(), self.max_velocities.flatten(), np.array([self.infected_state_length - 1]))), # highest positions and velocities
                    dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"] # Check if the render mode is correct
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        # If human-rendering is used, `self.screen` will be a reference to the screen that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in human-mode. They will remain `None` until human-mode is used for the first time.   

        # Reset the environment and start
        self.reset()
    
    def _get_obs(self):
        info = {f'robot{i}': self.robot_positions[i] for i in range(self.num_robots)} # Current position of each robot
        infected = binary_list_to_decimal(list(self.infected_dict.values())) # Convert the binary list of infected locations to a decimal value
        state = np.concatenate((self.robot_positions.flatten(), self.robot_velocities.flatten(), np.array([infected])), dtype=np.float32) # Current state of the robots
        return state, info        

    def reset(self, seed=None, options={}):
        # Reset the visited states and counts
        self.step_count = 0
        self.visited = set()
        self.infected_locations = copy.deepcopy(self.initial_inf_locations) # Initial infected locations
        self.infected_dict = {v:0 for v in self.infected_locations} # 0 for unvisited infected locations, 1 for visited
        self.robot_positions = copy.deepcopy(self.init_robot_positions) # Initial positions of each robot
        self.robot_velocities = np.zeros((self.num_robots, 2)) # Initial velocities of each robot (zero)
        return self._get_obs()
    
    def step(self, actions):
        terminated, truncated = False, False
        rewards = 0
        self.step_count += 1
        for i in range(self.num_robots): # For every robot
            ax, ay = actions[i] * self.thrust_power # What actions to take

            # Update velocity
            self.robot_velocities[i][0] += ax / self.mass + self.wind_f_a * np.cos(np.radians(self.wind_beta_a))
            self.robot_velocities[i][1] += ay / self.mass + self.wind_f_a * np.sin(np.radians(self.wind_beta_a))

            # Limit velocity
            self.robot_velocities[i] = np.clip(self.robot_velocities[i], self.min_speed, self.max_speed)

            # Predict new position
            new_position = self.robot_positions[i] + self.robot_velocities[i]

            # Boundary conditions (keep robot within polygon)
            if is_inside_polygon(new_position, self.poly_vertices):
                pass
            else: # Hits the wall!
                rewards -= 10000 # Medium negative reward for hitting the wall
                self.robot_velocities[i][:] = 0 # Stop movement

            # Update position
            self.robot_positions[i] += self.robot_velocities[i]
            
            # Boundary conditions (keep robot within screen)
            self.robot_positions[i] = np.clip(self.robot_positions[i], [0, 0], [self.WIDTH, self.HEIGHT])

            # Check if location is visited before, and add it to the visited locations
            if tuple(self.robot_positions[i]) in self.visited:
                rewards -= 100 # Small negative reward for visiting previous location
            else:
                rewards -= 10 # Very small negative reward for visiting new locations
            self.visited.add(tuple(self.robot_positions[i]))            

            # Check if any infected location is visited        
            nearby_infected_locations = [] # To store the nearby infected locations
            for j, inf_loc in enumerate(self.infected_locations): # Loop through each infected location
                dist = np.linalg.norm(self.robot_positions[i]-inf_loc) # Distance between robot position and infected location
                if dist <= self.infected_size: # If the distance is within the radius of the infected location size
                    nearby_infected_locations.append(inf_loc) # Add the infected location
                    rewards += 10000 # Medium positive rewards for visiting each infected location
                    # input("Pause!") # Only pause if you want to visualize visiting infected locations
            for inf_loc in nearby_infected_locations:
                self.infected_locations.remove(inf_loc) # Delete each visited infected location
                self.infected_dict[tuple(inf_loc)] = 1 # Update the infected dictionary
        
        # Check if all infected locations are visited
        if len(self.infected_locations) == 0:
            rewards += 100000 # Big positive rewards for visiting all infected locations
            terminated = True
        
        # Check if any collisions occurred
        if self.num_robots > 1:
            min_dist_between_robots = min_dist(self.robot_positions) # Minimum distance between robots
            if min_dist_between_robots < self.robot_size:
                rewards = -100000 # Big negative rewards for collisions
                terminated = True

        obs, info = self._get_obs() # Get the updated observations
        # rewards = rewards * self.gamma ** self.step_count
        return obs, rewards, terminated, truncated, info
    
    def render(self):
        # Initialize pygame
        if self.screen is None and self.render_mode == "human": # Initialize pygame if it is not initialized
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Multi-robot RL Environment")
            if self.clock is None:
                self.clock = pygame.time.Clock()
                self.running = True
        
        self.screen.fill((255, 255, 255)) # White color for the background
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 128, 0), (128, 0, 255), (255, 0, 255), (128, 128, 128)]  # Colors for each robot: Red, Green, Blue, Orange, Violet, Pink, Grey
        pix_size = 10

        # Draw the polygon
        # pixel_poly_vertices = [(point[0] * pix_size, point[1] * pix_size) for point in self.poly_vertices]
        pygame.draw.polygon(surface=self.screen, 
                            color=(255, 255, 0), # Yello color for the polygon
                            points=self.poly_vertices)
        
        # Draw the visited regions
        for point in self.visited:
            pygame.draw.circle(self.screen, pygame.Color(100, 100, 100, a=0.2), point, pix_size/2) # Light grey color for visited regions, with transparency alpha

        # Draw robots
        for i in range(self.num_robots):
            pygame.draw.circle(self.screen, colors[i], (int(self.robot_positions[i][0]), int(self.robot_positions[i][1])), pix_size/2) # Pick the colors from above list

        # Draw infected locations
            for l in self.infected_locations:
                pygame.draw.circle(self.screen, (0, 255, 255), (int(l[0]), int(l[1])), pix_size/2) # Cyan color for infected locations
        
        pygame.display.flip() # Allows only a portion of the screen to be updated
        self.clock.tick(60)
    
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


class MultiWheeled(gym.Env):
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
        obs_high = np.concatenate((obs_high, np.array([2**(len(self.obstacles))-1], dtype=np.float64))) # Adding the binary encoding of obstacles
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