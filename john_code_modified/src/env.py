# Import libraries
import copy
import pygame
import gymnasium as gym
import itertools
from src.utils import binary_list_to_decimal, is_inside_polygon, min_dist
import numpy as np
np.random.seed(33) # seeding

class MultiBotNavigator(gym.Env):
    metadata = {'render_modes': ['human', 'print', 'rgb_array'], "render_fps": 4}
    def __init__(self, render_mode=None, env_config=None, wind_par=[0,0], num_robots=3):
        super(MultiBotNavigator, self).__init__()
        self.config = copy.deepcopy(env_config) # Load the config file
        self.edge_buffer = 10 # Boundary above the max values
        self.poly_vertices = self.config['field'] # Vertices of polygon
        self.xs, self.ys = zip(*self.config['field']) # x and y values of the vertices of the polygonal field
        self.WIDTH, self.HEIGHT = 1000, 1000 # Use this if we want to have fixed width and height 
        # self.WIDTH, self.HEIGHT = max(self.xs) + self.edge_buffer, max(self.ys) + self.edge_buffer

        # Robot parameters
        self.num_robots = num_robots # Rendering error if more than 7
        self.init_robot_positions = np.array(self.config['init_positions'])[:self.num_robots]
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
        self.initial_inf_locations = self.config['infected_locations']
        self.infected_size = 10 # Radius of infected locations
        self.infected_length = len(self.config['infected_locations'])
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
