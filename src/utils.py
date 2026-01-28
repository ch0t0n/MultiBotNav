import os
import yaml
import numpy as np
from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO, ARS, CrossQ, TQC, RecurrentPPO
import distutils
import itertools
import inspect
from shapely import Polygon
import configparser
import json


# Function to return minimum distance in a list of points
def compute_min_dist(x):
    x = np.array(x).astype('float32')
    dists = []
    for p1, p2 in itertools.combinations(x, 2):
        dist = np.linalg.norm(p1-p2)
        dists.append(dist)
    return float(np.min(dists))

# Load experiment json file
def load_experiment_dict_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    for set_name, cfg in data.items():
        # Convert field to list of tuples
        cfg["field"] = [tuple(p) for p in cfg["field"]]
        # Convert init_positions to NumPy arrays
        cfg["init_positions"] = [np.array(p, dtype=float) for p in cfg["init_positions"]]
        # Convert infected_locations to set of tuples
        cfg["infected_locations"] = [tuple(p) for p in cfg["infected_locations"]]
    return data

# Loads in an experiment config file
def load_experiment(path, sf):
    with open(path, 'r') as experiment_file:
        config = yaml.load(experiment_file, Loader=yaml.FullLoader)
        config['field'] = list(map(lambda x: tuple(x), config['field']))
        config['init_positions'] = list(map(lambda x: np.array(x), config['init_positions']))
        config['infected_locations'] = list(map(lambda x: tuple(x), config['infected_locations']))
    # scaling
    config['field'] = [(x*sf, y*sf) for (x,y) in config['field']]
    # old infected_locations line
    # config['infected_locations'] = [(x*sf, y*sf) for (x,y) in config['infected_locations']]
    config['infected_locations'] = [(x*sf, y*sf, level) for (x, y, level) in config['infected_locations']]
    config['init_positions'] = [v*sf for v in config['init_positions']]
    return config

# Parses a string into a bool
def parse_bool(string):
    return bool(distutils.util.strtobool(string))

# Loads in a trained model
def load_model(algorithm, 
               experiment_set, 
               seed, 
               device, 
               verbose, 
               log_dir, 
               trained_model_path,
               run_name
):
    model_args = {
        'path': trained_model_path, # f'{models_dir}/{algorithm}_set{experiment_set}.zip',
        #'tb_log_name': run_name, # f'{algorithm}_set{experiment_set}',
        'device': device,
        'seed': seed,
        'verbose': verbose,
        'tensorboard_log': log_dir,
    }

    if algorithm == 'A2C':
        model = A2C.load(**model_args)
    elif algorithm == 'PPO':
        model = PPO.load(**model_args)
    elif algorithm == 'TRPO':
        model = TRPO.load(**model_args)
    elif algorithm == 'TQC':
        model = TQC.load(**model_args)
    elif algorithm == 'ARS':
        model = ARS.load(**model_args)
    else:
        model = CrossQ.load(**model_args)
    return model

# Converts a list of binary digits to its decimal equivalent
def binary_list_to_decimal(bin_list):
    bin = ''
    for b in bin_list:
        bin += str(int(b))
    dec = int(bin,2)
    return dec

# Function to check if a point is inside a polygon (Ray-casting algorithm)
def is_inside_polygon(point, poly):
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

# Function to return minimum distance in a list of points
def min_dist(x):
    x = np.array(x).astype('float32')
    dists = []
    for p1, p2 in itertools.combinations(x, 2):
        dist = np.linalg.norm(p1-p2)
        dists.append(dist)
    return float(np.min(dists))

# Filters out arguments that are not present in a model's constructor
def filter_args(args, model):
    model_kwargs = inspect.getfullargspec(model).args
    return {k:args[k] for k in args if k in model_kwargs}

# Updates the robot polygon
def get_robot_polygon(x, y, theta, robot_length, robot_width):
    # Robot corners relative to center
    dx = robot_length / 2
    dy = robot_width / 2
    corners = np.array([
        [ dx,  dy],
        [ dx, -dy],
        [-dx, -dy],
        [-dx,  dy]
    ])

    # Rotation matrix
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    # Rotate and translate corners
    rotated = np.dot(corners, R.T) + np.array([x, y])
    return Polygon(rotated)

# Read set config file
def read_env_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path) # Read the config file
    env_params = {}

    # Load screen parameters
    env_params['SCREEN_WIDTH'] = float(config['SCREEN']['WIDTH'])
    env_params['SCREEN_HEIGHT'] = float(config['SCREEN']['HEIGHT'])

    # Load initial robot positions
    env_params['ROBOT_LENGTH'] = float(config['ROBOTS']['LENGTH'])
    env_params['ROBOT_WIDTH'] = float(config['ROBOTS']['WIDTH'])
    env_params['MAX_SPEED'] = float(config['ROBOTS']['MAX_SPEED'])
    env_params['MAX_STEER'] = float(config['ROBOTS']['MAX_STEER'])
    env_params['NUM_ROBOTS'] = int(config['ROBOTS']['NUM_ROBOTS'])
    env_params['ROBOT_INIT_CONFIGS'] = []
    for i in range(env_params['NUM_ROBOTS']):
        conf = config['ROBOTS'][f'ROBOT_{i+1}']
        x, y, theta = map(float, conf.split(','))
        # env_params[f'ROBOT_{i+1}'] = (x,y,theta)
        env_params['ROBOT_INIT_CONFIGS'].append((x,y,float(np.radians(theta))))

    # Load goal positions
    env_params['NUM_GOALS'] = int(config['GOALS']['NUM_GOALS'])
    env_params['GOAL_SIZE'] = float(config['GOALS']['GOAL_SIZE'])
    env_params['GOAL_POSITIONS'] = []
    for i in range(env_params['NUM_GOALS']):
        g_pos = config['GOALS'][f'GOAL_{i+1}']
        x, y = map(float, g_pos.split(','))
        # env_params[f'GOAL_{i+1}'] = (x,y)
        env_params['GOAL_POSITIONS'].append((x,y))

    # Load polygonal obstacles
    env_params['NUM_OBSTACLES'] = int(config['OBSTACLES']['NUM_OBSTACLES'])
    env_params['OBSTACLES'] = []
    for i in range(env_params['NUM_OBSTACLES']):
        ver_str = config['OBSTACLES'][f'OBSTACLE_{i+1}']
        points = [tuple(map(int, pt.split(','))) for pt in ver_str.split(';')]
        # env_params[f'OBSTACLE_{i+1}'] = (x,y)
        env_params['OBSTACLES'].append(points)    
    return env_params