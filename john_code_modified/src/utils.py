import yaml
import numpy as np
from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO, ARS, CrossQ, TQC
import distutils
import itertools

# Loads in an experiment config file
def load_experiment(path, sf):
    with open(path, 'r') as experiment_file:
        config = yaml.load(experiment_file, Loader=yaml.FullLoader)
        config['field'] = list(map(lambda x: tuple(x), config['field']))
        config['init_positions'] = list(map(lambda x: np.array(x), config['init_positions']))
        config['infected_locations'] = list(map(lambda x: tuple(x), config['infected_locations']))
    # scaling
    config['field'] = [(x*sf, y*sf) for (x,y) in config['field']]
    config['infected_locations'] = [(x*sf, y*sf) for (x,y) in config['infected_locations']]
    config['init_positions'] = [v*sf for v in config['init_positions']]
    return config

# Loads in a trained model
def load_model(algorithm, st):
    model_path = f'trained_models/{algorithm}_set{st}.zip'
    if algorithm == 'A2C':
        model = A2C.load(model_path, tb_log_name=f'{algorithm}_set{st}')
    elif algorithm == 'PPO':
        model = PPO.load(model_path, tb_log_name=f'{algorithm}_set{st}')
    elif algorithm == 'TRPO':
        model = TRPO.load(model_path, tb_log_name=f'{algorithm}_set{st}')
    elif algorithm == 'ARS':
        model = ARS.load(model_path, tb_log_name=f'{algorithm}_set{st}')
    elif algorithm == 'CrossQ':
        model = CrossQ.load(model_path, tb_log_name=f'{algorithm}_set{st}')
    elif algorithm == 'TQC':
        model = TQC.load(model_path, tb_log_name=f'{algorithm}_set{st}')
    return model

# Parses a string into a bool
def parse_bool(string):
    return bool(distutils.util.strtobool(string))

# convert binary list to decimal
def binary_list_to_decimal(bin_list):
    bin = ''
    for b in bin_list:
        bin += str(b)
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
