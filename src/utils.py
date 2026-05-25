import os
import re
import json
import random
import configparser
import inspect
import numpy as np
import torch
from shapely import Polygon
from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO, ARS, CrossQ, TQC


# ================================
# Seed / reproducibility
# ================================

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ================================
# Geometry helpers
# ================================

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
    """Scale polygon vertices about their centroid by *scale* (0 < scale ≤ 1 shrinks)."""
    pts = np.asarray(poly_pts, dtype=np.float64)
    centroid = np.mean(pts, axis=0)
    scaled = centroid + scale * (pts - centroid)
    return [tuple(p) for p in scaled.tolist()]


# ================================
# Encoding helpers
# ================================

def binary_list_to_decimal(bin_list):
    """Convert a list of 0/1 values to its decimal integer equivalent."""
    return int("".join(str(int(b)) for b in bin_list), 2)


# ================================
# Config loaders
# ================================

def read_uav_json(json_path, sf=1):
    """
    Load UAV field-info dicts from JSON, applying a scale factor sf.

    The JSON stores coordinates in a compact 0-100 range and the default
    sf=1 keeps them in that 0-100 world-unit space used by MultiUAV.
    Pass a different sf only if you need to rescale the world.

    Expected JSON format per set:
      {
        "field":             [[x, y], ...],
        "init_positions":    [[x, y], ...],
        "infected_locations":[[x, y], ...]   ← 2-D tuples only (no infection level)
      }
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    for set_name, cfg in data.items():
        cfg["field"]              = [tuple((p[0] * sf, p[1] * sf)) for p in cfg["field"]]
        cfg["init_positions"]     = [np.array(p, dtype=float) * sf for p in cfg["init_positions"]]
        cfg["infected_locations"] = [tuple((p[0] * sf, p[1] * sf)) for p in cfg["infected_locations"]]
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

    The ``num_robots`` field in each env records the *maximum* number of
    robot starting positions available.  MultiWheeled accepts a ``num_robots``
    override at construction time to use only the first N of them, so the
    same JSON supports 2-, 3-, 4-, and 5-robot experiments.

    Obstacles in env2–env9 are shrunk to 30 % of their original area about
    their centroid so that robots can traverse those more cluttered maps.
    """
    # Cluttered maps (env2-env9) benefit from reduced obstacle footprints
    # so the policy can discover feasible paths during early training.
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
                obs = [scale_polygon_about_centroid(poly, obstacle_scale_env2_9)
                       for poly in obs]
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
        print(f"  Loaded wheeled config set{i} <- {key}  "
              f"({env_params['NUM_ROBOTS']} robots, "
              f"{int(env_params['SCREEN_WIDTH'])}x{int(env_params['SCREEN_HEIGHT'])}")

    return configs


def read_wheeled_configs(config_dir):
    """
    DEPRECATED — scans a directory for .ini files.

    Kept for backward compatibility only.  New code should call
    ``read_wheeled_json(json_path)`` with the consolidated JSON file at
    ``exp_sets/wheeled/wheeled_configs.json``.
    """
    import warnings
    warnings.warn(
        "read_wheeled_configs() is deprecated.  "
        "Use read_wheeled_json('exp_sets/wheeled/wheeled_configs.json') instead.",
        DeprecationWarning, stacklevel=2,
    )
    ini_files = sorted(
                (f for f in os.listdir(config_dir) if f.endswith(".ini")),
                key=lambda f: int(re.search(r'\d+', f).group())
                )
    if not ini_files:
        raise FileNotFoundError(f"No .ini files found in '{config_dir}'")
    configs = {}
    for i, fname in enumerate(ini_files, start=1):
        configs[f"set{i}"] = _read_ini_env_config(os.path.join(config_dir, fname))
        print(f"  Loaded wheeled config set{i} <- {fname}")
    return configs


def _read_ini_env_config(config_path):
    """
    Internal: parse a single legacy .ini environment config file.
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    env_params = {}

    env_params['SCREEN_WIDTH']  = float(config['SCREEN']['WIDTH'])
    env_params['SCREEN_HEIGHT'] = float(config['SCREEN']['HEIGHT'])
    env_params['ROBOT_LENGTH']  = float(config['ROBOTS']['LENGTH'])
    env_params['ROBOT_WIDTH']   = float(config['ROBOTS']['WIDTH'])
    env_params['MAX_SPEED']     = float(config['ROBOTS']['MAX_SPEED'])
    env_params['MAX_STEER']     = float(config['ROBOTS']['MAX_STEER'])
    env_params['NUM_ROBOTS']    = int(config['ROBOTS']['NUM_ROBOTS'])

    env_params['ROBOT_INIT_CONFIGS'] = []
    for i in range(env_params['NUM_ROBOTS']):
        conf = config['ROBOTS'][f'ROBOT_{i + 1}']
        x, y, theta = map(float, conf.split(','))
        env_params['ROBOT_INIT_CONFIGS'].append((x, y, float(np.radians(theta))))

    env_params['NUM_GOALS']      = int(config['GOALS']['NUM_GOALS'])
    env_params['GOAL_SIZE']      = float(config['GOALS']['GOAL_SIZE'])
    env_params['GOAL_POSITIONS'] = []
    for i in range(env_params['NUM_GOALS']):
        g_pos = config['GOALS'][f'GOAL_{i + 1}']
        x, y  = map(float, g_pos.split(','))
        env_params['GOAL_POSITIONS'].append((x, y))

    env_params['NUM_OBSTACLES'] = int(config['OBSTACLES']['NUM_OBSTACLES'])
    env_params['OBSTACLES']     = []
    for i in range(env_params['NUM_OBSTACLES']):
        ver_str = config['OBSTACLES'][f'OBSTACLE_{i + 1}']
        points  = [tuple(map(int, pt.split(','))) for pt in ver_str.split(';')]
        env_params['OBSTACLES'].append(points)

    return env_params


# Backward-compatible alias for old scripts still calling read_env_config
read_env_config = _read_ini_env_config


# Backward-compatible alias for scripts that still call load_experiment_dict_json
def load_experiment_dict_json(json_path):
    """Alias for read_uav_json (sf=1). Kept for backward compatibility."""
    return read_uav_json(json_path, sf=1)


# ================================
# Model loading
# ================================

_ALG_CLASSES = {
    "A2C":    A2C,
    "PPO":    PPO,
    "TRPO":   TRPO,
    "ARS":    ARS,
    "CrossQ": CrossQ,
    "TQC":    TQC,
}


def load_model(algorithm, model_path, device="cpu"):
    """Load a saved SB3 / sb3-contrib model.

    Parameters
    ----------
    algorithm : str
        One of 'A2C', 'PPO', 'TRPO', 'ARS', 'CrossQ', 'TQC'.
    model_path : str
        Path to the saved zip (with or without the .zip extension).
    device : str
        Torch device ('cpu' or 'cuda').
    """
    if algorithm not in _ALG_CLASSES:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Choose from "
            f"{sorted(_ALG_CLASSES)}."
        )
    return _ALG_CLASSES[algorithm].load(model_path, device=device)


# ================================
# Misc helpers
# ================================

_TRUE_STRINGS  = {"y", "yes", "t", "true",  "on",  "1"}
_FALSE_STRINGS = {"n", "no",  "f", "false", "off", "0"}


def parse_bool(string):
    """Parse a string into a bool (replacement for distutils.util.strtobool)."""
    s = str(string).strip().lower()
    if s in _TRUE_STRINGS:
        return True
    if s in _FALSE_STRINGS:
        return False
    raise ValueError(f"Cannot interpret {string!r} as a boolean.")


def filter_args(args, model):
    """Filter out arguments not present in a model's constructor."""
    model_kwargs = inspect.getfullargspec(model).args
    return {k: args[k] for k in args if k in model_kwargs}


# ================================
# Cross-platform file locking (HPC / Linux)
# ================================

try:
    import fcntl as _fcntl
    _HAVE_FCNTL = True
except ImportError:
    _fcntl = None
    _HAVE_FCNTL = False


def flock_exclusive(f):
    """Acquire an exclusive lock when fcntl is available (Unix/HPC)."""
    if _HAVE_FCNTL:
        _fcntl.flock(f, _fcntl.LOCK_EX)


def flock_unlock(f):
    """Release a file lock acquired with flock_exclusive."""
    if _HAVE_FCNTL:
        _fcntl.flock(f, _fcntl.LOCK_UN)


# ================================
# STDOUT / STDERR redirect
# ================================

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()
