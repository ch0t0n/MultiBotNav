from __future__ import annotations
import json, os, socket, sys, traceback
from datetime import datetime
from typing import Any, Dict
import yaml
import numpy as np
import distutils
import itertools
import inspect
from shapely import Polygon
import configparser
import glob
import re
# from shapely.geometry import Polygon  # type: ignore
# import argparse


def set_global_seeds(seed: int) -> None:
    if seed is None:
        return

    import os
    import random
    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def get_policy_type(algorithm):
    # policy type
    if algorithm == "ARS":
        policy = "LinearPolicy"
    elif algorithm == "RPPO":
        policy = "MlpLstmPolicy" # RecurrentPPO policies are LSTM-based
    else:
        policy = "MlpPolicy"
    return policy

def get_model_type(algorithm):
    # model type
    if algorithm == 'A2C':
        from stable_baselines3 import A2C
        model_type = A2C
    elif algorithm == 'PPO':
        from stable_baselines3 import PPO
        model_type = PPO
    elif algorithm == 'TRPO':
        from sb3_contrib import TRPO
        model_type = TRPO
    elif algorithm == 'TQC':
        from sb3_contrib import TQC
        model_type = TQC
    elif algorithm == 'ARS':
        from sb3_contrib import ARS
        model_type = ARS
    elif algorithm == 'RPPO':
        from sb3_contrib import RecurrentPPO
        model_type = RecurrentPPO
    elif algorithm == 'CrossQ':
        from sb3_contrib import CrossQ
        model_type = CrossQ
    else:
        raise ValueError(f"Unsupported algorithm for resuming: {algorithm}")
    
    return model_type

def setup_trial_dirs(trials_dir: str, trial_number: int):
    trial_dir = os.path.join(trials_dir, f"trial_{trial_number:03d}")
    tensorboard_dir = os.path.join(trial_dir, "tensorboard")
    monitor_dir = os.path.join(trial_dir, "monitor")

    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(monitor_dir, exist_ok=True)

    return trial_dir, tensorboard_dir, monitor_dir

def setup_log_dirs(run_name, log_type, args):
    """
    logs/training_{best|default}_logs/<run_name>/
      ├─ tensorboard/
      ├─ checkpoints/
      ├─ monitor/
      └─ run_config.yaml
    """
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Unique run directory
    base_dir = os.path.join("logs", log_type)
    os.makedirs(base_dir, exist_ok=True)
    log_dir = os.path.join(base_dir, run_name)
    # If already exists, back up old logs (change name)
    if os.path.exists(log_dir):
        resume = getattr(args, "resume", False)
        if resume:
            pass # Resume into existing directory
        else:
            backup = f"{log_dir}__old_{now}" 
            os.rename(log_dir, backup)
            os.makedirs(log_dir, exist_ok=False) # new dir
    else:
        os.makedirs(log_dir, exist_ok=False)
    # Sub-dirs
    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    checkpoints_dir = os.path.join(log_dir, "checkpoints")
    monitor_dir = os.path.join(log_dir, "monitor")
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(monitor_dir, exist_ok=True)
    # Save run config 
    run_config: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "run_name": run_name,
        "log_type": log_type,
        "log_path": log_dir,
        "tensorboard_dir": tensorboard_dir,
        "checkpoints_dir": checkpoints_dir,
        "monitor_dir": monitor_dir,
    }
    with open(os.path.join(log_dir, "run_config.yaml"), "w") as f:
        yaml.safe_dump(run_config, f, sort_keys=False)

    return log_dir, tensorboard_dir, checkpoints_dir, monitor_dir, run_config

def setup_optuna_study_dirs(study_name: str, args, resume: bool = False):
    """
    logs/tuning_logs/<study_name>/
      ├─ study_config.yaml
      ├─ best_model.zip
      ├─ trials/
      │   ├─ trial_000/
      │   │   ├─ tensorboard/
      │   │   ├─ monitor/
      │   │   └─ trial_config.yaml
      │   └─ ...
      └─ tuned_hyperparameters/
    """
    study_dir = os.path.join("logs", "tuning_logs", study_name)
    if os.path.exists(study_dir) and not resume:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = f"{study_dir}__old_{ts}"
        os.rename(study_dir, backup)

    os.makedirs(study_dir, exist_ok=True)
    trials_dir = os.path.join(study_dir, "trials")
    tuned_params_dir = os.path.join(study_dir, "tuned_hyperparameters")
    os.makedirs(trials_dir, exist_ok=True)
    os.makedirs(tuned_params_dir, exist_ok=True)
    best_model_path = os.path.join(study_dir, "best_model.zip")
    best_params_path = os.path.join(study_dir, "best_hyperparameters.yaml")

    # Save a study-level config file
    study_config: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "study_name": study_name,
        "algorithm": args.algorithm,
        "set": args.set,
        "seed": args.seed,
        "device": args.device,
        "trials": args.trials,
        "steps_per_trial": args.steps,
        "num_envs": args.num_envs,
        "num_eval_eps": args.num_eval_eps,
    }
    use_wandb = getattr(args, "use_wandb", False)
    if use_wandb:
        study_config["wandb"] = {
            "use_wandb": use_wandb,
            "project": args.wandb_project_name,
            "group": args.wandb_group,
            "tags": args.wandb_tags,
            "wandb_each_trial": args.wandb_each_trial,
        }
    with open(os.path.join(study_dir, "study_config.yaml"), "w") as f:
        yaml.safe_dump(study_config, f, sort_keys=False)

    return study_dir, trials_dir, tuned_params_dir, best_model_path, best_params_path, study_config

def find_latest_checkpoint(
    checkpoints_dir: str,
    *,
    fallback_name: str = "trained_model.zip",
) -> str:
    """Find newest checkpoint. Prefers SB3 CheckpointCallback naming:
       <prefix>_<N>_steps.zip
    Falls back to 'trained_model.zip' then newest mtime.
    """
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoints directory does not exist: {checkpoints_dir}")

    candidates = glob.glob(os.path.join(checkpoints_dir, "*.zip"))
    if not candidates:
        raise FileNotFoundError(f"No .zip checkpoints found in: {checkpoints_dir}")

    step_candidates = []
    for p in candidates:
        base = os.path.basename(p)
        m = re.search(r"_(\d+)_steps\.zip$", base)
        if m:
            step_candidates.append((int(m.group(1)), p))

    if step_candidates:
        step_candidates.sort(key=lambda t: t[0])
        return step_candidates[-1][1]

    fallback = os.path.join(checkpoints_dir, fallback_name)
    if os.path.exists(fallback):
        return fallback

    return max(candidates, key=os.path.getmtime)

def _slurm_env() -> dict:
    keys = [
        "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NAME", "SLURM_NODELIST", "SLURMD_NODENAME",
        "SLURM_PROCID", "SLURM_LOCALID",
    ]
    return {k: os.environ.get(k) for k in keys if os.environ.get(k) is not None}

def append_failure(
    *,
    log_path: str = "failed_runs.jsonl",
    scheme: str,
    script: str,
    run_name: str | None = None,
    extra: dict | None = None,
) -> None:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    payload = {
        "ts": datetime.now().isoformat(),
        "host": socket.gethostname(),
        "scheme": scheme,     # "tune" | "train" | "transfer" | "train_from_tuned"
        "script": script,     # "tune_v2.py", etc.
        "run_name": run_name, # your naming string if available
        "argv": sys.argv,
        "slurm": _slurm_env(),
        "exc_type": None,
        "exc": None,
        "traceback": None,
    }
    if extra:
        payload.update(extra)

    et, ev, tb = sys.exc_info()
    if et is not None:
        payload["exc_type"] = getattr(et, "__name__", str(et))
        payload["exc"] = str(ev)
        payload["traceback"] = "".join(traceback.format_exception(et, ev, tb))

    line = json.dumps(payload, sort_keys=False)

    # Atomic-ish append with advisory lock (Linux clusters)
    try:
        import fcntl
        with open(log_path, "a", buffering=1) as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception:
        # Last-resort: try plain append (still line-buffered)
        try:
            with open(log_path, "a", buffering=1) as f:
                f.write(line + "\n")
                f.flush()
        except Exception:
            pass



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
# def parse_bool(v):
#     """Argparse-friendly bool parser. Safe on Python >=3.12 (no distutils)."""
#     if isinstance(v, bool):
#         return v
#     if v is None:
#         return False
#     s = str(v).strip().lower()
#     if s in {"1", "true", "t", "yes", "y", "on"}:
#         return True
#     if s in {"0", "false", "f", "no", "n", "off"}:
#         return False
#     raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")
def parse_bool(string):
    return bool(distutils.util.strtobool(string))

# Loads in a trained model
def load_model(algorithm, 
               load_set, 
               seed, 
               device, 
               verbose, 
               tensorboard_dir, 
               trained_model_path,
               run_name
):
    model_type = get_model_type(algorithm)
    policy = get_policy_type(algorithm)
    model_args = {
        #"policy": policy,
        'path': trained_model_path, # f'{models_dir}/{algorithm}_set{experiment_set}.zip',
        #'tb_log_name': run_name, # f'{algorithm}_set{experiment_set}',
        'device': device,
        'seed': seed,
        'verbose': verbose,
        'tensorboard_log': tensorboard_dir,
    }
    model = model_type.load(**model_args)
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
