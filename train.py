import os
import argparse
import yaml
from datetime import datetime
import gymnasium as gym

from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO, ARS, CrossQ, TQC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import LogEveryNTimesteps
from stable_baselines3.common.logger import configure

from src.utils import (
    parse_bool,
    filter_args,
    load_model,
    read_wheeled_config,
    read_uav_json
)

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Ensure script runs from its own directory (same as run.py)
    # ------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)

    # ------------------------------------------------------------------
    # Parse arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, required=True, choices=['A2C', 'PPO', 'TRPO', 'ARS', 'CrossQ', 'TQC'])
    parser.add_argument('--robot_type', type=str, choices=['uav', 'wheeled_robot'], default='uav')
    parser.add_argument('--set', required=True, type=int, help='Experiment set number')
    parser.add_argument('--num_robots', type=int, default=3, help='Number of robots (only used for UAV)')
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--log_steps', type=int, default=2000)
    parser.add_argument('--resume', type=parse_bool, default=False)
    parser.add_argument('--use_tuned_params', type=parse_bool, default=False)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    print(args)

    # ------------------------------------------------------------------
    # Logging paths
    # ------------------------------------------------------------------
    time = datetime.now().strftime("%B%d_%H")
    log_type = "training_best_logs" if args.use_tuned_params else "training_default_logs"
    run_name = f"{args.robot_type}_{args.algorithm}_set{args.set}_seed{args.seed}_v0"
    log_path = os.path.join("logs", log_type, run_name)
    os.makedirs(log_path, exist_ok=True)

    logger1 = LogEveryNTimesteps(n_steps=args.log_steps)
    logger2 = configure(
        os.path.join(log_path, "tensorboard"),
        ["stdout", "log", "csv", "json", "tensorboard"]
    )

    # ------------------------------------------------------------------
    # Create Environment (Aligned with run.py)
    # ------------------------------------------------------------------
    if args.robot_type == 'wheeled_robot':
        env_config = read_wheeled_config(f'exp_sets/wheeled/env{args.set}.ini')

        vec_env = make_vec_env(
            'MultiWheeled-v0',
            env_kwargs={
                'render_mode': None,
                'env_params': env_config
            },
            n_envs=args.num_envs,
            seed=args.seed
        )

    else:  # UAV
        env_json = read_uav_json(
            r'.\exp_sets\uav\cont_sets.json'
        )[f'set{args.set}']

        vec_env = make_vec_env(
            'MultiUAV-v0',
            env_kwargs={
                'render_mode': None,
                'field_info': env_json,
                'num_robots': args.num_robots
            },
            n_envs=args.num_envs,
            seed=args.seed
        )

    vec_env.action_space.seed(args.seed)

    # ------------------------------------------------------------------
    # Select Model Type
    # ------------------------------------------------------------------
    model_dict = {
        'A2C': A2C,
        'PPO': PPO,
        'TRPO': TRPO,
        'TQC': TQC,
        'ARS': ARS,
        'CrossQ': CrossQ
    }

    model_type = model_dict[args.algorithm]

    # ------------------------------------------------------------------
    # Load tuned hyperparameters (if requested)
    # ------------------------------------------------------------------
    if args.use_tuned_params:
        tuning_path = (
            f'logs/tuning_logs/'
            f'{args.robot_type}_{args.algorithm}_set{args.set}_seed{args.seed}_v0/'
            f'best_hyperparameters.yaml'
        )

        with open(tuning_path) as file:
            try:
                hyperparameters = filter_args(
                    yaml.safe_load(file),
                    model_type
                )
            except yaml.YAMLError as err:
                print(err)
                hyperparameters = {}
    else:
        hyperparameters = {}

    # ------------------------------------------------------------------
    # Resume or Create Model
    # ------------------------------------------------------------------
    device = args.device if (args.algorithm.lower()!='crossq') else 'cuda'
    if args.resume:
        model = load_model(
            algorithm=args.algorithm,
            seed=args.seed,
            device=device,
            verbose=args.verbose,
            trained_model_path=os.path.join(log_path, "checkpoints", "trained_model.zip")
        )
        model.set_env(vec_env)
    else:
        model_args = {
            'policy': 'LinearPolicy' if args.algorithm == 'ARS' else 'MlpPolicy',
            'env': vec_env,
            'verbose': args.verbose,
            'tensorboard_log': os.path.join(log_path, "tensorboard"),
            'seed': args.seed,
            'device': device,
            **hyperparameters
        }
        model = model_type(**model_args)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    start_time = datetime.now()
    print(f'Training started on {start_time.ctime()} in device {device}')

    model.set_logger(logger2)

    model.learn(
        total_timesteps=args.steps,
        callback=logger1,
        log_interval=None,
        tb_log_name=run_name,
        reset_num_timesteps=False
    )

    end_time = datetime.now()
    print(f'Training ended on {end_time.ctime()}')
    print(f'Training lasted {end_time - start_time}')

    # ------------------------------------------------------------------
    # Save Model
    # ------------------------------------------------------------------
    checkpoints_path = os.path.join(log_path, "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)

    model.save(os.path.join(checkpoints_path, "trained_model.zip"))

    vec_env.close()