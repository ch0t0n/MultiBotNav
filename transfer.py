import os
import argparse
from datetime import datetime
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import LogEveryNTimesteps
from stable_baselines3.common.logger import configure

from src.utils import (
    load_model,
    read_wheeled_config,
    read_uav_json,
    parse_bool
)

if __name__ == '__main__':

    # ------------------------------------------------------------------
    # Ensure correct working directory
    # ------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)

    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithm', type=str, required=True,
                        choices=['A2C', 'PPO', 'TRPO', 'TQC', 'ARS', 'CrossQ'])

    parser.add_argument('--robot_type', type=str,
                        choices=['uav', 'wheeled_robot'],
                        default='uav')

    parser.add_argument('--load_set', required=True, type=int)
    parser.add_argument('--train_set', required=True, type=int)

    parser.add_argument('--num_robots', type=int, default=3)

    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--steps', type=int, default=5_000_000)
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--log_steps', type=int, default=2000)
    parser.add_argument('--device', type=str,
                        choices=['cpu', 'cuda'],
                        default='cpu')

    args = parser.parse_args()
    print(args)

    if args.load_set == args.train_set:
        raise ValueError("load_set and train_set must be different.")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_dir = os.path.join(
        "logs",
        "transfer_logs",
        f"{args.robot_type}_{args.algorithm}_from{args.load_set}_to{args.train_set}"
    )
    os.makedirs(log_dir, exist_ok=True)

    logger = configure(
        os.path.join(log_dir, "tensorboard"),
        ["stdout", "log", "csv", "json", "tensorboard"]
    )

    # ------------------------------------------------------------------
    # Create training environment
    # ------------------------------------------------------------------
    if args.robot_type == "wheeled_robot":
        env_config = read_wheeled_config(
            f'exp_sets/wheeled/env{args.train_set}.ini'
        )

        vec_env = make_vec_env(
            'MultiWheeled-v0',
            env_kwargs={
                'render_mode': None,
                'env_params': env_config
            },
            n_envs=args.num_envs,
            seed=args.seed
        )

    else:
        env_json = read_uav_json(
            r'.\exp_sets\uav\icra_2026_cont_sets.json'
        )[f'set{args.train_set}']

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
    # Load model from training logs
    # ------------------------------------------------------------------
    trained_model_path = os.path.join(
        "logs",
        "training_default_logs",
        f"{args.robot_type}_{args.algorithm}_set{args.load_set}_seed{args.seed}_v0",
        "checkpoints",
        "trained_model.zip"
    )

    model = load_model(
        algorithm=args.algorithm,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
        trained_model_path=trained_model_path
    )

    model.set_env(vec_env)
    model.set_logger(logger)

    # ------------------------------------------------------------------
    # Train (Transfer)
    # ------------------------------------------------------------------
    start_time = datetime.now()
    print(f"Transfer started on {start_time.ctime()}")

    callback = LogEveryNTimesteps(n_steps=args.log_steps)

    model.learn(
        total_timesteps=args.steps,
        callback=callback,
        log_interval=None,
        tb_log_name=f"{args.robot_type}_{args.algorithm}_transfer"
    )

    end_time = datetime.now()
    print(f"Transfer ended on {end_time.ctime()}")
    print(f"Transfer lasted {end_time - start_time}")

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    save_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    model.save(os.path.join(save_dir, "transfer_model.zip"))

    vec_env.close()