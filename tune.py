import os
import optuna
import argparse
import yaml
import gc
from datetime import datetime
import gymnasium as gym

from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO, ARS, TQC, CrossQ
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import LogEveryNTimesteps

from src.utils import (
    filter_args,
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

    parser.add_argument('--set', required=True, type=int)

    parser.add_argument('--num_robots', type=int, default=3)

    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--num_eval_eps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--log_steps', type=int, default=2000)
    parser.add_argument('--device', type=str,
                        choices=['cpu', 'cuda'],
                        default='cpu')

    args = parser.parse_args()
    print(args)

    # ------------------------------------------------------------------
    # Logging structure
    # ------------------------------------------------------------------
    log_base = os.path.join("logs", "tuning_logs")
    os.makedirs(log_base, exist_ok=True)

    model_dict = {
        'A2C': A2C,
        'PPO': PPO,
        'TRPO': TRPO,
        'TQC': TQC,
        'ARS': ARS,
        'CrossQ': CrossQ
    }

    model_type = model_dict[args.algorithm]

    best_reward = -1e10

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def objective(trial):
        nonlocal best_reward

        # Environment creation
        if args.robot_type == "wheeled_robot":
            env_config = read_wheeled_config(
                f'exp_sets/wheeled/env{args.set}.ini'
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

        model_args = {
            'policy': 'LinearPolicy' if args.algorithm == 'ARS' else 'MlpPolicy',
            'env': vec_env,
            'tensorboard_log': log_base,
            'seed': args.seed,
            'device': args.device,
            'n_steps': trial.suggest_categorical(
                "n_steps",
                [5, 10, 20] if args.algorithm == 'A2C' else [1024, 2048, 4096]
            ),
            'gamma': trial.suggest_float("gamma", 0.90, 0.99),
            'learning_rate': trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
            'ent_coef': trial.suggest_float("ent_coef", 0.0, 0.05),
            'gae_lambda': trial.suggest_float("gae_lambda", 0.9, 1.0),
            'max_grad_norm': trial.suggest_float("max_grad_norm", 0.3, 0.99),
            'vf_coef': trial.suggest_float("vf_coef", 0.2, 0.7),
            'buffer_size': trial.suggest_int("buffer_size", 1000, 100000, step=1000)
        }

        filtered_args = filter_args(model_args, model_type)
        model = model_type(**filtered_args)

        logger = LogEveryNTimesteps(n_steps=args.log_steps)

        model.learn(
            total_timesteps=args.steps,
            callback=logger,
            log_interval=None,
            tb_log_name=f"{args.robot_type}_{args.algorithm}_set{args.set}_trial{trial.number}"
        )

        mean_reward, _ = evaluate_policy(
            model,
            vec_env,
            n_eval_episodes=args.num_eval_eps,
            deterministic=True
        )

        if mean_reward > best_reward:
            best_reward = mean_reward
            save_path = os.path.join(
                log_base,
                f"{args.robot_type}_{args.algorithm}_set{args.set}_best_model.zip"
            )
            model.save(save_path)

        del model
        gc.collect()

        if args.device == "cuda":
            import torch
            torch.cuda.empty_cache()

        vec_env.close()

        return mean_reward

    # ------------------------------------------------------------------
    # Run Study
    # ------------------------------------------------------------------
    start_time = datetime.now()
    print(f"Tuning started on {start_time.ctime()}")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    end_time = datetime.now()
    print(f"Tuning ended on {end_time.ctime()}")
    print(f"Tuning lasted {end_time - start_time}")

    # Save best hyperparameters
    best_params = filter_args(study.best_params, model_type)

    hyperparam_dir = os.path.join(log_base, "best_hyperparameters")
    os.makedirs(hyperparam_dir, exist_ok=True)

    with open(
        os.path.join(
            hyperparam_dir,
            f"{args.robot_type}_{args.algorithm}_set{args.set}.yaml"
        ),
        "w"
    ) as f:
        yaml.dump(best_params, f)

    print("Best hyperparameters:", best_params)
    print("Best reward:", best_reward)