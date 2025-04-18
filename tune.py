import optuna
import os
import argparse
import inspect
import yaml
import gc
from datetime import datetime
from stable_baselines3 import A2C, PPO, DQN
from sb3_contrib import TRPO, ARS, RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import LogEveryNTimesteps
from src.utils import load_experiment

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithm', type=str, required=True, choices=['A2C', 'PPO', 'TRPO', 'DQN', 'ARS', 'RecurrentPPO'], help='The DRL algorithm to use')
    parser.add_argument('--set', required=True, type=int, help='The experiment set to use, from the sets defined in the experiments directory')
    parser.add_argument('--trials', type=int, default=20, help='The number of trials used for tuning')
    parser.add_argument('--steps', type=int, default=1_000_000, help='The amount of steps to train the DRL model for while tuning')
    parser.add_argument('--num_envs', type=int, default=4, help='The number of parallel environments to run')
    parser.add_argument('--num_eval_eps', type=int, default=10, help='The number of episodes for evaluating a trial')
    parser.add_argument('--seed', type=int, default=None, help='The random seed to use')
    parser.add_argument('--log_steps', type=int, default=2000, help='The number of steps between each log entry')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='The device to tune on')

    args = parser.parse_args()
    print(args)

    os.makedirs('tuning_logs', exist_ok=True)
    os.makedirs('tuned_models', exist_ok=True)
    
    mean_rewards = []
    best_reward = -1e10

    # Objective function for optimization
    def objective(trial):
        global best_reward

        # Configure environment
        env_config = load_experiment(f'experiments/set{args.set}.yaml')
        vec_env = make_vec_env('ThreeAgentGridworld-v1', env_kwargs={'env_config': env_config, 'seed': args.seed}, n_envs=args.num_envs)
        vec_env.seed(seed=args.seed)
        vec_env.action_space.seed(seed=args.seed)

        # Base model args
        model_args = {
            'policy': 'MlpLstmPolicy' if args.algorithm == 'RecurrentPPO' else 'MlpPolicy',
            'env': vec_env,
            'tensorboard_log': './tuning_logs',
            'seed': args.seed,
            'device': args.device,
            'n_steps': trial.suggest_categorical("n_steps", [5, 10, 20]),
            'gamma': trial.suggest_float("gamma", 0.90, 0.99),
            'learning_rate': trial.suggest_float("learning_rate", 0.0001, 0.02, log=True),
            'ent_coef': trial.suggest_float("ent_coef", 0.001, 0.05),
            'gae_lambda': trial.suggest_float("gae_lambda", 0.9, 1.0),
            'max_grad_norm': trial.suggest_float("max_grad_norm", 0.30, 0.99),
            'vf_coef': trial.suggest_float("vf_coef", 0.2, 0.7),
        }
        
        # Configure model
        if args.algorithm == 'A2C':
            model = A2C
        elif args.algorithm == 'PPO':
            model = PPO
        elif args.algorithm == 'TRPO':
            model = TRPO
        elif args.algorithm == 'DQN':
            model = DQN
        elif args.algorithm == 'ARS':
            model = ARS
        else:
            model = RecurrentPPO

        model_kwargs = inspect.getfullargspec(model).args
        filtered_args = {k:model_args[k] for k in model_args if k in model_kwargs}
        model = model(**filtered_args)

        # Train model
        logger = LogEveryNTimesteps(n_steps=args.log_steps)
        model.learn(total_timesteps=args.steps, callback=logger, log_interval=None, tb_log_name=f"{args.algorithm}_set{args.set}_{trial.number}")
        vec_env.reset()

        # Evaluate model performance
        mean_reward, _ = evaluate_policy(model, vec_env, n_eval_episodes=args.num_eval_eps, deterministic=True)
        mean_rewards.append(mean_reward)

        if best_reward < mean_reward:
            best_reward = mean_reward
            model.save(f'tuned_models/{args.algorithm}_set{args.set}.zip')

        if args.device == 'cpu':
            del model
            gc.collect()
        else:
            del model
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            
        return mean_reward

    # Optuna study
    start_time = datetime.now()
    print(f'Tuning started on {start_time.ctime()}')
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    end_time = datetime.now()
    print(f'Tuning ended on {end_time.ctime()}')
    print(f'Tuning lasted {end_time - start_time}\n')

    # Best hyperparameters
    os.makedirs('tuned_hyperparameters', exist_ok=True)
    with open(f'tuned_hyperparameters/{args.algorithm}_set{args.set}.yaml', 'w') as save_file:
        yaml.dump(study.best_params, save_file)
    print(f"Best hyperparameters for {args.algorithm}: {study.best_params}")
    print(f"Best mean reward for {args.algorithm}: {best_reward}")
