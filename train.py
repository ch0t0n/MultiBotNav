import os
import argparse
import yaml
from datetime import datetime
from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO, ARS, CrossQ, TQC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import LogEveryNTimesteps
from stable_baselines3.common.logger import configure
from src.utils import read_env_config, load_model, parse_bool, filter_args, load_experiment_dict_json

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithm', type=str, required=True, choices=['A2C', 'PPO', 'TRPO', 'ARS', 'CrossQ', 'TQC'], help='The DRL algorithm to use')
    parser.add_argument('--set', required=True, type=int, help='The experiment set to use, from the sets defined in the experiments directory')
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=0, help='The verbosity level: 0 no output, 1 info, 2 debug')
    parser.add_argument('--steps', type=int, default=2_000_000, help='The amount of steps to train the DRL model for')
    parser.add_argument('--num_envs', type=int, default=4, help='The number of parallel environments to run')
    parser.add_argument('--seed', type=int, default=None, help='The random seed to use')
    parser.add_argument('--log_steps', type=int, default=2000, help='The number of steps between each log entry')
    parser.add_argument('--resume', type=parse_bool, default=False, help='If true, loads an existing model to resume training. If false, trains a new model')
    parser.add_argument('--use_tuned_params', type=parse_bool, default=False, help='If true, uses tuned hyperparameters. If false, uses default hyperparameters')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='The device to train on')
    
    args = parser.parse_args()
    print(args)

    # Experiments file
    json_path  = 'src/new_2026_cont_sets.json'
    json_dict  = load_experiment_dict_json(json_path)
    num_robots = len(json_dict[f"set{args.set}"]['init_positions'])
    
    # Necessary paths
    time       = datetime.now().strftime("%B%d_%H") # Get experiment time 
    log_type   = "training_best_logs" if args.use_tuned_params else "training_default_logs"
    run_name   = f"{args.algorithm}_set{args.set}_seed{args.seed}_v0" 
    log_path   = os.path.join("logs", log_type, run_name)
    os.makedirs(log_path, exist_ok=True) # Make the log directory

    # Configure the logger to save to log, csv, and json files
    logger1 = LogEveryNTimesteps(n_steps=args.log_steps) # Define the logger
    logger2 = configure(os.path.join(log_path, "tensorboard"), ["stdout", "log", "csv", "json", "tensorboard"])

    vec_env = make_vec_env(
        'MultiRobotEnv-v0',
        env_kwargs={'field_info': json_dict[f"set{args.set}"], 'render_mode': None, 'num_robots': num_robots},
        n_envs=args.num_envs,
        # seed=args.seed
    )
    vec_env.seed(seed=args.seed)
    vec_env.action_space.seed(seed=args.seed)
    
    # Configure model
    if args.resume:
        # TODO: fix
        model = load_model(args.algorithm, args.set, args.seed, args.device, args.verbose, log_path, log_type, run_name)
        model.set_env(vec_env)
    else:
        if args.algorithm == 'A2C':
            model_type = A2C
        elif args.algorithm == 'PPO':
            model_type = PPO
        elif args.algorithm == 'TRPO':
            model_type = TRPO
        elif args.algorithm == 'TQC':
            model_type = TQC
        elif args.algorithm == 'ARS':
            model_type = ARS
        else:
            model_type = CrossQ

        if args.use_tuned_params:
            with open(f'logs/tuning_logs/{args.algorithm}_set{args.set}_seed{args.seed}_v0/best_hyperparameters.yaml') as file:
                try:
                    hyperparameters = filter_args(yaml.safe_load(file), model_type)
                except yaml.YAMLError as err:
                    print(err)
                    hyperparameters = {}
        else:
            hyperparameters = {}

        model_args = {
            'policy': 'LinearPolicy' if args.algorithm == 'ARS' else 'MlpPolicy',
            'env': vec_env,
            'verbose': args.verbose,
            'tensorboard_log': os.path.join(log_path, "tensorboard"),
            'seed': args.seed,
            'device': args.device,
            **hyperparameters
        }
        model = model_type(**model_args)
    
    # Train model
    start_time = datetime.now()
    print(f'Training started on {start_time.ctime()}')
    # logger = LogEveryNTimesteps(n_steps=args.log_steps)
    # model.learn(total_timesteps=args.steps, callback=logger, log_interval=None, tb_log_name=f"{args.algorithm}_set{args.set}", reset_num_timesteps=False)
    
    # ???
    # callback=logger1, 
    model.set_logger(logger2)
    model.learn(total_timesteps=args.steps, callback=logger1, log_interval=None, tb_log_name=run_name, reset_num_timesteps=False)

    end_time = datetime.now()
    print(f'Training ended on {end_time.ctime()}')
    print(f'Training lasted {end_time - start_time}')
    
    # Save model
    checkpoints_path = os.path.join(log_path, "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)
    model.save(os.path.join(checkpoints_path, "trained_model.zip"))

    vec_env.close()