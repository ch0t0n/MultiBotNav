import os
import argparse
from datetime import datetime
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import LogEveryNTimesteps
from stable_baselines3.common.logger import configure
from src.utils import read_env_config, load_model, parse_bool, filter_args, load_experiment_dict_json

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithm', type=str, required=True, choices=['A2C', 'PPO', 'TRPO', 'TQC', 'ARS', 'CrossQ'], help='The DRL algorithm to use')
    parser.add_argument('--load_set', required=True, type=int, help='The experiment set to load, from the sets defined in the experiments directory')
    parser.add_argument('--train_set', required=True, type=int, help='The experiment set to train on, from the sets defined in the experiments directory. Must be different from load_set for transfer learning')
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=0, help='The verbosity level: 0 no output, 1 info, 2 debug')
    parser.add_argument('--steps', type=int, default=2_000_000, help='The amount of steps to train the DRL model for while tuning')
    parser.add_argument('--num_envs', type=int, default=4, help='The number of parallel environments to run')
    parser.add_argument('--seed', type=int, default=None, help='The random seed to use')
    parser.add_argument('--log_steps', type=int, default=2000, help='The number of steps between each log entry')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='The device to tune on')

    args = parser.parse_args()
    print(args)

    # Make sure load_set and train_set are different
    if args.load_set == args.train_set:
        raise ValueError('load_set and train_set must be different for transfer learning')

    # Experiments file
    json_path  = 'src/new_2026_cont_sets.json'
    json_dict  = load_experiment_dict_json(json_path)
    num_robots = len(json_dict[f"set{args.train_set}"]['init_positions'])
    
    # Necessary paths
    time       = datetime.now().strftime("%B%d_%H") # Get experiment time 
    run_name   = f"{args.algorithm}_from{args.load_set}_to{args.train_set}_seed{args.seed}_v0"
    transfer_log_path = os.path.join("logs", "transfer_logs", run_name)
    os.makedirs(transfer_log_path, exist_ok=True) # Make the log directory

    # from train.py 
    trained_model_run_name = f"{args.algorithm}_set{args.load_set}_seed{args.seed}_v0" 
    trained_model_path     = os.path.join("logs", "training_default_logs", trained_model_run_name, "checkpoints", "trained_model.zip")

    # Configure the logger to save to log, csv, and json files
    logger1 = LogEveryNTimesteps(n_steps=args.log_steps) # Define the logger
    logger2 = configure(os.path.join(transfer_log_path, "tensorboard"), ["stdout", "log", "csv", "json", "tensorboard"])

    vec_env = make_vec_env(
        'MultiRobotEnv-v0',
        env_kwargs={'field_info': json_dict[f"set{args.train_set}"], 'render_mode': None, 'num_robots': num_robots},
        n_envs=args.num_envs,
        # seed=args.seed
    )
    vec_env.seed(seed=args.seed)
    vec_env.action_space.seed(seed=args.seed)

    model = load_model(args.algorithm, 
                       args.load_set, 
                       args.seed, 
                       args.device,  
                       args.verbose, 
                       os.path.join(transfer_log_path, "tensorboard"),
                       trained_model_path,
                       run_name)
    
    # loaded model's spaces match the new env's  spaces
    model.observation_space = vec_env.observation_space
    model.action_space = vec_env.action_space
    if hasattr(model, "policy"):
        model.policy.observation_space = vec_env.observation_space
        model.policy.action_space = vec_env.action_space

    model.set_env(vec_env)

    # Train model
    start_time = datetime.now()
    print(f'Transfer learning started on {start_time.ctime()}')
    # logger = LogEveryNTimesteps(n_steps=args.log_steps)
    # model.learn(total_timesteps=args.steps, callback=logger, log_interval=None, tb_log_name=f"{args.algorithm}_from{args.load_set}_to{args.train_set}")
    # ???
    # callback=logger1, 
    model.set_logger(logger2)
    model.learn(total_timesteps=args.steps, callback=logger1, log_interval=None, tb_log_name=run_name, reset_num_timesteps=True)

    end_time = datetime.now()
    print(f'Transfer learning ended on {end_time.ctime()}')
    print(f'Transfer learning lasted {end_time - start_time}')
    
    # Save model
    checkpoints_path = os.path.join(transfer_log_path, "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)
    model.save(os.path.join(checkpoints_path, "trained_model.zip"))

    vec_env.close()
