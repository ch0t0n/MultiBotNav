import os
import argparse
from datetime import datetime
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import LogEveryNTimesteps
from src.utils import load_experiment, load_model

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithm', type=str, required=True, choices=['A2C', 'PPO', 'TRPO', 'DQN', 'ARS', 'RecurrentPPO'], help='The DRL algorithm to use')
    parser.add_argument('--load_set', required=True, type=int, help='The experiment set to load, from the sets defined in the experiments directory')
    parser.add_argument('--train_set', required=True, type=int, help='The experiment set to train on, from the sets defined in the experiments directory. Must be different from load_set for transfer learning')
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=0, help='The verbosity level: 0 no output, 1 info, 2 debug')
    parser.add_argument('--steps', type=int, default=1_000_000, help='The amount of steps to train the DRL model for while tuning')
    parser.add_argument('--num_envs', type=int, default=4, help='The number of parallel environments to run')
    parser.add_argument('--seed', type=int, default=None, help='The random seed to use')
    parser.add_argument('--log_steps', type=int, default=2000, help='The number of steps between each log entry')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='The device to tune on')

    args = parser.parse_args()
    print(args)

    # Make sure load_set and train_set are different
    if args.load_set == args.train_set:
        raise ValueError('load_set and train_set must be different for transfer learning')

    # Configure environment
    env_config = load_experiment(f'experiments/set{args.train_set}.yaml')
    vec_env = make_vec_env('ThreeAgentGridworld-v1', env_kwargs={'env_config': env_config, 'seed': args.seed}, n_envs=args.num_envs)
    vec_env.seed(seed=args.seed)
    vec_env.action_space.seed(seed=args.seed)

    os.makedirs('transfer_logs', exist_ok=True)

    model = load_model(args.algorithm, args.load_set, args.seed, args.device, 'tuned_models', args.verbose, 'transfer_logs')
    model.set_env(vec_env)

    # Train model
    start_time = datetime.now()
    print(f'Transfer learning started on {start_time.ctime()}')
    logger = LogEveryNTimesteps(n_steps=args.log_steps)
    model.learn(total_timesteps=args.steps, callback=logger, log_interval=None, tb_log_name=f"{args.algorithm}_from{args.load_set}_to{args.train_set}")
    end_time = datetime.now()
    print(f'Transfer learning ended on {end_time.ctime()}')
    print(f'Transfer learning lasted {end_time - start_time}')
    
    # Save model
    os.makedirs('transfer_models', exist_ok=True)
    model.save(f'transfer_models/{args.algorithm}_from{args.load_set}_to{args.train_set}.zip')

    vec_env.close()
