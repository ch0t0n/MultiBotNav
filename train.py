import os
import argparse
from datetime import datetime
from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO, ARS, CrossQ, TQC
from stable_baselines3.common.env_util import make_vec_env
from src.utils import load_experiment, load_model, parse_bool

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, required=True, choices=['A2C', 'PPO', 'TRPO', 'ARS', 'CrossQ', 'TQC'], help='The DRL algorithm to use')
    parser.add_argument('--set', required=True, type=int, help='The experiment set to use, from the sets defined in the experiments directory')
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=0, help='The verbosity level: 0 no output, 1 info, 2 debug')
    parser.add_argument('--gamma', type=float, default=0.99, help='The discount factor for the DRL algorithm')
    parser.add_argument('--steps', type=int, default=5_000_000, help='The amount of steps to train the DRL model for')
    parser.add_argument('--num_envs', type=int, default=1, help='The number of parallel environments to run')
    parser.add_argument('--resume', type=parse_bool, default=False, help='If true, loads an existing model to resume training. If false, trains a new model')
    
    args = parser.parse_args()
    print(f'Algorithm: {args.algorithm}\nSet: {args.set}\nGamma: {args.gamma}\nTraining steps: {args.steps}\n')
    
    # Configure environment
    env_config = load_experiment(path=f'experiments/set{args.set}.yaml', sf=10)
    vec_env = make_vec_env('MultiBotNavigator-v0', env_kwargs={'env_config': env_config}, n_envs=args.num_envs)
    
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Configure model
    if args.resume:
        model = load_model(args.algorithm, args.set)
        model.set_env(vec_env)
    else:
        if args.algorithm == 'A2C':
            model = A2C("MlpPolicy", vec_env, verbose=args.verbose, tensorboard_log="./logs", gamma=args.gamma)
        elif args.algorithm == 'PPO':
            model = PPO("MlpPolicy", vec_env, verbose=args.verbose, tensorboard_log="./logs", gamma=args.gamma)
        elif args.algorithm == 'TRPO':
            model = TRPO("MlpPolicy", vec_env, verbose=args.verbose, tensorboard_log="./logs", gamma=args.gamma)
        elif args.algorithm == 'ARS':
            model = ARS("LinearPolicy", vec_env, verbose=args.verbose, tensorboard_log="./logs")
        elif args.algorithm == 'CrossQ':
            model = CrossQ("MlpPolicy", vec_env, verbose=args.verbose, tensorboard_log="./logs", gamma=args.gamma)
        elif args.algorithm == 'TQC':
            model = TQC("MlpPolicy", vec_env, verbose=args.verbose, tensorboard_log="./logs", gamma=args.gamma)

    # Train model
    start_time = datetime.now()
    print(f'Training started on {start_time.ctime()}')
    model.learn(total_timesteps=args.steps, tb_log_name=f"{args.algorithm}_set{args.set}", reset_num_timesteps=False)
    end_time = datetime.now()
    print(f'Training ended on {end_time.ctime()}')
    print(f'Training lasted {end_time - start_time}')
    
    # Save model
    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')
    model.save(f'trained_models/{args.algorithm}_set{args.set}.zip')
    vec_env.close()
