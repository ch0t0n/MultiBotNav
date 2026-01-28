import os
import argparse
from datetime import datetime
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import LogEveryNTimesteps, CheckpointCallback
from stable_baselines3.common.logger import configure
from src.EpisodeMetricsInfoWrapper import EpisodeMetricsInfoWrapper
from src.utils_v2 import (
    get_policy_type,
    get_model_type,
    load_model,
    parse_bool, 
    load_experiment_dict_json,
    setup_log_dirs,
    set_global_seeds,
    append_failure,
)
from src.wandb_utils import (
    init_wandb,
    get_wandb_callback,
    compose_callbacks,
    finish_wandb,
    build_metrics_callback,
)

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithm', type=str, required=True, choices=['A2C', 'PPO', 'TRPO', 'TQC', 'ARS', 'CrossQ', 'RPPO'], help='The DRL algorithm to use')
    parser.add_argument('--load_set', required=True, type=int, help='The experiment set to load, from the sets defined in the experiments directory')
    parser.add_argument('--train_set', required=True, type=int, help='The experiment set to train on, from the sets defined in the experiments directory. Must be different from load_set for transfer learning')
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=0, help='The verbosity level: 0 no output, 1 info, 2 debug')
    parser.add_argument('--steps', type=int, default=2_000_000, help='The amount of steps to train the DRL model for while tuning')
    parser.add_argument('--num_envs', type=int, default=4, help='The number of parallel environments to run')
    parser.add_argument('--seed', type=int, default=None, help='The random seed to use')
    parser.add_argument('--log_steps', type=int, default=2000, help='The number of steps between each log entry')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='The device to tune on')
    # new
    parser.add_argument('--version', type=int, default=None, help='The version of the experiment')
    parser.add_argument('--use_wandb', type=parse_bool, default=False, help='If true, logs training to Weights & Biases')
    parser.add_argument('--wandb_project_name', type=str, default="MultiBotNav", help='The W&B project name for the experiments')
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated tags (optional)")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument('--checkpoint_freq', type=int, default=0, help='The frequency (in timesteps) to save model checkpoints')


    args = parser.parse_args()
    set_global_seeds(args.seed)

    # Make sure load_set and train_set are different
    if args.load_set == args.train_set:
        raise ValueError('load_set and train_set must be different for transfer learning')

    # Experiments file
    json_path  = 'src/new_2026_cont_sets.json'
    json_dict  = load_experiment_dict_json(json_path)
    num_robots = len(json_dict[f"set{args.train_set}"]['init_positions'])
    
    # Unique, consistent run name and directory
    run_name = (
        f"{args.algorithm}_from{args.load_set}_to{args.train_set}_seed{args.seed}_{args.version}"
        if args.version is not None
        else f"{args.algorithm}_from{args.load_set}_to{args.train_set}_seed{args.seed}"
    )
    log_type = "transfer_logs"
    log_dir, tensorboard_dir, checkpoints_dir, monitor_dir, run_config = setup_log_dirs(run_name, log_type, args)

    # from train_v2.py
    trained_model_run_name = f"{args.algorithm}_set{args.load_set}_seed{args.seed}_{args.version}" if args.version is not None else f"{args.algorithm}_set{args.load_set}_seed{args.seed}"
    trained_model_path     = os.path.join("logs", "training_default_logs", trained_model_run_name, "checkpoints", "trained_model.zip")

    # W&B init 
    wandb_run = None
    if args.use_wandb:
        wandb_run = init_wandb(
            project=args.wandb_project_name,
            run_name=f"{run_name}_{log_type}",
            config=run_config,
            log_dir=log_dir,
            tensorboard_dir=tensorboard_dir,
            group=args.wandb_group,
            tags= [t.strip() for t in args.wandb_tags.split(",") if t.strip()] if args.wandb_tags else None ,
            notes=None,
    )

    # Build env
    vec_env = make_vec_env(
        "MultiRobotEnv-v2",
        n_envs=args.num_envs,
        seed=args.seed, 
        monitor_dir=monitor_dir,
        wrapper_class=EpisodeMetricsInfoWrapper,
        env_kwargs={
            "field_info": json_dict[f"set{args.train_set}"],
            "render_mode": None,
            "num_robots": num_robots,
        },
    )
    # vec_env.seed(seed=args.seed)
    # vec_env.action_space.seed(seed=args.seed)

    # Logger 
    sb3_logger = configure(tensorboard_dir, ["stdout", "log", "csv", "json", "tensorboard"])

    # Callbacks
    log_every  = LogEveryNTimesteps(n_steps=args.log_steps)
    metrics_cb = build_metrics_callback(log_dir=log_dir, verbose=0)
    wandb_cb   = get_wandb_callback(args.use_wandb, log_dir=log_dir, verbose=args.verbose)
    if args.checkpoint_freq > 0:
        save_freq_calls = max(args.checkpoint_freq // max(args.num_envs, 1), 1)
        checkpoint_cb = CheckpointCallback(
            save_freq=save_freq_calls,
            save_path=checkpoints_dir,
            name_prefix='checkpoint',
        )
        callback = compose_callbacks([log_every, checkpoint_cb, metrics_cb, wandb_cb])
    else:
        callback = compose_callbacks([log_every, metrics_cb, wandb_cb])

    # sb3 model/policy 
    policy = get_policy_type(args.algorithm)
    model_type = get_model_type(args.algorithm)
    model = load_model(args.algorithm, 
                       args.load_set, 
                       args.seed, 
                       args.device,  
                       args.verbose, 
                       tensorboard_dir,
                       trained_model_path,
                       run_name)
    # loaded model's spaces match the new env's  spaces
    model.observation_space = vec_env.observation_space
    model.action_space = vec_env.action_space
    if hasattr(model, "policy"):
        model.policy.observation_space = vec_env.observation_space
        model.policy.action_space = vec_env.action_space

    model.set_env(vec_env)

    # Train
    start_time = datetime.now()
    print(f'Training started on {start_time.ctime()}')
    model.set_logger(sb3_logger)
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=callback,
            log_interval=None,
            tb_log_name=run_name,
            reset_num_timesteps=True, # ? 
            progress_bar=False,
        )
        checkpoint_path = os.path.join(checkpoints_dir, "trained_model.zip")
        model.save(checkpoint_path)
    except Exception as e:
        append_failure(
            scheme="transfer_v2",
            script="transfer_v2.py",
            run_name=run_name,
            extra={
                "algorithm": args.algorithm,
                "load_set": args.load_set,
                "train_set": args.train_set,
                "seed": args.seed,
                "version": args.version,
                "device": args.device,
                "exception_type": type(e).__name__,
                "exception_message": str(e),
            },
        )
        raise
    finally:
        end_time = datetime.now()
        print(f'Training ended on {end_time.ctime()}')
        print(f'Training lasted {end_time - start_time}')
        try:
            vec_env.close()
        except Exception:
            pass
        finish_wandb(wandb_run)