import os
import argparse
import yaml
from datetime import datetime
from typing import Dict, Any
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import LogEveryNTimesteps, CheckpointCallback
from stable_baselines3.common.logger import configure
from src.EpisodeMetricsInfoWrapper import EpisodeMetricsInfoWrapper
from src.utils_v2 import (
    get_policy_type,
    get_model_type,
    read_env_config, 
    load_model, 
    parse_bool, 
    filter_args, 
    load_experiment_dict_json,
    setup_log_dirs,
    find_latest_checkpoint,
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

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithm', type=str, required=True, choices=['A2C', 'PPO', 'TRPO', 'ARS', 'CrossQ', 'TQC', 'RPPO'], help='The DRL algorithm to use')
    parser.add_argument('--set', required=True, type=int, help='The experiment set to use, from the sets defined in the experiments directory')
    parser.add_argument('--verbose', type=int, choices=[0, 1, 2], default=0, help='The verbosity level: 0 no output, 1 info, 2 debug')
    parser.add_argument('--steps', type=int, default=2_000_000, help='The amount of steps to train the DRL model for')
    parser.add_argument('--num_envs', type=int, default=4, help='The number of parallel environments to run')
    parser.add_argument('--seed', type=int, default=None, help='The random seed to use')
    parser.add_argument('--log_steps', type=int, default=2000, help='The number of steps between each log entry')
    parser.add_argument('--resume', type=parse_bool, default=False, help='If true, loads an existing model to resume training. If false, trains a new model')
    parser.add_argument('--use_tuned_params', type=parse_bool, default=False, help='If true, uses tuned hyperparameters. If false, uses default hyperparameters')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='The device to train on')
    # new
    parser.add_argument('--version', type=int, default=None, help='The version of the experiment')
    parser.add_argument('--tuned_params_path', type=str, default=None, help='Path to the tuned hyperparameters YAML file')
    parser.add_argument('--use_wandb', type=parse_bool, default=False, help='If true, logs training to Weights & Biases')
    parser.add_argument('--wandb_project_name', type=str, default="MultiBotNav", help='The W&B project name for the experiments')
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated tags (optional)")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument('--checkpoint_freq', type=int, default=0, help='The frequency (in timesteps) to save model checkpoints')

    args = parser.parse_args()
    set_global_seeds(args.seed)

    # Experiments file
    json_path  = 'src/new_2026_cont_sets.json'
    json_dict  = load_experiment_dict_json(json_path)
    num_robots = len(json_dict[f"set{args.set}"]['init_positions'])

    # Unique, consistent run name and directory
    run_name = (
        f"{args.algorithm}_set{args.set}_seed{args.seed}_{args.version}"
        if args.version is not None
        else f"{args.algorithm}_set{args.set}_seed{args.seed}"
    )
    log_type = "training_best_logs" if args.use_tuned_params else "training_default_logs"
    log_dir, tensorboard_dir, checkpoints_dir, monitor_dir, run_config = setup_log_dirs(run_name, log_type, args)

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
            "field_info": json_dict[f"set{args.set}"],
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

    # load (tuned) hyperparameters
    if args.use_tuned_params:
        if args.tuned_params_path is not None:
            tuned_path = args.tuned_params_path
        else:
            # Rebuild path from tune_v2.py 
            study_name = (
                f"{args.algorithm}_set{args.set}_seed{args.seed}_{args.version}"
                if args.version is not None
                else f"{args.algorithm}_set{args.set}_seed{args.seed}"
            )
            study_dir = os.path.join("logs", "tuning_logs", study_name)
            tuned_path = os.path.join(study_dir, "best_hyperparameters.yaml")
        # load from file
        with open(tuned_path) as file:
            try:
                payload = yaml.safe_load(file)
                # tune_v2.py writes {"filtered_params": {...}, ...}
                params = payload.get("filtered_params", payload)
                hyperparameters = filter_args(params, model_type)
            except yaml.YAMLError as err:
                print(err)
                hyperparameters = {}
    else:
        hyperparameters = {}

    # Model 
    model_kwargs: Dict[str, Any] = {
        "policy": policy,
        "env": vec_env,
        "seed": args.seed,
        "device": args.device,
        "verbose": args.verbose,
        "tensorboard_log": tensorboard_dir,
        **hyperparameters
    }

    # Load or create model
    if args.resume:
        checkpoint_path = find_latest_checkpoint(checkpoints_dir)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"--resume was set but checkpoint not found: {checkpoint_path}")
        model = model_type.load(checkpoint_path, env=vec_env, device=args.device)
    else:
        filtered_kwargs = filter_args(model_kwargs, model_type)
        model = model_type(**filtered_kwargs)

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
            reset_num_timesteps=not args.resume,
            progress_bar=False,
        )
        checkpoint_path = os.path.join(checkpoints_dir, "trained_model.zip")
        model.save(checkpoint_path)
    except Exception as e:
        append_failure(
            scheme="train_from_tuned" if args.use_tuned_params else "train",
            script="train_v2.py",
            run_name=run_name,
            extra={
                "algorithm": args.algorithm,
                "set": args.set,
                "seed": args.seed,
                "version": args.version,
                "device": args.device,
                "use_tuned_params": bool(args.use_tuned_params),
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

