import optuna
import os
import argparse
import yaml
import gc
from datetime import datetime
from typing import Any, Dict, Optional
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import LogEveryNTimesteps
from stable_baselines3.common.logger import configure
from src.EpisodeMetricsInfoWrapper import EpisodeMetricsInfoWrapper
from src.utils_v2 import (
    get_policy_type,
    get_model_type,
    parse_bool,
    filter_args,
    load_experiment_dict_json,
    setup_optuna_study_dirs,
    setup_trial_dirs,
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

# Global tracking 
BEST_REWARD: float = -1e18
BEST_TRIAL_NUMBER: Optional[int] = None

def _cuda_cleanup(device: str) -> None:
    if device != "cuda":
        return
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

def objective(trial: optuna.Trial) -> float:

    global BEST_REWARD, BEST_TRIAL_NUMBER

    # Setup trial directories
    trial_dir, tensorboard_dir, monitor_dir = setup_trial_dirs(trials_dir, trial.number)

    trial_config: Dict[str, Any] = {
        "trial_number": int(trial.number),
        "started_at": datetime.now().isoformat(),
        "algorithm": args.algorithm,
        "set": args.set,
        "seed": args.seed,
        "device": args.device,
        "steps": args.steps,
        "num_envs": args.num_envs,
        "num_eval_eps": args.num_eval_eps,
    }

    # env (training)
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

    # separate eval env
    eval_env = make_vec_env(
        "MultiRobotEnv-v2",
        n_envs=1,
        seed=args.seed,
        wrapper_class=EpisodeMetricsInfoWrapper,
        monitor_dir=os.path.join(trial_dir, "monitor_eval"),
        env_kwargs={
            "field_info": json_dict[f"set{args.set}"],
            "render_mode": None,
            "num_robots": num_robots,
        },
    )
    # vec_env.seed(seed=args.seed)
    # vec_env.action_space.seed(seed=args.seed)

    # Base model args
    model_args = {
        'policy': policy,
        'env': vec_env,
        'tensorboard_log': tensorboard_dir,
        'seed': args.seed,
        'device': args.device,
        'n_steps': trial.suggest_categorical("n_steps", [5, 10, 20] if args.algorithm == 'A2C' else [1024, 2048, 4096]),
        'gamma': trial.suggest_float("gamma", 0.90, 0.99),
        'ent_coef': trial.suggest_float("ent_coef", 0.0, 0.05),
        'gae_lambda': trial.suggest_float("gae_lambda", 0.9, 1.0),
        'max_grad_norm': trial.suggest_float("max_grad_norm", 0.30, 0.99),
        'vf_coef': trial.suggest_float("vf_coef", 0.2, 0.7), # 0.10, 1.00)?
        # Update Optuna (comment out)
        # 'buffer_size': trial.suggest_int("buffer_size", 1000, 100000, step=1000) #10_000, 200_000, step=10_000),
        # 'learning_rate': trial.suggest_float("learning_rate", 0.0001, 0.05, log=True),
    }
    # Update Optuna (comment out)
    # Learning rate - algorithm specific 
    if args.algorithm in {"PPO","TQC","CrossQ"}:
        lr_low, lr_high = 1e-5, 5e-4
    else:
        lr_low, lr_high = 1e-4, 5e-2
    model_args["learning_rate"] = trial.suggest_float("learning_rate", lr_low, lr_high, log=True)
    # On-policy
    if args.algorithm in {"PPO", "TRPO", "A2C"}:
        # PPO/TRPO: batch_size, n_epochs, clip_range are key
        if args.algorithm in {"PPO", "TRPO"}:
            model_args.update({
                    "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
                    "n_epochs": trial.suggest_categorical("n_epochs", [5, 10, 20]),
            })
            # PPO has clip_range; TRPO typically doesn't (filter_args will drop if unsupported)
            model_args.update({"clip_range": trial.suggest_float("clip_range", 0.1, 0.3),})
    # Off-policy
    if args.algorithm in {"TQC", "CrossQ"}:
        model_args.update({
                "buffer_size": trial.suggest_int("buffer_size", 50_000, 500_000, step=50_000),
                "learning_starts": trial.suggest_int("learning_starts", 1_000, 20_000, step=1_000),
                "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8, 16]),
                "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 4, 8, 16]),
                "tau": trial.suggest_float("tau", 0.001, 0.02, log=True),
        })
    # ARS specific
    if args.algorithm == "ARS":
        model_args.update({
                "n_delta": trial.suggest_categorical("n_delta", [8, 16, 32, 64]),
                "n_top": trial.suggest_categorical("n_top", [4, 8, 16, 32]),
                "delta_std": trial.suggest_float("delta_std", 0.01, 0.15),
        })
    # record - needed?
    trial_config["sampled_params"] = dict(trial.params)
    with open(os.path.join(trial_dir, "trial_config.yaml"), "w") as f:
        yaml.safe_dump(trial_config, f, sort_keys=False)
    # Callbacks / logger (same pattern as train_v2.py)
    sb3_logger = configure(tensorboard_dir, ["stdout", "log", "csv", "json", "tensorboard"])
    log_every = LogEveryNTimesteps(n_steps=args.log_steps)
    metrics_cb = build_metrics_callback(log_dir=trial_dir, verbose=0)
    # W&B
    wandb_run = None
    wandb_cb = None
    if args.use_wandb and args.wandb_each_trial:
        wandb_run = init_wandb(
            project=args.wandb_project_name,
            run_name=f"{study_name}_trial{trial.number:03d}",
            config={**study_config, **trial_config},
            log_dir=trial_dir,
            tensorboard_dir=tensorboard_dir,
            group=args.wandb_group,
            tags=[t.strip() for t in args.wandb_tags.split(",") if t.strip()] if args.wandb_tags else None,
        )
        wandb_cb = get_wandb_callback(True, log_dir=trial_dir, verbose=0)
    callback = compose_callbacks([log_every, metrics_cb, wandb_cb])

    model = None
    try:
        filtered_args = filter_args(model_args, model_type)
        model = model_type(**filtered_args)
        model.set_logger(sb3_logger)
        # Train
        model.learn(
            total_timesteps=args.steps,
            callback=callback,
            log_interval=None,
            tb_log_name=f"{study_name}_trial{trial.number:03d}",
            reset_num_timesteps=True, #?
            progress_bar=False, #?
        )
        # Evaluate
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env, # ? vec_env,
            n_eval_episodes=args.num_eval_eps,
            deterministic=True,
        )
        # Record into Optuna ?
        trial.set_user_attr("eval_mean_reward", float(mean_reward))
        trial.set_user_attr("eval_std_reward", float(std_reward))
        # Also record to W&B (so you can sort trials by eval score in the UI)
        if wandb_run is not None:
            try:
                import wandb
                wandb.log({
                        "eval/mean_reward": float(mean_reward),
                        "eval/std_reward": float(std_reward),
                        "trial_number": int(trial.number),
                })
            except Exception:
                pass
        # Best !
        if mean_reward > BEST_REWARD:
            BEST_REWARD = float(mean_reward)
            BEST_TRIAL_NUMBER = int(trial.number)
            filtered_args = dict(filtered_args)
            filtered_args.pop("env", None)
            best_payload = {
                "study_name": study_name,
                "best_trial_number": int(trial.number),
                "best_value_mean_reward": float(mean_reward),
                "filtered_params": filtered_args,
            }
            with open(best_params_path, "w") as f:
                yaml.safe_dump(best_payload, f, sort_keys=False)
            model.save(best_model_path)
            
        return float(mean_reward)
    # Failure anywhere in trial (objective function)
    except Exception as e:
        append_failure(
            scheme="tune_trial",
            script="tune_v2.py",
            run_name=study_name,
            extra={"trial_number": int(trial.number), 
                   "params": dict(trial.params),            
                   "exception_type": type(e).__name__,
                   "exception_message": str(e),},
        )
        raise

    finally:
        try:
            vec_env.close()
            eval_env.close()
            if model is not None: del model
        except Exception:
            pass
        gc.collect()
        _cuda_cleanup(args.device)
        finish_wandb(wandb_run)





if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithm', type=str, required=True, choices=['A2C', 'PPO', 'TRPO', 'TQC', 'ARS', 'CrossQ', 'RPPO'], help='The DRL algorithm to use')
    parser.add_argument('--set', required=True, type=int, help='The experiment set to use, from the sets defined in the experiments directory')
    parser.add_argument('--trials', type=int, default=20, help='The number of trials used for tuning')
    parser.add_argument('--steps', type=int, default=1_000_000, help='The amount of steps to train the DRL model for while tuning')
    parser.add_argument('--num_envs', type=int, default=4, help='The number of parallel environments to run')
    parser.add_argument('--num_eval_eps', type=int, default=10, help='The number of episodes for evaluating a trial')
    parser.add_argument('--seed', type=int, default=None, help='The random seed to use')
    parser.add_argument('--log_steps', type=int, default=2000, help='The number of steps between each log entry')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='The device to tune on')
    # new
    parser.add_argument("--version", type=int, default=None, help="Optional version tag for this tuning run")
    parser.add_argument("--show_progress_bar", type=parse_bool, default=True, help="Optuna progress bar (disable for cluster logs)")
    parser.add_argument("--use_wandb", type=parse_bool, default=False, help="Log to Weights & Biases")
    parser.add_argument("--wandb_project_name", type=str, default="MultiBotNav", help='The W&B project name for the experiments')
    parser.add_argument("--wandb_group", type=str, default=None, help="Optional W&B group")
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated tags")
    parser.add_argument("--wandb_each_trial", type=parse_bool, default=True, help="If true, creates a separate W&B run per trial (recommended).")


    args = parser.parse_args()
    set_global_seeds(args.seed)

    # Experiments file
    json_path  = 'src/new_2026_cont_sets.json'
    json_dict  = load_experiment_dict_json(json_path)
    num_robots = len(json_dict[f"set{args.set}"]['init_positions'])

    # Unique, consistent run name and directory
    study_name = (
        f"{args.algorithm}_set{args.set}_seed{args.seed}_{args.version}"
        if args.version is not None
        else f"{args.algorithm}_set{args.set}_seed{args.seed}"
    )
    study_dir, trials_dir, tuned_params_dir, best_model_path, best_params_path, study_config = setup_optuna_study_dirs(study_name, args)

    # sb3 model/policy 
    policy = get_policy_type(args.algorithm)
    model_type = get_model_type(args.algorithm)

    BEST_REWARD = -1e18
    BEST_TRIAL_NUMBER = None
    # Optuna study
    try:
        start_time = datetime.now()
        print(f'Tuning started on {start_time.ctime()}')
        create_kwargs: Dict[str, Any] = {
            "direction": "maximize",
            "study_name": study_name,
            "sampler": optuna.samplers.TPESampler(seed=args.seed) if args.seed is not None else optuna.samplers.TPESampler(),
        }
        study = optuna.create_study(**create_kwargs) 
        study.optimize(objective, n_trials=args.trials, show_progress_bar=args.show_progress_bar, gc_after_trial=True, catch=(ValueError,Exception,))
        # study = optuna.create_study(direction="maximize")
        # study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
        end_time = datetime.now()
        print(f'Tuning ended on {end_time.ctime()}')
        print(f'Tuning lasted {end_time - start_time}\n')

        # Best hyperparameters
        filtered_params = filter_args(study.best_params, model_type)
        filtered_params = dict(filtered_params)
        filtered_params.pop("env", None)
        best_payload = {
            "study_name": study_name,
            "best_trial_number": int(study.best_trial.number),
            "best_value_mean_reward": float(study.best_value),
            "filtered_params": filtered_params,
        }
        with open(best_params_path, "w") as f:
            yaml.safe_dump(best_payload, f, sort_keys=False)
        print(f"Best hyperparameters for {args.algorithm}: {filtered_params}")
        print(f"Best mean reward for {args.algorithm}: {BEST_REWARD}")
        print(f"Best trial number: {study.best_trial.number}")
        print(f"Saved best params to: {best_params_path}")
        print(f"Saved best model to: {best_model_path}")
    # Failure anywhere in entire tuning run 
    except Exception as e:
        append_failure(
            scheme="tune",
            script="tune_v2.py",
            run_name=study_name,
            extra={
                "algorithm": args.algorithm,
                "set": args.set,
                "seed": args.seed,
                "version": args.version,
                "device": args.device,
                "exception_type": type(e).__name__,
                "exception_message": str(e),
            },
        )
        raise
