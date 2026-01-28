import os
import json
from typing import Any, Dict, Optional, List

from stable_baselines3.common.callbacks import BaseCallback


def init_wandb(
    *,
    project: str,
    run_name: str,
    config: Dict[str, Any],
    log_dir: str,
    tensorboard_dir: Optional[str] = None,
    group: Optional[str] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> Optional["wandb.sdk.wandb_run.Run"]:
    """Initialize a W&B run or raise if W&B was requested but cannot be used."""
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError(f"use_wandb=True but wandb is not importable: {exc}") from exc

    if tensorboard_dir is None:
        raise RuntimeError("tensorboard_dir must be provided when use_wandb=True")

    # Ensure W&B uses the intended TB directory.
    os.environ["WANDB_TENSORBOARD_LOG_DIR"] = tensorboard_dir

    # Do not rely on WANDB_PROJECT env var; make it explicit.
    run = wandb.init(
        project=project,
        #entity=entity,
        name=run_name,
        group=group,
        tags=tags,
        notes=notes,
        config=config,
        dir=log_dir,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    if run is None:
        raise RuntimeError("wandb.init() returned None while use_wandb=True")

    return run

def get_wandb_callback(
    use_wandb: bool,
    *,
    log_dir: str,
    verbose: int = 0,
    model_save_path: Optional[str] = None,
) -> Optional[BaseCallback]:
    """Return WandbCallback, raising if unavailable when use_wandb=True."""
    if not use_wandb:
        return None

    try:
        from wandb.integration.sb3 import WandbCallback
    except Exception as exc:
        raise RuntimeError(
            f"use_wandb=True but wandb.integration.sb3.WandbCallback is unavailable: {exc}"
        ) from exc

    if model_save_path is not None:
        os.makedirs(model_save_path, exist_ok=True)
    return WandbCallback(model_save_path=model_save_path, verbose=verbose)

def finish_wandb(run: Optional[Any]) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass

def finish_wandb_quietly(run: Optional[Any]) -> None:
    try:
        finish_wandb(run)
    except Exception:
        pass

def compose_callbacks(callbacks: List[Optional[BaseCallback]]) -> Optional[BaseCallback]:
    callbacks = [cb for cb in callbacks if cb is not None]
    if not callbacks:
        return None
    if len(callbacks) == 1:
        return callbacks[0]

    from stable_baselines3.common.callbacks import CallbackList

    return CallbackList(callbacks)

class EpisodeMetricsCallback(BaseCallback):
    """Logs episode-level metrics injected via env info dicts.

    Guarantees:
    - Metrics are written to SB3 logger (csv/json/tensorboard if configured).
    - All per-episode metric payloads are appended to a JSONL file so nothing is lost
      even if SB3 logging windows aggregate values.

    Expected env info format (at episode end):
        info["episode_metrics"] = { ... }
    """

    def __init__(
        self,
        *,
        log_dir: str,
        verbose: int = 0,
        jsonl_filename: str = "episode_metrics.jsonl",
    ):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.jsonl_path = os.path.join(log_dir, jsonl_filename)
        self._fh = None
        self.episode_count = 0
        self.success_count = 0

    def _init_callback(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        # Line-buffered for safety on clusters.
        self._fh = open(self.jsonl_path, "a", buffering=1)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if not infos or dones is None:
            return True

        for info, done in zip(infos, dones):
            if not done:
                continue
            metrics = info.get("episode_metrics")
            if not metrics:
                continue

            self.episode_count += 1
            success = bool(metrics.get("success", False))
            if success:
                self.success_count += 1

            # Global success rate across the whole run.
            success_rate_global = (self.success_count / self.episode_count if self.episode_count else 0.0)

            # Record means so multiple episode completions within one dump window are not overwritten.
            self.logger.record_mean("episode/success", int(success))
            self.logger.record("episode/success_rate_global", success_rate_global)
            self.logger.record_mean("episode/collision_count", int(metrics.get("collision_count", 0)))
            self.logger.record_mean("episode/mean_speed", float(metrics.get("mean_speed", 0.0)))
            self.logger.record_mean("episode/wasted_spray_fraction",float(metrics.get("wasted_spray_fraction", 0.0)),)
            self.logger.record_mean("episode/steps", int(metrics.get("episode_steps", 0)))
            self.logger.record_mean("episode/spray_attempted", float(metrics.get("spray_attempted", 0.0)))
            self.logger.record_mean("episode/spray_applied", float(metrics.get("spray_applied", 0.0)))

            # Reward component breakdown
            self.logger.record_mean("reward_component/spray", float(metrics.get("reward_spray", 0.0)))
            self.logger.record_mean("reward_component/penalty_boundary",float(metrics.get("reward_penalty_boundary", 0.0)),)
            self.logger.record_mean("reward_component/penalty_visit",float(metrics.get("reward_penalty_visit", 0.0)),)
            self.logger.record_mean("reward_component/collision", float(metrics.get("reward_collision", 0.0)))
            self.logger.record_mean("reward_component/success", float(metrics.get("reward_success", 0.0)))
            self.logger.record_mean("reward_component/penalty_total",float(metrics.get("reward_penalty_total", 0.0)),)

            # Per-robot metrics
            distances = metrics.get("distance_by_robot", [])
            for idx, dist in enumerate(distances):
                self.logger.record_mean(f"episode/distance_robot{idx}", float(dist))

            rewards_by_robot = metrics.get("reward_by_robot", [])
            for idx, rew in enumerate(rewards_by_robot):
                self.logger.record_mean(f"reward_by_robot/robot{idx}", float(rew))

            # Append raw episode metrics to JSONL.
            # Avoid extremely large per-episode payloads (e.g. step-wise reward histories).
            if self._fh is not None:
                metrics_for_file = dict(metrics)
                metrics_for_file.pop("step_rewards", None)
                payload = {
                    "timestep": int(self.num_timesteps),
                    "episode": int(self.episode_count),
                    "metrics": metrics_for_file,
                }
                self._fh.write(json.dumps(payload) + "\n")

        return True

    def _on_training_end(self) -> None:
        try:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
        finally:
            self._fh = None


def build_metrics_callback(*, log_dir: str, verbose: int = 0) -> EpisodeMetricsCallback:
    return EpisodeMetricsCallback(log_dir=log_dir, verbose=verbose)
