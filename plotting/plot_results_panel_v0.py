import glob
import logging
import os

# Silence TF spam before importing tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").disabled = True

import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
from typing import Tuple, List


ALGO_COLORS = {
    "A2C": "#1f77b4",      # blue
    "ARS": "#ff7f0e",      # orange
    "CrossQ": "#2ca02c",   # green
    "PPO": "#d62728",      # red
    "TQC": "#9467bd",      # purple
    "TRPO": "#8c564b",     # brown
    "RPPO": "#e377c2",     # pink
}
ALGO_ORDER = list(ALGO_COLORS.keys())


def hue_order_for(df: pd.DataFrame) -> List[str]:
    if df.empty or "algorithm" not in df.columns:
        return []
    present = set(df["algorithm"].unique())
    return [a for a in ALGO_ORDER if a in present]


def apply_step_cap(df: pd.DataFrame, max_steps: int) -> Tuple[pd.DataFrame, int]:
    if df.empty:
        return df, max_steps
    df = df[df["step"] <= max_steps]
    if df.empty:
        return df, max_steps
    xmax = int(min(max_steps, df["step"].max()))
    return df, xmax


def _sns_lineplot(*, ax, df: pd.DataFrame, x: str, y: str, hue: str):
    order = hue_order_for(df)
    try:
        sns.lineplot(
            data=df,
            x=x,
            y=y,
            hue=hue,
            palette=ALGO_COLORS,
            hue_order=order,
            errorbar=None,  # seaborn>=0.12
            ax=ax,
        )
    except TypeError:
        sns.lineplot(
            data=df,
            x=x,
            y=y,
            hue=hue,
            palette=ALGO_COLORS,
            hue_order=order,
            ci=None,  # seaborn<0.12
            ax=ax,
        )


def _reward_label(reward_scale: float) -> str:
    if reward_scale == 1:
        return "reward"
    if reward_scale == 1e6:
        return "reward (x10^6)"
    return f"reward / {reward_scale:g}"


def collect_setting_a(run_glob: str) -> pd.DataFrame:
    rows = []
    logs = glob.glob(
        f"logs/training_default_logs/{run_glob}/tensorboard/**/events.out.tfevents.*",
        recursive=True,
    )
    for log in logs:
        experiment_info = log.split("/")[2].split("_")
        algorithm = experiment_info[0]
        st = int(experiment_info[1][3:])
        for e in tf.compat.v1.train.summary_iterator(log):
            for v in e.summary.value:
                if v.tag == "rollout/ep_rew_mean":
                    rows.append({"algorithm": algorithm, "set": st, "step": e.step, "reward": v.simple_value})
    return pd.DataFrame(rows)


def collect_setting_b(run_glob: str) -> pd.DataFrame:
    rows = []
    logs = glob.glob(
        f"logs/training_best_logs/{run_glob}/tensorboard/**/events.out.tfevents.*",
        recursive=True,
    )
    for log in logs:
        experiment_info = log.split("/")[2].split("_")
        algorithm = experiment_info[0]
        st = int(experiment_info[1][3:])
        for e in tf.compat.v1.train.summary_iterator(log):
            for v in e.summary.value:
                if v.tag == "rollout/ep_rew_mean":
                    rows.append({"algorithm": algorithm, "set": st, "step": e.step, "reward": v.simple_value})
    return pd.DataFrame(rows)


def collect_setting_c(run_glob: str) -> pd.DataFrame:
    rows = []
    logs = glob.glob(
        f"logs/transfer_logs/{run_glob}/tensorboard/**/events.out.tfevents.*",
        recursive=True,
    )
    for log in logs:
        experiment_info = log.split("/")[2].split("_")
        algorithm = experiment_info[0]
        st = int(experiment_info[2][2:])  # "to7" -> 7
        for e in tf.compat.v1.train.summary_iterator(log):
            for v in e.summary.value:
                if v.tag == "rollout/ep_rew_mean":
                    rows.append({"algorithm": algorithm, "set": st, "step": e.step, "reward": v.simple_value})
    return pd.DataFrame(rows)


def collect_optuna(run_glob: str) -> pd.DataFrame:
    rows = []
    logs = glob.glob(
        f"logs/tuning_logs/{run_glob}/trials/trial_*/tensorboard/**/events.out.tfevents.*",
        recursive=True,
    )
    for log in logs:
        experiment_info = log.split("/")[2].split("_")
        algorithm = experiment_info[0]
        st = int(experiment_info[1][3:])
        trial_dir = log.split("/")[4]  # e.g. "trial_000"
        trial = int(trial_dir.split("_")[1])
        for e in tf.compat.v1.train.summary_iterator(log):
            for v in e.summary.value:
                if v.tag == "rollout/ep_rew_mean":
                    rows.append(
                        {"algorithm": algorithm, "set": st, "trial": trial, "step": e.step, "reward": v.simple_value}
                    )
    return pd.DataFrame(rows)


def plot_panel(
    *,
    ax,
    df: pd.DataFrame,
    max_steps: int,
    reward_scale: float,
    subtitle: str,
    legend_loc: str,
) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df = df.copy()
    if reward_scale != 1:
        df["reward"] = df["reward"] / float(reward_scale)

    df, xmax = apply_step_cap(df, max_steps)

    if df.empty:
        ax.text(0.5, 0.5, "No data (after step cap)", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    _sns_lineplot(ax=ax, df=df, x="step", y="reward", hue="algorithm")
    ax.set_xlabel("steps")
    ax.set_ylabel(_reward_label(reward_scale))
    ax.set_xlim(0, xmax)
    ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")
    ax.grid(True)

    leg = ax.legend(loc=legend_loc)
    if leg is not None:
        leg.set_title("")

    ax.text(
        0.5,
        -0.22,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=18,
    )


def main() -> None:
    tf.get_logger().setLevel("INFO")

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, default=33, help="Filter runs by seed (default: 33).")
    parser.add_argument("--version", "-v", type=int, default=0, help="Version suffix used in run names (default: 0 -> _v0).")
    parser.add_argument("--max_steps", type=int, default=2_000_000, help="Cap plots at this many steps.")
    parser.add_argument("--reward_scale", type=float, default=1e6, help="Divide reward by this value for display.")
    parser.add_argument("--title", type=str, default="", help="Optional extra title text (default: blank).")
    parser.add_argument("--outdir", type=str, default="plotting/plots", help="Directory to save the panel figure.")
    parser.add_argument("--outfile", type=str, default=None, help="Optional override for output filename (inside --outdir).")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    seed_label = f"seed{args.seed}" if args.seed is not None else "allseeds"
    run_glob = f"*_seed{args.seed}_v{args.version}" if args.seed is not None else f"*_v{args.version}"

    df_a = collect_setting_a(run_glob)
    df_b = collect_setting_b(run_glob)
    df_c = collect_setting_c(run_glob)
    df_o = collect_optuna(run_glob)

    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    plot_panel(
        ax=axes[0, 0],
        df=df_a,
        max_steps=args.max_steps,
        reward_scale=args.reward_scale,
        subtitle="(a) Setting A (default)",
        legend_loc="lower right",
    )
    plot_panel(
        ax=axes[0, 1],
        df=df_b,
        max_steps=args.max_steps,
        reward_scale=args.reward_scale,
        subtitle="(b) Setting B (random)",
        legend_loc="lower right",
    )
    plot_panel(
        ax=axes[1, 0],
        df=df_c,
        max_steps=args.max_steps,
        reward_scale=args.reward_scale,
        subtitle="(c) Setting C (transfer)",
        legend_loc="lower right",
    )
    plot_panel(
        ax=axes[1, 1],
        df=df_o,
        max_steps=args.max_steps,
        reward_scale=args.reward_scale,
        subtitle="(d) Optuna trials",
        legend_loc="upper left",
    )

    header = f"Seed: {seed_label} | Version: {args.version}"
    if args.title.strip():
        header += f" | {args.title.strip()}"
    fig.suptitle(header, fontsize=18)

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_name = args.outfile or f"panel_v{args.version}_{seed_label}.png"
    out_path = os.path.join(args.outdir, out_name)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
