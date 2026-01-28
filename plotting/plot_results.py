import glob
import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import argparse

MAX_STEPS = 2_000_000
ALGO_COLORS = {
    "A2C": "#1f77b4",      # blue
    "PPO": "#ff7f0e",      # orange
    "TRPO": "#2ca02c",     # green
    "ARS": "#d62728",      # red
    "TQC": "#9467bd",      # purple
    "CrossQ": "#e377c2",   # pink
    "RPPO": "#8c564b",     # brown (if used)
}


def plot_setting_a():
    print('Plotting Setting A figure...')
    
    # Gather default hyperparameter training data
    training_data = []
    # training_logs = glob.glob("./training_default_logs/*/*")
    training_logs = glob.glob("logs/training_default_logs/*_v0/tensorboard/**/events.out.tfevents.*", recursive=True)
    for log in training_logs:
        experiment_info = log.split('/')[2].split('_')
        algorithm = experiment_info[0]
        st = int(experiment_info[1][3:])

        for e in tf.compat.v1.train.summary_iterator(log):
            for v in e.summary.value:
                if v.tag == 'rollout/ep_rew_mean':
                    training_data.append({
                        'algorithm': algorithm,
                        'set': st,
                        'step': e.step,
                        'reward': v.simple_value
                    })
    train_df = pd.DataFrame(training_data)
    train_df = train_df[train_df["step"] <= MAX_STEPS]
    if train_df.empty:
        print("No data found (after filtering). Skipping.")
        return

    # Plot Setting A figure
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10,8))
    seaborn.lineplot(data=train_df, x='step', y='reward', hue='algorithm', palette=ALGO_COLORS,  hue_order=sorted(ALGO_COLORS.keys()))
    plt.xlim(0, MAX_STEPS)
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1)).set_title('')
    plt.ticklabel_format(style='sci', scilimits=(0,0))
    plt.grid()
    plt.tight_layout()
    plt.savefig('plotting/plots/setting_a.png')
    
def plot_setting_b():
    print('Plotting Setting B figure...')
    
    # Gather best hyperparameter training data
    training_data = []
    # training_logs = glob.glob("./training_best_logs/*/*")
    training_logs = glob.glob("logs/training_best_logs/*_v0/tensorboard/**/events.out.tfevents.*", recursive=True)
    for log in training_logs:
        experiment_info = log.split('/')[2].split('_')
        algorithm = experiment_info[0]
        st = int(experiment_info[1][3:])

        for e in tf.compat.v1.train.summary_iterator(log):
            for v in e.summary.value:
                if v.tag == 'rollout/ep_rew_mean':
                    training_data.append({
                        'algorithm': algorithm,
                        'set': st,
                        'step': e.step,
                        'reward': v.simple_value
                    })
    train_df = pd.DataFrame(training_data)
    train_df = train_df[train_df["step"] <= MAX_STEPS]
    if train_df.empty:
        print("No data found (after filtering). Skipping.")
        return

    # Plot Setting B figure
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10,8))
    seaborn.lineplot(data=train_df, x='step', y='reward', hue='algorithm', palette=ALGO_COLORS,  hue_order=sorted(ALGO_COLORS.keys()))
    plt.xlim(0, MAX_STEPS)
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1)).set_title('')
    plt.ticklabel_format(style='sci', scilimits=(0,0))
    plt.grid()
    plt.tight_layout()
    plt.savefig('plotting/plots/setting_b.png')
        
def plot_setting_c():
    print('Plotting Setting C figure...')
    
    # Gather transfer data
    transfer_data = []
    # transfer_logs = glob.glob("./transfer_logs/*/*")
    transfer_logs = glob.glob("logs/transfer_logs/*_v0/tensorboard/**/events.out.tfevents.*", recursive=True)
    for log in transfer_logs:
        experiment_info = log.split('/')[2].split('_')
        algorithm = experiment_info[0]
        st = int(experiment_info[2][2:])

        for e in tf.compat.v1.train.summary_iterator(log):
            for v in e.summary.value:
                if v.tag == 'rollout/ep_rew_mean':
                    transfer_data.append({
                        'type': 'transfer',
                        'algorithm': algorithm,
                        'set': st,
                        'step': e.step,
                        'reward': v.simple_value
                    })
    transfer_df = pd.DataFrame(transfer_data)
    transfer_df = transfer_df[transfer_df["step"] <= MAX_STEPS]
    if transfer_df.empty:
        print("No data found (after filtering). Skipping.")
        return

    # Plot Setting C figure
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10,8))
    seaborn.lineplot(data=transfer_df, x='step', y='reward', hue='algorithm', palette=ALGO_COLORS,  hue_order=sorted(ALGO_COLORS.keys()))
    plt.xlim(0, MAX_STEPS)
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1)).set_title('')
    plt.ticklabel_format(style='sci', scilimits=(0,0))
    plt.grid()
    plt.tight_layout()
    plt.savefig('plotting/plots/setting_c.png')
    
def plot_optuna():
    print('Plotting Optuna figure...')
    
    # Gather tuning data
    tuning_data = []
    # tuning_logs = glob.glob("./tuning_logs/*/*")
    tuning_logs = glob.glob("logs/tuning_logs/*_v0/trials/trial_*/tensorboard/**/events.out.tfevents.*", recursive=True)
    for log in tuning_logs:
        experiment_info = log.split('/')[2].split('_')
        algorithm = experiment_info[0]
        st = int(experiment_info[1][3:])
        # trial = int(experiment_info[2])
        trial_dir = log.split('/')[4]          # e.g. "trial_000"
        trial = int(trial_dir.split('_')[1])   # -> 0

        for e in tf.compat.v1.train.summary_iterator(log):
            for v in e.summary.value:
                if v.tag == 'rollout/ep_rew_mean':
                    tuning_data.append({
                        'algorithm': algorithm,
                        'set': st,
                        'step': e.step,
                        'reward': v.simple_value,
                        'trial': trial
                    })
    tune_df = pd.DataFrame(tuning_data)
    tune_df = tune_df[tune_df["step"] <= MAX_STEPS]
    if tune_df.empty:
        print("No data found (after filtering). Skipping.")
        return
    # Plot Optuna figure
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10,8))
    seaborn.lineplot(data=tune_df, x='step', y='reward', hue='algorithm', palette=ALGO_COLORS,  hue_order=sorted(ALGO_COLORS.keys()))
    plt.xlim(0, MAX_STEPS)
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1)).set_title('')
    plt.ticklabel_format(style='sci', scilimits=(0,0))
    plt.grid()
    plt.tight_layout()
    plt.savefig('plotting/plots/optuna.png')
    
if __name__ == '__main__':
    
    tf.get_logger().setLevel('INFO')
    os.makedirs("plotting/plots", exist_ok=True)
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-a', action='store_true', help='Plot results for Setting A')
    parser.add_argument('-b', action='store_true', help='Plot results for Setting B')
    parser.add_argument('-c', action='store_true', help='Plot results for Setting C')
    parser.add_argument('-o', action='store_true', help='Plot results for Optuna trials')
    
    args = parser.parse_args()
    
    plot_all = not any(vars(args).values())

    if args.a or plot_all:
        plot_setting_a()
    if args.b or plot_all:
        plot_setting_b()
    if args.c or plot_all:
        plot_setting_c()
    if args.o or plot_all:
        plot_optuna()
