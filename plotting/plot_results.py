import glob
import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import argparse

def plot_setting_a():
    print('Plotting Setting A figure...')
    
    # Gather default hyperparameter training data
    training_data = []
    training_logs = glob.glob("./training_default_logs/*/*")
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

    # Plot Setting A figure
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10,8))
    seaborn.lineplot(data=train_df, x='step', y='reward', hue='algorithm')
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1)).set_title('')
    plt.ticklabel_format(style='sci', scilimits=(0,0))
    plt.grid()
    plt.tight_layout()
    plt.savefig('plotting/plots/setting_a.png')
    
def plot_setting_b():
    print('Plotting Setting B figure...')
    
    # Gather best hyperparameter training data
    training_data = []
    training_logs = glob.glob("./training_best_logs/*/*")
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

    # Plot Setting B figure
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10,8))
    seaborn.lineplot(data=train_df, x='step', y='reward', hue='algorithm')
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1)).set_title('')
    plt.ticklabel_format(style='sci', scilimits=(0,0))
    plt.grid()
    plt.tight_layout()
    plt.savefig('plotting/plots/setting_b.png')
        
def plot_setting_c():
    print('Plotting Setting C figure...')
    
    # Gather transfer data
    transfer_data = []
    transfer_logs = glob.glob("./transfer_logs/*/*")
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
    
    # Plot Setting C figure
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10,8))
    seaborn.lineplot(data=transfer_df, x='step', y='reward', hue='algorithm')
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1)).set_title('')
    plt.ticklabel_format(style='sci', scilimits=(0,0))
    plt.grid()
    plt.tight_layout()
    plt.savefig('plotting/plots/setting_a.png')
    
def plot_optuna():
    print('Plotting Optuna figure...')
    
    # Gather tuning data
    tuning_data = []
    tuning_logs = glob.glob("./tuning_logs/*/*")
    for log in tuning_logs:
        experiment_info = log.split('/')[2].split('_')
        algorithm = experiment_info[0]
        st = int(experiment_info[1][3:])
        trial = int(experiment_info[2])

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
    
    # Plot Optuna figure
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10,8))
    seaborn.lineplot(data=tune_df, x='step', y='reward', hue='algorithm')
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1)).set_title('')
    plt.ticklabel_format(style='sci', scilimits=(0,0))
    plt.grid()
    plt.tight_layout()
    plt.savefig('plotting/plots/setting_b.png')
    
if __name__ == '__main__':
    
    tf.get_logger().setLevel('INFO')
    
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
