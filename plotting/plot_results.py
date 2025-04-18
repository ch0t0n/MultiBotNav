import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from cycler import cycler
import seaborn

if __name__ == '__main__':

    # Gather data
    data = []
    logs = glob.glob("./logs/*/*")
    for log in logs:
        experiment_info = log.split('/')[2].split('_')
        algorithm = experiment_info[0]
        st = experiment_info[1]

        for e in tf.compat.v1.train.summary_iterator(log):
            for v in e.summary.value:
                if v.tag == 'rollout/ep_rew_mean':
                    data.append({
                        'algorithm': algorithm,
                        'set': st,
                        'step': e.step,
                        'reward': v.simple_value
                    })
    df = pd.DataFrame(data)

    # Set up experiment comparison figure
    plt.figure()
    plt.title('Mean reward for each experiment')
    plt.xlabel('step')
    plt.ylabel('reward')
    plt.grid()
    plt.tight_layout()
    colors = plt.cm.prism(np.linspace(0,1,len(df['algorithm'].unique())*len(df['set'].unique())))
    plt.gca().set_prop_cycle(cycler('color', colors))

    # Plot all individual experiments
    for algorithm in df['algorithm'].unique():
        for st in df['set'].unique():
            exp_df = df[(df.algorithm == algorithm) & (df.set == st)]
            plt.plot(exp_df.step, exp_df.reward, label=f'{algorithm}_{st}')

    # Set legend to top 5 experiments
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_pairs = sorted(zip(handles, labels), key=lambda t: max(t[0].get_ydata()), reverse=True)
    handles, labels = zip(*sorted_pairs[:5])
    plt.legend(handles, labels, title='Top 5 experiments')

    # Save figure
    if not os.path.exists('plotting/plots'):
        os.makedirs('plotting/plots')
    plt.savefig('plotting/plots/experiment_comparison.png')

    # Plot algorithm comparison figure
    plt.figure()
    plt.title('Mean reward for each algorithm')
    seaborn.lineplot(data=df, x='step', y='reward', hue='algorithm')
    plt.grid()
    plt.tight_layout()
    plt.savefig('plotting/plots/algorithm_comparison.png')
