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
    logs = glob.glob("./tuning_logs/*/*")
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

    # Plot algorithm comparison figure
    plt.figure()
    plt.title('Mean reward for each algorithm')
    seaborn.lineplot(data=df, x='step', y='reward', hue='algorithm')
    plt.grid()
    plt.tight_layout()
    plt.savefig('plotting/plots/tuning_comparison.png')
