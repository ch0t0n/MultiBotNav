import glob
import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import pandas as pd

if __name__ == '__main__':
    
    print('Generating results table...')
    
    # Table header
    table = \
'''\\begin{table}[tbp]
\\centering
\\caption{All experiment results for each algorithm over all environments}
\\resizebox{\\linewidth}{!}{
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}
\\hline
\\multirow{3}{*}{\\textbf{Algorithm}} & \\multicolumn{4}{c|}{\\textbf{Setting A}} & \\multicolumn{4}{c|}{\\textbf{Setting B}} & \\multicolumn{4}{c|}{\\textbf{Setting C}} \\\\ \\cline{2-13}
 & \\textbf{Mean} & \\textbf{SD} & \\textbf{Max} & \\textbf{Range} & \\textbf{Mean} & \\textbf{SD} & \\textbf{Max} & \\textbf{Range} & \\textbf{Mean} & \\textbf{SD} & \\textbf{Max} & \\textbf{Range} \\\\ 
 & $\\times 10^{6}$ & $\\times 10^{6}$ & $\\times 10^{6}$ & $\\times 10^{6}$  & $\\times 10^{6}$ & $\\times 10^{6}$ & $\\times 10^{6}$ & $\\times 10^{6}$ & $\\times 10^{6}$ & $\\times 10^{6}$ & $\\times 10^{6}$ & $\\times 10^{6}$ \\\\ \\hline'''

    # Gather default hyperparameter data
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
    
    # Gather best hyperparameter data
    tuning_data = []
    tuning_logs = glob.glob("./training_best_logs/*/*")
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
                        'algorithm': algorithm,
                        'set': st,
                        'step': e.step,
                        'reward': v.simple_value
                    })
    transfer_df = pd.DataFrame(transfer_data)
    
    # Add results to table
    for algorithm in sorted(train_df['algorithm'].unique()):
        rewards_a = train_df[train_df['algorithm'] == algorithm]['reward'] / 1e6
        a_mean = rewards_a.mean()
        a_std = rewards_a.std()
        a_max = rewards_a.max()
        a_range = a_max - rewards_a.min()
        
        table += f'\n& {algorithm} '
        table += f'& {a_mean:.3f} '
        table += f'& {a_std:.3f} '
        table += f'& {a_max:.3f} '
        table += f'& {a_range:.3f} '
        
        rewards_b = tune_df[tune_df['algorithm'] == algorithm]['reward'] / 1e6
        b_mean = rewards_b.mean()
        b_std = rewards_b.std()
        b_max = rewards_b.max()
        b_range = b_max - rewards_b.min()
        
        table += f'& {b_mean:.3f} '
        table += f'& {b_std:.3f} '
        table += f'& {b_max:.3f} '
        table += f'& {b_range:.3f} '
        
        rewards_c = transfer_df[transfer_df['algorithm'] == algorithm]['reward'] / 1e6
        c_mean = rewards_c.mean()
        c_std = rewards_c.std()
        c_max = rewards_c.max()
        c_range = c_max - rewards_c.min()
        
        table += f'& {c_mean:.3f} '
        table += f'& {c_std:.3f} '
        table += f'& {c_max:.3f} '
        table += f'& {c_range:.3f} \\\\ \\hline'
    
    # Table footer
    table += \
'''\n\\end{tabular}}
\\label{tab:alg_analysis}
\\end{table}'''

    # Save table to file
    with open('tables/results_table.tex', 'w') as file:
        file.write(table)
