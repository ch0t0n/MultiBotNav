import matplotlib.pyplot as plt
from shapely import Polygon

# Weird Python hackery to get the last import to work
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from src.utils import load_experiment

if __name__ == '__main__':

    plt.figure(figsize=(25,10))

    # Plot each field as a subfigure
    for i in range(1, 11):
        exp = load_experiment(f'experiments/set{i}.yaml')
        field_poly = Polygon(exp['field'])
        plt.subplot(2, 5, i)
        plt.plot(*field_poly.exterior.xy)
        plt.scatter(*zip(*exp['init_positions']), color='red', marker='s')
        plt.scatter(*zip(*exp['infected_locations']), color='green')
        plt.title(f'Set {i}')
        
    plt.savefig('plotting/plots/all_fields.png', bbox_inches='tight')       
        