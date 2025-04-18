from shapely import Polygon
from shapely.geometry import Point
from scipy.spatial import ConvexHull, distance
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import argparse

if __name__ == '__main__':

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('number', type=int, nargs='?', default=1, help='The number of experiments to generate')
    parser.add_argument('--max_size', type=int, default=50, help='The maximum allowed size of the field')
    args = parser.parse_args()

    if args.max_size > 100:
        raise ValueError('Max size cannot be greater than 100')

    # Generate the specified number of experiments
    for _ in range(args.number):
        experiment = {'grid_size': args.max_size}

        response = 'n'
        while response.lower() != 'y':
            # Generate field
            num_points = np.random.randint(5,10)
            x = [np.random.randint(0, args.max_size) for i in range(num_points)]
            y = [np.random.randint(0, args.max_size) for i in range(num_points)]
            vertices = list(set(zip(x,y)))
            hull = ConvexHull(vertices)
            conv_vertices = [vertices[i] for i in hull.vertices]
            field_poly = Polygon(conv_vertices)
            experiment['field'] = conv_vertices

            # Plot field to verify that it's correct
            plt.plot(*field_poly.exterior.xy)
            plt.show()

            response = input('Does the field look good (y/n)? ')

        # Generate initial drone positions
        experiment['init_positions'] = []
        while len(experiment['init_positions']) < 3:
            new_point = (np.random.randint(0, args.max_size), np.random.randint(0, args.max_size))
            if field_poly.contains(Point(new_point)) and all(distance.euclidean(new_point, p) >= 5 for p in experiment['init_positions']):
                experiment['init_positions'].append(new_point)

        # Generate weed locations
        num_weeds = np.random.randint(5,10)
        experiment['infected_locations'] = []
        while len(experiment['infected_locations']) < num_weeds:
            new_point = (np.random.randint(0, args.max_size), np.random.randint(0, args.max_size))
            if field_poly.contains(Point(new_point)) and new_point not in experiment['infected_locations']:
                experiment['infected_locations'].append(new_point)

        # Save as yaml file
        file = lambda x: f'experiments/set{x}.yaml'
        counter = 1
        while os.path.exists(file(counter)):
            counter += 1
        with open(file(counter), 'w') as save_file:
            yaml.dump(experiment, save_file)
