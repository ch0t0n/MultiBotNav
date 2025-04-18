import argparse
import pygame
import gymnasium as gym
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from src.sim import DroneSimulator
from src.utils import load_experiment, load_model, parse_bool

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type='str', required=True, help='The directory to look for trained models in')
    parser.add_argument('--algorithm', type=str, required=True, choices=['A2C', 'PPO', 'TRPO', 'ARS', 'CrossQ', 'TQC'], help='The DRL algorithm to use')    
    parser.add_argument('--set', type=int, required=True, help='The experiment set to use, from the sets defined in the experiments directory')
    parser.add_argument('--simulate', type=parse_bool, default=False, help='If true, uses the Coppelia Simulator to show the environment. If false, renders the environment using PyGame')
    parser.add_argument('--seed', type=int, default=None, help='The random seed to use')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='The device to run the model on')

    args = parser.parse_args()

    # Load the model
    model = load_model(args.algorithm, args.set, args.seed, args.device, args.path)

    # Make the environment
    env_config = load_experiment(f'experiments/set{args.set}.yaml')
    env = gym.make('MultiRobotEnv-v0', seed=args.seed, render_mode='human', env_config=env_config)
    env.metadata['render_fps'] = 1
    obs, info = env.reset(seed=args.seed)

    # Set up CoppeliaSim
    if args.simulate:
        client = RemoteAPIClient()
        sim = client.getObject('sim')
        defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
        sim.setInt32Param(sim.intparam_idle_fps, 0)

        drone_simulator = DroneSimulator(sim=sim, polygon=env.unwrapped.poly_vertices, scaling_factor=50, height=0.35)
        drone_simulator.draw_field()
        drone_simulator.set_agent_positions(info=info)
        drone_simulator.set_weed_locations(weed_locations=env.unwrapped.initial_inf_locations)
        drone_simulator.start_simulation()

    # Run trained model
    terminated, truncated = False, False
    total_rewards = 0
    while not (terminated or truncated):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(list(action))
        if args.simulate:
            drone_simulator.move_agents(info=info)
        else:
            env.render()
            pygame.event.get()
        total_rewards += reward
        print(f"Obs: {obs}, Reward: {reward}, terminated: {terminated}, total_rewards: {total_rewards}, action: {action}")
    print('terminated:', terminated, 'truncated:', truncated)
    
    # Close simulator and environment
    if args.simulate:
        drone_simulator.stop_simulation()
    env.close()
