import os
import argparse
import pygame
import gymnasium as gym
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from src.sim import DroneSimulator
from src.utils import *

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)

    # Parse arguments
    parser = argparse.ArgumentParser()

    # parser.add_argument('--path', type=str, default=r'.\trained_models\wheeled\icra2026_wheeled_env1_3robots_CrossQ.zip', help='The directory to look for trained models in')
    # parser.add_argument('--robot_type', type=str, choices=['uav', 'wheeled_robot'], default='wheeled_robot', help='The device to run the model on')
    # parser.add_argument('--num_robots', type=int, default=3, help='Number of robots')
    parser.add_argument('--path', type=str, default=r'.\trained_models\uav\cont_env3_3robots_CrossQ.zip', help='The directory to look for trained models in')
    parser.add_argument('--robot_type', type=str, choices=['uav', 'wheeled_robot'], default='uav', help='The device to run the model on')
    parser.add_argument('--num_robots', type=int, default=3, help='Number of robots')
    parser.add_argument('--set', type=int, default=3, help='The experiment set to use, from the sets defined in the experiments directory')
    parser.add_argument('--algorithm', type=str, choices=['A2C', 'PPO', 'TRPO', 'ARS', 'CrossQ', 'TQC'], default=['CrossQ'], help='The DRL algorithm to use')    
    parser.add_argument('--simulate', type=parse_bool, default=False, help='If true, uses the Coppelia Simulator to show the environment. If false, renders the environment using PyGame')
    parser.add_argument('--seed', type=int, default=None, help='The random seed to use')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='The device to run the model on')
    
    args = parser.parse_args()

    # Load the model
    model = load_model(algorithm=args.algorithm, seed=args.seed, device=args.device, verbose=False, trained_model_path=args.path)

    # Make the environment
    if args.robot_type == 'wheeled_robot':
        env_config = read_wheeled_config(f'exp_sets/wheeled/env{args.set}.ini')
        env = gym.make('MultiWheeled-v0', render_mode='human', env_params=env_config)
    else:
        env_json = read_uav_json(rf'.\exp_sets\uav\cont_sets.json')[rf'set{args.set}']
        env = gym.make('MultiUAV-v0', render_mode='human', field_info=env_json, num_robots=args.num_robots)

    env.metadata['render_fps'] = 1
    obs, info = env.reset(seed=args.seed)

    # Set up CoppeliaSim
    if args.simulate:
        client = RemoteAPIClient()
        sim = client.getObject('sim')
        defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
        sim.setInt32Param(sim.intparam_idle_fps, 0)

        drone_simulator = DroneSimulator(sim=sim, polygon=env.unwrapped.poly_vertices, scaling_factor=50, height=0.35, num_robots=args.num_robots)
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
        pygame.time.delay(100)
    print('terminated:', terminated, 'truncated:', truncated)
    pygame.time.delay(2000)
    
    # Close simulator and environment
    if args.simulate:
        drone_simulator.stop_simulation()
    env.close()
