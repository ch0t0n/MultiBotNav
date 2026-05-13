import gymnasium as gym
from src.env import MultiUAV, MultiWheeled

if "MultiUAV-v0" not in gym.envs.registry:
    gym.register(id="MultiUAV-v0", entry_point=MultiUAV, max_episode_steps=1000)

if "MultiWheeled-v0" not in gym.envs.registry:
    gym.register(id="MultiWheeled-v0", entry_point=MultiWheeled, max_episode_steps=1000)
