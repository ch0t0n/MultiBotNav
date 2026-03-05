import gymnasium as gym
from src.env import MultiUAV, MultiWheeled

# Register environment
gym.register(
    id='MultiUAV-v0', 
    entry_point=MultiUAV,
    max_episode_steps=1000
)

# Register environment
gym.register(
    id='MultiWheeled-v0', 
    entry_point=MultiWheeled,
    max_episode_steps=1000
)