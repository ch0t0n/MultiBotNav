import gymnasium as gym
from src.env import MultiRobotEnv

# Register environment
gym.register(
    id='MultiRobotEnv-v0', 
    entry_point=MultiRobotEnv,
    max_episode_steps=1000
)
