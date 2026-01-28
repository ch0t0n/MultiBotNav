import gymnasium as gym
from src.env import MultiRobotEnv
from src.env_v2 import MultiRobotEnv_v2

# Register environment
gym.register(
    id='MultiRobotEnv-v0', 
    entry_point=MultiRobotEnv,
    max_episode_steps=1000
)

# Register environment
gym.register(
    id='MultiRobotEnv-v2', 
    entry_point=MultiRobotEnv_v2,
    max_episode_steps=1000
)