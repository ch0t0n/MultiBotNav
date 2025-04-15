import gymnasium as gym
from src.env import MultiBotNavigator

# Register environment
gym.register(id='MultiBotNavigator-v0', 
             entry_point=MultiBotNavigator,
             max_episode_steps=1000)
