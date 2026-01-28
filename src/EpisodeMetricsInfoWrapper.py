# src/episode_metrics_wrapper.py
from __future__ import annotations

import gymnasium as gym
from typing import Any, Dict, Tuple

"""
Makes info["episode_metrics"] have all episodes info,
including TimeLimit truncations (which happen in wrappers, not inside the env).
so we know when time limit is hit (metrics might not show episodes that hit the time limit)
    
how to use:
from src.EpisodeMetricsInfoWrapper import EpisodeMetricsInfoWrapper

wherever you use make_vec_env:
wrapper_class=EpisodeMetricsInfoWrapper,  
"""

class EpisodeMetricsInfoWrapper(gym.Wrapper):
    def step(self, action) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            base = self.env.unwrapped
            if hasattr(base, "_get_episode_metrics"):
                info = dict(info)
                info.setdefault("episode_metrics", base._get_episode_metrics())
        return obs, reward, terminated, truncated, info
