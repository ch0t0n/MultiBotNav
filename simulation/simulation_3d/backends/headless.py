"""Headless simulation loop (no 3D window) for smoke testing."""

from __future__ import annotations


def run_headless(env, model, episodes: int = 3):
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        while True:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                print(
                    f"Episode {ep + 1} | steps: {steps} | reward: {total_reward:.1f} "
                    f"| term_cond: {info.get('term_cond', '?')}"
                )
                break
    env.close()
