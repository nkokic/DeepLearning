import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO

gym.register_envs(gymnasium_robotics)

env = gym.make("HandManipulateBlockRotateZ-v1", render_mode="human")

model = PPO.load("models/ppo_hand_rotate_z", env=env)

obs, info = env.reset()

for _ in range(3000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
