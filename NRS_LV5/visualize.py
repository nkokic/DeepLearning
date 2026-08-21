import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO, A2C, SAC, TD3

gym.register_envs(gymnasium_robotics)

env = gym.make("HandManipulateBlockRotateZ-v1", render_mode="human")

model = SAC.load("models\\sac_her_20260821_132605\\sac_her_hand_rotate_z_final.zip", env=env)

obs, info = env.reset()

for _ in range(3000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
