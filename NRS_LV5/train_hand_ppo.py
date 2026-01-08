from pathlib import Path
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Registracija robotics okruženja
gym.register_envs(gymnasium_robotics)

ENV_ID = "HandManipulateBlockRotateZ-v1"
TOTAL_TIMESTEPS = 1_000_000

LOG_DIR = Path("logs") / "ppo_hand"
MODEL_DIR = Path("models")

LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Okruženje
env = gym.make(ENV_ID)
env = Monitor(env)

# PPO model
model = PPO(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=4096,          # veći batch zbog kompleksnosti
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.001,       # malo entropije pomaže istraživanju
    policy_kwargs=dict(net_arch=[256, 256, 256]),
    verbose=1,
    tensorboard_log=str(LOG_DIR),
)

# Učenje
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    progress_bar=True
)

# Spremanje
model_path = MODEL_DIR / "ppo_hand_rotate_z"
model.save(model_path)

env.close()
print("PPO treniranje gotovo.")
