import gymnasium as gym
import numpy as np
from DQN_agent import DQNAgent

# EDIT HERE
ENV_ID = "LunarLander-v3"

MODEL_PATH = "saved_models\\dqn_LunarLander-v3_final_2025-12-18_10-01-30.pth"
# Mean reward:   226.465
# Std reward:    69.587
# Median reward: 242.680

# MODEL_PATH = "saved_models\\dqn_LunarLander-v3_final_2025-12-18_10-24-34.pth"
# Mean reward:   148.286
# Std reward:    130.337
# Median reward: 204.589


# MODEL_PATH = "saved_models\\dqn_LunarLander-v3_final_2025-12-18_10-50-08.pth"
# Mean reward:   148.286
# Std reward:    130.337
# Median reward: 204.589

# MODEL_PATH = "saved_models\\dqn_LunarLander-v3_final_2026-01-13_17-52-00.pth"
# Mean reward:   131.118
# Std reward:    151.415
# Median reward: 208.559

N_EPISODES = 1000
RENDER = False  # True = vizualizacija, False = bez prozora

SEED = 123

def main():
    env = gym.make(ENV_ID, render_mode="human" if RENDER else None)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    agent.load_model(MODEL_PATH)

    rng = np.random.default_rng(SEED)
    rewards = []

    for ep in range(N_EPISODES):
        state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        total = 0.0

        while not done:
            action = agent.act_exploit(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += float(reward)

        rewards.append(total)
        print(f"Episode {ep+1}/{N_EPISODES} | Total reward: {total:.1f}")

    env.close()

    rewards = np.array(rewards, dtype=np.float32)
    print("\n=== Summary ===")
    print(f"Env: {ENV_ID}")
    print(f"Episodes: {N_EPISODES}")
    print(f"Mean reward:   {rewards.mean():.3f}")
    print(f"Std reward:    {rewards.std():.3f}")
    print(f"Median reward: {np.median(rewards):.3f}")


if __name__ == "__main__":
    main()
