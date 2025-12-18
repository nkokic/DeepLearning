import gymnasium as gym
from DQN_agent import DQNAgent
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

# Direktorij za pohranu modela
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# Inicijaliziraj okruženje
env = gym.make("CartPole-v1", render_mode="rgb_array")  # ili LunarLander-v3
env_id = env.spec.id
safe_env_id = env_id.replace("/", "-")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# TODO: odredite vrijednosti hiperparametara
agent = DQNAgent(
    state_size,
    action_size,
    batch_size=128,
    learning_rate=0.0003,
    epsilon_start=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    gamma=0.99,
    buff_capacity=10000,
    update_target_steps=50,
)

num_episodes = 500

writer = SummaryWriter()

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0.0
    episode_loss_sum = 0.0
    loss_steps = 0

    while True:
        action = agent.act(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.remember(state, action, reward, next_state, done, episode)

        # Treniraj ako ima dovoljno uzoraka
        if len(agent.memory) >= agent.batch_size:
            loss = agent.train()
            if loss is not None:
                episode_loss_sum += loss
                loss_steps += 1

        state = next_state
        total_reward += reward

        for value in state:
            total_reward -= abs(value) * 0.01

        if done:
            break

    agent.decay_epsilon()

    writer.add_scalar("Epsilon", agent.epsilon, episode)
    writer.add_scalar("Total Reward", total_reward, episode)

    if loss_steps > 0:
        writer.add_scalar("Average Loss", episode_loss_sum / loss_steps, episode)

    print(f"Episode {episode + 1}/{num_episodes} | Total Reward: {total_reward}")

writer.close()
env.close()

# Pohrani model s vremenskom oznakom
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_name = f"{agent.agent_type}_{safe_env_id}_final_{current_time}.pth"
final_model_path = os.path.join(save_dir, model_name)
agent.save_model(final_model_path)
print(f"Final model saved with filename: {final_model_path}")