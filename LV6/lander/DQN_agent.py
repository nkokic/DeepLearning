import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        batch_size=8,
        learning_rate=0.001,
        epsilon_start=0.95,
        epsilon_min=0.05,
        epsilon_decay=0.999,
        gamma=0.99,
        buff_capacity=500000,
        update_target_steps=1000,
        seed=None,
    ):
        """Inicijalizira DQN agenta (hiperparametri, replay buffer, mreže, optimizer i brojače)."""
        self.agent_type = "dqn"
        self.state_size = state_size
        self.action_size = action_size

        # (Opcionalno) reproducibilnost
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Replay buffer: sprema prijelaze (s,a,r,s',done)
        self.memory = deque(maxlen=buff_capacity)

        # Osnovni hiperparametri RL-a i treniranja
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_target_steps = update_target_steps

        # Online mreža (uči) i target mreža (stabilizira učenje)
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        # Optimizer i funkcija gubitka za Q-vrijednosti
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Brojač koraka treniranja (za periodično ažuriranje target mreže)
        self.train_step = 0

    def build_model(self):
        """Gradi neuronsku mrežu koja aproksimira Q(s,a) za sve akcije."""
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.action_size),
        )
        return model

    def update_target_model(self):
        """Hard update: kopira težine online mreže u target mrežu."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done, episode_index=None):
        """Sprema iskustvo (prijelaz) u replay buffer za kasnije uzorkovanje."""
        self.memory.append((state, action, reward, next_state, done, episode_index))

    def act(self, state):
        """Odabire akciju epsilon-greedy politikom (istraživanje vs. eksploatacija)."""
        if np.random.rand() <= self.epsilon:
            return random.choice(np.arange(self.action_size))

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def act_exploit(self, state):
        """Odabire akciju pohlepno (argmax Q), bez istraživanja (epsilon=0)."""
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def decay_epsilon(self):
        """Smanjuje epsilon nakon epizode (postupno manje nasumičnih akcija)."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def train(self):
        """Uzorkuje minibatch iz buffera i radi jedan DQN update (TD target + backprop)."""
        if len(self.memory) < self.batch_size:
            return None

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.vstack([exp[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch], dtype=np.float32)
        next_states = np.vstack([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch], dtype=np.float32)

        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32)

        # Q(s,a) za odigrane akcije
        q_all = self.model(states_t)
        q_sa = q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # TD target: r + gamma * max_a' Q_target(s',a') * (1-done)
        with torch.no_grad():
            next_q_max = self.target_model(next_states_t).max(dim=1)[0]
            target = rewards_t + (self.gamma * next_q_max * (1.0 - dones_t))

        # Gubitak: razlika između predikcije i cilja
        loss = self.criterion(q_sa, target)

        # Korak optimizacije
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodično ažuriranje target mreže
        self.train_step += 1
        if self.train_step % self.update_target_steps == 0:
            self.update_target_model()

        return loss.item()

    def save_model(self, filename):
        """Sprema parametre online mreže na disk (PyTorch state_dict)."""
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        """Učitava parametre online mreže i sinkronizira target mrežu."""
        self.model.load_state_dict(torch.load(filename, map_location="cpu"))
        self.update_target_model()
