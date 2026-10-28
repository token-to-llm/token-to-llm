"""Deep Q-network / DQN (textbook section 7.5.5, algorithm "DQN")."""

import copy

import torch
import torch.nn as nn

from base import parse_render_flag
from envs.cartpole import CartPole

torch.manual_seed(0)


class DQN:
    def __init__(self, env, learning_rate=1.5e-4, gamma=0.99, batch_size=64,
                 buffer_size=50000, target_sync=500, epsilon_decay=10000):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.target_sync = target_sync
        self.epsilon_decay = epsilon_decay

        self.q_network = nn.Sequential(
            nn.Linear(env.observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_size),
        )
        self.target_network = copy.deepcopy(self.q_network)
        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer: keeps past transitions so one sample is reused many times.
        self.buffer_states = torch.zeros(buffer_size, env.observation_size)
        self.buffer_actions = torch.zeros(buffer_size, 1, dtype=torch.long)
        self.buffer_rewards = torch.zeros(buffer_size, 1)
        self.buffer_next_states = torch.zeros(buffer_size, env.observation_size)
        self.buffer_is_stop = torch.zeros(buffer_size, 1)

        self.size = 0
        self.updates = 0

    def act(self, s, epsilon):
        # epsilon-greedy behaviour policy, eq. (7-43).
        if torch.rand(1).item() < epsilon:
            return int(torch.randint(self.env.action_size, (1,)))

        with torch.no_grad():
            return int(self.q_network(s.unsqueeze(0)).argmax())

    def remember(self, s, a, r, s2, is_stop):
        index = self.size % self.buffer_size

        self.buffer_states[index] = s
        self.buffer_actions[index] = a
        self.buffer_rewards[index] = r
        self.buffer_next_states[index] = s2
        self.buffer_is_stop[index] = float(is_stop)

        self.size += 1

    def update(self):
        # Uniform sample from the buffer, as in eq. (7-51).
        index = torch.randint(min(self.size, self.buffer_size), (self.batch_size,))

        states = self.buffer_states[index]
        actions = self.buffer_actions[index]
        rewards = self.buffer_rewards[index]
        next_states = self.buffer_next_states[index]
        is_stop = self.buffer_is_stop[index]

        with torch.no_grad():
            # The target network supplies max_a' Q(s',a'); is_stop cuts the bootstrap.
            next_value = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + self.gamma * next_value * (1 - is_stop)

        # eq. (7-51): mean squared TD error on the sampled batch.
        prediction = self.q_network(states).gather(dim=1, index=actions)
        loss = ((target - prediction) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.updates += 1
        if self.updates % self.target_sync == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def evaluate_return(self, episodes=10):
        # CartPole pays +1 per surviving step, so the return is the survival length.
        total = 0
        for _ in range(episodes):
            s = self.env.reset()
            for step in range(500):
                s, _, is_stop, is_cut = self.env.step(self.act(s, 0.0))
                if is_stop or is_cut:
                    break
            total += step + 1

        return total / episodes

    def train(self, steps=40000, marks=()):
        s = self.env.reset()
        returns = {}

        for t in range(1, steps + 1):
            epsilon = max(0.05, 1.0 + (0.05 - 1.0) * t / self.epsilon_decay)
            a = self.act(s, epsilon)
            s2, r, is_stop, is_cut = self.env.step(a)

            # Only a real stop may cut the bootstrap; hitting the step limit is a cut,
            # and the next state is still alive. eq. (7-51).
            self.remember(s, a, r, s2, is_stop)
            s = s2

            if self.size >= self.batch_size:
                self.update()

            if is_stop or is_cut:
                s = self.env.reset()

            if t in marks:
                returns[t] = self.evaluate_return()
                s = self.env.reset()

        return returns


if __name__ == "__main__":
    agent = DQN(CartPole(render=parse_render_flag()))
    returns = agent.train(40000, (10000, 20000))

    print(f"step 10000: greedy policy average return {returns[10000]:.0f}")
    print(f"step 20000: greedy policy average return {returns[20000]:.0f}")
    print(f"end of training: greedy policy average return {agent.evaluate_return(10):.0f} (max 500)")
