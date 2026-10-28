"""Proximal policy optimization / PPO-Clip (textbook section 7.7.3)."""

import copy

import torch
import torch.nn as nn

from base import parse_render_flag
from envs.cartpole import CartPole

torch.manual_seed(0)


class PPO:
    def __init__(self, env, learning_rate=3e-3, critic_learning_rate=3e-3, gamma=0.99, lam=0.95,
                 clip=0.2, epochs=10, batch_size=64, rollout_size=2048, entropy_coef=0.01, kl_target=0.01):
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.rollout_size = rollout_size
        self.entropy_coef = entropy_coef
        self.kl_target = kl_target

        self.actor_model = nn.Sequential(
            nn.Linear(env.observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_size),
        )
        self.critic_model = nn.Sequential(
            nn.Linear(env.observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=critic_learning_rate)

    def action_distribution(self, states, actor_model=None):
        return torch.distributions.Categorical(logits=(actor_model or self.actor_model)(states))

    def collect(self):
        # Fill one batch of experience with the current policy, keeping both S_t and
        # S_{t+1}: the critic needs the successor state to compute the TD error.
        states, actions, rewards, next_states, is_stop = [], [], [], [], []

        s = self.env.reset()
        while len(rewards) < self.rollout_size:
            with torch.no_grad():
                a = int(self.action_distribution(s.unsqueeze(0)).sample())

            s2, r, done, cut = self.env.step(a)

            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s2)
            is_stop.append(float(done))
            s = s2

            if done or cut:
                s = self.env.reset()

        return (torch.stack(states), torch.tensor(actions), torch.tensor(rewards),
                torch.stack(next_states), torch.tensor(is_stop))

    def update(self):
        states, actions, rewards, next_states, is_stop = self.collect()

        with torch.no_grad():
            old_values = self.critic_model(states).squeeze(-1)
            next_values = self.critic_model(next_states).squeeze(-1) * (1 - is_stop)

        # eq. (7-103): GAE, computed in place rather than through gae.py.
        advantages = torch.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] - old_values[t]
            running = delta + self.gamma * self.lam * running * (1 - is_stop[t])
            advantages[t] = running

        targets = advantages + old_values

        old_actor_model = copy.deepcopy(self.actor_model)
        with torch.no_grad():
            old_log_prob = self.action_distribution(states, old_actor_model).log_prob(actions)

        size = len(rewards)
        for _ in range(self.epochs):
            order = torch.randperm(size)
            kl = 0.0

            for start in range(0, size, self.batch_size):
                index = order[start:start + self.batch_size]

                distribution = self.action_distribution(states[index])
                log_prob = distribution.log_prob(actions[index])

                # eq. (7-107): r_t(theta) = pi_theta(a|s) / pi_old(a|s)
                ratio = (log_prob - old_log_prob[index]).exp()

                # eq. (7-109): clipped surrogate objective, plus the entropy bonus.
                clipped = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
                actor_loss = -torch.min(ratio * advantages[index], clipped * advantages[index]).mean()
                actor_loss -= self.entropy_coef * distribution.entropy().mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Value loss targets A + V_old, held fixed inside the inner loop.
                critic_loss = ((self.critic_model(states[index]).squeeze(-1) - targets[index]) ** 2).mean()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                kl += (old_log_prob[index] - log_prob).sum()

            if kl.item() / size > 1.5 * self.kl_target:
                break

        return size

    def average_return(self, episodes=10):
        # CartPole pays +1 per surviving step, so the return is the survival length.
        total = 0
        for _ in range(episodes):
            s = self.env.reset()
            for step in range(500):
                with torch.no_grad():
                    a = int(self.actor_model(s.unsqueeze(0)).argmax())
                s, _, is_stop, is_cut = self.env.step(a)
                if is_stop or is_cut:
                    break
            total += step + 1

        return total / episodes


if __name__ == "__main__":
    agent = PPO(CartPole(render=parse_render_flag()))

    total_steps = 60000
    marks = sorted({total_steps // 8, total_steps // 4, total_steps // 2, 3 * total_steps // 4})

    steps_done, returns, next_mark = 0, {}, 0
    while steps_done < total_steps:
        steps_done += agent.update()

        if next_mark < len(marks) and steps_done >= marks[next_mark]:
            returns[marks[next_mark]] = agent.average_return()
            next_mark += 1

    for step in sorted(returns):
        print(f"step {step}: greedy policy average return {returns[step]:.0f}")
    print(f"end of training ({steps_done} steps): greedy policy average return "
          f"{agent.average_return():.0f} (max 500)")
