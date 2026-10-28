"""Actor-critic (textbook section 7.7.1, algorithm "actor-critic")."""

import torch
import torch.nn as nn

from base import parse_render_flag
from envs.cartpole import CartPole

torch.manual_seed(0)
GAMMA = 0.99


class ActorCritic:
    def __init__(self, env, actor_learning_rate=5e-3, critic_learning_rate=5e-3):
        self.env = env

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
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=critic_learning_rate)

    def action_distribution(self, states):
        return torch.distributions.Categorical(logits=self.actor_model(states))

    def act(self, s, greedy=False):
        distribution = self.action_distribution(s.unsqueeze(0))

        return int(distribution.logits.argmax()) if greedy else int(distribution.sample())

    def value(self, s):
        return self.critic_model(s.unsqueeze(0)).squeeze(-1)

    def train(self, episodes=2000):
        for _ in range(episodes):
            states, actions, td_errors, discounts = [], [], [], []

            s, discount = self.env.reset(), 1.0
            while True:
                a = self.action_distribution(s.unsqueeze(0)).sample()
                s2, r, is_stop, is_cut = self.env.step(a.item())

                # eq. (7-95): delta = R + gamma v(S';W) - v(S;W). A terminal state has no
                # value, so the bootstrap term drops out there.
                next_value = 0.0 if is_stop else self.value(s2)
                delta = torch.tensor(r) + GAMMA * next_value - self.value(s)

                states.append(s)
                actions.append(a)
                td_errors.append(delta.detach())
                discounts.append(discount)

                discount *= GAMMA
                s = s2
                if is_stop or is_cut:
                    break

            states, actions = torch.stack(states), torch.cat(actions)
            td_errors = torch.cat(td_errors).detach()
            discounts = torch.tensor(discounts)

            # Gradients of the per-step updates are accumulated and applied once per
            # episode. Parameters do not change inside an episode, so this is the same
            # update as the per-step form in the pseudocode.
            # eq. (7-95): actor <- actor + beta^theta * I * delta * grad log pi
            actor_loss = -(discounts * td_errors * self.action_distribution(states).log_prob(actions)).sum()

            # eq. (7-95): critic <- critic + beta^W * delta * grad v
            critic_loss = -(td_errors * self.critic_model(states).squeeze(-1)).sum()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def average_return(self, episodes=10):
        # CartPole pays +1 per surviving step, so the return is the survival length.
        total = 0
        for _ in range(episodes):
            s = self.env.reset()
            while True:
                s, _, is_stop, is_cut = self.env.step(self.act(s, greedy=True))
                total += 1
                if is_stop or is_cut:
                    break

        return total / episodes


if __name__ == "__main__":
    agent = ActorCritic(CartPole(render=parse_render_flag()))

    before = agent.average_return()
    agent.train(2000)
    after = agent.average_return()

    print(f"greedy policy before training: average return {before:.0f}")
    print(f"greedy policy after training:  average return {after:.0f} (max 500)")
