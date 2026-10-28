"""Policy gradient / REINFORCE (textbook section 7.6.3)."""

import copy

import torch
import torch.nn as nn

from envs.cartpole import CartPole

torch.manual_seed(42)
GAMMA = 0.99


def action_distribution(actor_model, states):
    # Categorical over the action ids. Sampling and log_prob both read from it.
    return torch.distributions.Categorical(logits=actor_model(states))


class REINFORCE:
    def __init__(self, env, learning_rate=5e-3):
        self.env = env
        self.actor_model = nn.Sequential(
            nn.Linear(env.observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_size),
        )
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=learning_rate)

    def collect_episode(self, actor_model=None, greedy=False):
        models = actor_model or self.actor_model
        states, actions, rewards = [], [], []

        s = self.env.reset()
        while True:
            distribution = action_distribution(models, s.unsqueeze(0))
            a = distribution.logits.argmax(dim=-1) if greedy else distribution.sample()

            states.append(s)
            actions.append(a)

            s, r, is_stop, is_cut = self.env.step(a.item())
            rewards.append(r)

            if is_stop or is_cut:
                return torch.stack(states), torch.cat(actions), rewards

    @staticmethod
    def discounted_returns(rewards):
        # eq. (7-7): G_t = sum_{k>t} gamma^(k-t-1) R_k
        running_return = 0.0
        returns = []

        for r in reversed(rewards):
            running_return = r + GAMMA * running_return
            returns.append(running_return)

        return torch.tensor(returns[::-1], dtype=torch.float)

    def update(self, states, actions, returns):
        # eq. (7-82): theta <- theta + alpha * G * grad log pi(a|s)
        log_prob = action_distribution(self.actor_model, states).log_prob(actions)
        actor_loss = -(returns * log_prob).sum()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def train(self, episodes=1000):
        for _ in range(episodes):
            states, actions, rewards = self.collect_episode()

            self.update(states, actions, self.discounted_returns(rewards))

    def average_return(self, episodes=10):
        # CartPole gives +1 per surviving step, so the return is the survival length.
        return sum(len(self.collect_episode(greedy=True)[2]) for _ in range(episodes)) / episodes


class REINFORCEWithBaseline(REINFORCE):
    def __init__(self, env, learning_rate=5e-3, critic_learning_rate=5e-3):
        super().__init__(env, learning_rate)

        self.critic_model = nn.Sequential(
            nn.Linear(env.observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=critic_learning_rate)

    def update(self, states, actions, returns):
        baseline = self.critic_model(states).squeeze(-1)
        advantage = (returns - baseline).detach()

        # actor: same policy gradient, with the return replaced by the advantage
        log_prob = action_distribution(self.actor_model, states).log_prob(actions)
        actor_loss = -(advantage * log_prob).sum()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # critic: regress the state value onto the sampled return
        critic_loss = ((returns - self.critic_model(states).squeeze(-1)) ** 2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


class OffPolicyREINFORCE(REINFORCE):
    def __init__(self, env, learning_rate=1e-2, refresh=50):
        super().__init__(env, learning_rate)

        self.behaviour_model = copy.deepcopy(self.actor_model)
        self.refresh = refresh

    def train(self, episodes=1000):
        for episode in range(episodes):
            # keep the log_prob ratio close to 1: resync behaviour policy with the target
            if episode % self.refresh == 0:
                self.behaviour_model.load_state_dict(self.actor_model.state_dict())

            states, actions, rewards = self.collect_episode(actor_model=self.behaviour_model)

            # importance sampling: reweight pi_theta(a|s) / beta(a|s)
            target_log_prob = action_distribution(self.actor_model, states).log_prob(actions)
            behaviour_log_prob = action_distribution(self.behaviour_model, states).log_prob(actions)
            ratio = (target_log_prob - behaviour_log_prob).exp()

            self.update(states, actions, self.discounted_returns(rewards) * ratio.detach())


if __name__ == "__main__":
    for agent in (REINFORCE(CartPole()), REINFORCEWithBaseline(CartPole()), OffPolicyREINFORCE(CartPole())):
        agent.train()
        print(f"{type(agent).__name__:<22} greedy policy, average return {agent.average_return():.0f}")
