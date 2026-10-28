"""SARSA and expected SARSA (textbook section 7.5.2, eq. (7-47) and (7-49))."""

import torch

from base import TabularAgent, run_with_cli

torch.manual_seed(0)


class Sarsa(TabularAgent):
    def update(self, s, a, r, s2, a2):
        # eq. (7-47). The target uses the action a2 actually taken by the behaviour
        # policy, so this is on-policy: the value being learned belongs to epsilon-greedy
        # itself, not to the greedy policy it contains.
        td_target = r + self.gamma * self.q_table[s2, a2]
        td_error = td_target - self.q_table[s, a]

        self.q_table[s, a] += self.alpha * td_error


class ExpectedSarsa(TabularAgent):
    def update(self, s, a, r, s2, a2):
        # eq. (7-49). Replace the sampled a2 by its expectation under epsilon-greedy:
        # (1 - epsilon) max_a' Q + epsilon mean_a' Q. Same fixed point, lower variance.
        epsilon_greedy = torch.full((self.q_table.shape[1],), self.epsilon / self.q_table.shape[1])
        epsilon_greedy[int(self.q_table[s2].argmax())] += 1 - self.epsilon

        td_target = r + self.gamma * (epsilon_greedy * self.q_table[s2]).sum()
        td_error = td_target - self.q_table[s, a]

        self.q_table[s, a] += self.alpha * td_error


if __name__ == "__main__":
    results = {}
    for agent in (Sarsa(alpha=0.1), ExpectedSarsa(alpha=0.1)):
        env, trained, (average_return, win_rate) = run_with_cli(agent, scenario="slip")

        name = type(agent).__name__
        results[name] = trained.q_table.argmax(1).tolist()

        print(f"{name:<15} 'slip', 500 episodes: return {average_return:.2f}, win rate {win_rate:.0%}")
        print(f"{'':<15} policy(s) = {results[name]}")
