"""Q-learning (textbook section 7.5.3, algorithm "Q-Learning")."""

import torch

from base import TabularAgent, run_with_cli

torch.manual_seed(0)


class QLearning(TabularAgent):
    def update(self, s, a, r, s2, a2):
        # eq. (7-48). The target uses max_a' Q(s',a') instead of the sampled a2,
        # which is what makes this an off-policy method.
        td_target = r + self.gamma * self.q_table[s2].max()
        td_error = td_target - self.q_table[s, a]

        self.q_table[s, a] += self.alpha * td_error


if __name__ == "__main__":
    env, agent, (average_return, win_rate) = run_with_cli(QLearning(alpha=0.1), scenario="slip")

    print("Q-learning on scenario 'slip', 500 episodes")
    print(f"greedy policy over 100 episodes: return {average_return:.2f}, win rate {win_rate:.0%}")
    print("max_a Q(s,a) =", [round(v, 2) for v in agent.q_table.max(1).values.tolist()])
    print("policy(s) =", agent.q_table.argmax(1).tolist())
