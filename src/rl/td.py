"""Tabular TD(0) policy evaluation (textbook section 7.5.1, algorithm "tabular TD(0)")."""

from base import ValueAgent, evaluate, run_with_cli
from envs.grid_world import GridWorld

import torch

torch.manual_seed(0)


class TD0(ValueAgent):
    def update(self, s, r, s2):
        # eq. (7-44): V(S) <- V(S) + alpha (R + gamma V(S') - V(S)).
        # The target bootstraps from V(S') instead of waiting for the return,
        # which is what separates TD from Monte Carlo.
        td_target = r + self.gamma * self.value_table[s2]
        td_error = td_target - self.value_table[s]

        self.value_table[s] += self.alpha * td_error


if __name__ == "__main__":
    env, agent, (average_return, win_rate) = run_with_cli(TD0(alpha=0.1), scenario="slip")

    print("TD(0) on scenario 'slip', alpha = 0.1, 500 episodes")
    print("V(s) =", [round(v, 2) for v in agent.value_table.tolist()])
    print(f"one-step lookahead on V, 100 episodes: return {average_return:.2f}, win rate {win_rate:.0%}")

    # Smaller step size over more episodes: the estimate settles closer to v_pi.
    long_env = GridWorld("slip")
    long_agent = TD0(alpha=0.02).train(long_env, episodes=20000)
    long_return, long_win = evaluate(long_agent.policy(long_env), long_env, episodes=100)

    print("TD(0) on 'slip', alpha = 0.02, 20000 episodes")
    print("V(s) =", [round(v, 2) for v in long_agent.value_table.tolist()])
    print(f"one-step lookahead on V, 100 episodes: return {long_return:.2f}, win rate {long_win:.0%}")
