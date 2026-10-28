"""Monte Carlo (textbook section 7.4, first-visit estimation and exploring starts)."""

import torch

from base import ValueAgent, evaluate
from envs.grid_world import GAMMA, MAX_STEP, GridWorld, non_terminal

torch.manual_seed(0)


class MCValue(ValueAgent):
    """First-visit Monte Carlo estimate of V under a fixed policy."""
    def __init__(self, state_size=9, gamma=GAMMA, behaviour=None):
        super().__init__(state_size, gamma=gamma, behaviour=behaviour)

        self.returns = [[] for _ in range(state_size)]

    def train(self, env, episodes=2000, max_step=MAX_STEP):
        for _ in range(episodes):
            # One full episode: states[t] is S_t and rewards[t] is R_{t+1}, so the two
            # lists are already aligned for the backward pass.
            s, _ = env.reset()
            states, rewards = [s], []

            for _ in range(max_step):
                s, r, is_stop, is_cut, _ = env.step(self.behaviour(s))

                states.append(s)
                rewards.append(r)
                if is_stop or is_cut:
                    break

            # Walk backwards, accumulating the return. eq. (7-39).
            # rewards[t] is R_{t+1}, so it pairs with states[t]. Overwriting keeps the
            # earliest visit of each state, which is what "first visit" means: the
            # backward pass sees the late visits first.
            first_return = {}
            running_return = 0.0
            for t in reversed(range(len(rewards))):
                running_return = self.gamma * running_return + rewards[t]
                first_return[states[t]] = running_return

            for s, g in first_return.items():
                self.returns[s].append(g)
                self.value_table[s] = sum(self.returns[s]) / len(self.returns[s])

        return self


class MCControl:
    """Exploring-starts first-visit Monte Carlo control; improves on every visited pair."""
    def __init__(self, state_size=9, action_size=4, gamma=GAMMA, epsilon=0.1):
        self.q_table = torch.zeros(state_size, action_size)
        self.policy_table = torch.zeros(state_size, dtype=torch.long)
        self.returns = {}
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_size = action_size
        self.non_terminal = non_terminal()

    def greedy_action(self, s):
        return int(self.q_table[s].argmax())

    def act(self, s):
        # epsilon-greedy behaviour policy, eq. (7-43).
        if torch.rand(1).item() < self.epsilon:
            return int(torch.randint(self.action_size, (1,)))

        return self.greedy_action(s)

    def policy(self, env):
        return self.greedy_action

    def train(self, env, episodes=3000, max_step=MAX_STEP):
        for _ in range(episodes):
            # Exploring start: any state-action pair may open the episode, so every
            # pair keeps being sampled even while the policy turns greedy.
            env.reset()
            env.s = self.non_terminal[int(torch.randint(len(self.non_terminal), (1,)))]

            states = [env.s]
            actions = [int(torch.randint(self.action_size, (1,)))]
            rewards = []

            for _ in range(max_step):
                s2, r, is_stop, is_cut, _ = env.step(actions[-1])
                states.append(s2)
                rewards.append(r)
                if is_stop or is_cut:
                    break

                actions.append(self.act(s2))

            # Backward pass, first visit of each state-action pair. eq. (7-41).
            # Same overwrite trick as above: the backward pass sees the last visit
            # of each pair first, so keep the earliest one.
            first_return = {}
            running_return = 0.0
            for t in reversed(range(len(rewards))):
                running_return = self.gamma * running_return + rewards[t]
                first_return[(states[t], actions[t])] = running_return

            for pair, g in first_return.items():
                self.returns.setdefault(pair, []).append(g)

                values = self.returns[pair]
                self.q_table[pair] = sum(values) / len(values)
                self.policy_table[pair[0]] = int(self.q_table[pair[0]].argmax())

        return self


if __name__ == "__main__":
    value_agent = MCValue().train(GridWorld("slip"), episodes=2000)
    print("MC value estimation, uniform random policy, 2000 episodes")
    print("V(s) =", [round(v, 2) for v in value_agent.value_table.tolist()])

    control_agent = MCControl(epsilon=0.1).train(GridWorld("slip"), episodes=3000)
    average_return, win_rate = evaluate(control_agent.policy(GridWorld("slip")), GridWorld("slip"), episodes=100)

    print("MC control, exploring starts, epsilon = 0.1, 3000 episodes")
    print(f"greedy policy over 100 episodes: return {average_return:.2f}, win rate {win_rate:.0%}")
    print("policy(s) =", control_agent.q_table.argmax(1).tolist())
