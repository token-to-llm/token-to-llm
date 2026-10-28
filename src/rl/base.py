"""Tabular agents and evaluation (textbook section 7.3, generalized policy iteration)."""

import argparse
from functools import partial

import torch

from envs.grid_world import GAMMA, GOAL, MAX_STEP, GridWorld


def parse_render_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="show the grid while training")

    return parser.parse_args().render


def uniform_policy(action_size=4):
    return lambda s: int(torch.randint(action_size, (1,)))


def greedy_from_v(env, value, s):
    # eq. (7-30): comparing actions through V needs the model p(s'|s,a).
    q = [env.reward(s, a) + GAMMA * sum(p * value[s2] for s2, p in env.p_next(s, a).items())
         for a in range(4)]

    return int(torch.tensor(q).argmax())


class TabularAgent:
    """Agent that stores an action value table Q(s,a)."""
    def __init__(self, state_size=9, action_size=4, gamma=GAMMA, alpha=0.5, epsilon=0.1):
        self.q_table = torch.zeros(state_size, action_size)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def greedy_action(self, s):
        return int(self.q_table[s].argmax())

    def act(self, s):
        # epsilon-greedy behaviour policy, eq. (7-43): explore with probability epsilon.
        if torch.rand(1).item() < self.epsilon:
            return int(torch.randint(self.q_table.shape[1], (1,)))

        return self.greedy_action(s)

    def update(self, s, a, r, s2, a2):
        raise NotImplementedError

    def policy(self, env):
        return self.greedy_action

    def train(self, env, episodes=500, max_step=MAX_STEP):
        for _ in range(episodes):
            s, _ = env.reset()
            a = self.act(s)

            for _ in range(max_step):
                s2, r, is_stop, is_cut, _ = env.step(a)
                a2 = self.act(s2)

                self.update(s, a, r, s2, a2)
                s, a = s2, a2

                env.render()
                if is_stop or is_cut:
                    break

        return self


class ValueAgent:
    """Agent that stores a state value table V(s) and estimates it for a fixed policy."""
    def __init__(self, state_size=9, gamma=GAMMA, alpha=0.1, behaviour=None):
        self.value_table = torch.zeros(state_size)
        self.gamma = gamma
        self.alpha = alpha
        self.behaviour = behaviour or uniform_policy()

    def update(self, s, r, s2):
        raise NotImplementedError

    def policy(self, env):
        # Improve the estimate into a greedy policy with one step of lookahead.
        return partial(greedy_from_v, env, self.value_table)

    def train(self, env, episodes=500, max_step=MAX_STEP):
        for _ in range(episodes):
            s, _ = env.reset()

            for _ in range(max_step):
                s2, r, is_stop, is_cut, _ = env.step(self.behaviour(s))

                self.update(s, r, s2)
                s = s2

                env.render()
                if is_stop or is_cut:
                    break

        return self


def evaluate(policy, env, episodes=100, max_step=MAX_STEP):
    total_return = 0.0
    wins = 0

    for _ in range(episodes):
        s, _ = env.reset()

        for t in range(max_step):
            s, r, is_stop, is_cut, _ = env.step(policy(s))
            total_return += GAMMA ** t * r
            if is_stop or is_cut:
                break

        wins += s == GOAL

    return total_return / episodes, wins / episodes


def run_with_cli(agent, scenario="slip"):
    env = GridWorld(scenario, render_mode="human" if parse_render_flag() else None)

    agent.train(env)
    return env, agent, evaluate(agent.policy(env), env)
