"""3x3 GridWorld (textbook section 7.1.3, figure 7.2(a))."""

import gymnasium as gym
import torch
from gymnasium import spaces

# Map fixed by the textbook (section 7.1). State s = row * 3 + col, row/col count from 1.
#   (1,1) is the start, (3,3) the goal with reward +10, (1,3) the bomb with reward -5,
#   every other move costs -1. The episode ends on entering the goal or the bomb.
START, BOMB, GOAL = 0, 2, 8
REWARD = [-1.0] * 9
REWARD[GOAL] = 10.0
REWARD[BOMB] = -5.0

# Action ids. A move into a wall keeps the agent in place.
UP, DOWN, LEFT, RIGHT = range(4)
DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))

GAMMA = 0.9
MAX_STEP = 50

# Transition scenarios. They differ only in p(s'|s,a); the reward stays deterministic.
#   "deterministic"  the intended action always succeeds. Used by DP and value iteration
#                    as the clean baseline.
#   "slip"           intended direction with p = 0.8, each perpendicular direction with
#                    p = 0.1. Makes p(s',r|s,a) non-degenerate, so Monte Carlo, TD and
#                    SARSA have something to sample. Side effect worth knowing: the
#                    return-optimal action from (1,2) is "down", whose slip branch walks
#                    into the bomb, so the best policy still fails about 10% of its
#                    episodes. Success rate and return rank policies differently.
#   "wind"           intended direction with p = 0.8, plus p = 0.2 of one extra cell
#                    further down. A model that ignores the wind picks a different,
#                    cheaper-looking action; this is the scenario used to show why
#                    comparing actions through V needs a correct model.
SCENARIOS = ("deterministic", "slip", "wind")
SLIP = 0.1
WIND = 0.2

CELL = ["S", ".", "X", ".", ".", ".", ".", ".", "G"]


class GridWorld(gym.Env):
    def __init__(self, scenario="slip", render_mode=None):
        self.scenario = scenario
        self.render_mode = render_mode
        self.observation_space = spaces.Discrete(9)
        self.action_space = spaces.Discrete(4)

        self.generator = torch.Generator().manual_seed(0)
        self.s = START

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)

        self.s = START
        return self.s, {}

    def step(self, action):
        # Sample (s', r) from the joint distribution p(s',r|s,a), eq. (7-3).
        joint = self.p_next_r(self.s, action)
        weights = torch.tensor(list(joint.values()))
        index = torch.multinomial(weights, 1, generator=self.generator).item()

        self.s = list(joint)[index][0]
        is_stop = self.s in (GOAL, BOMB)

        return self.s, REWARD[self.s], is_stop, False, {}

    def p_next_r(self, s, action):
        # p(s',r|s,a), eq. (7-3). The reward is a function of the landing state,
        # so every next state carries exactly one reward value.
        return {(s2, REWARD[s2]): p for s2, p in self.p_next(s, action).items()}

    def p_next(self, s, action):
        # p(s'|s,a) = sum_r p(s',r|s,a), eq. (7-4).
        dist = {}
        for delta, weight in self._moves(action):
            s2 = self._move(s, delta)
            dist[s2] = dist.get(s2, 0.0) + weight

        return dist

    def reward(self, s, action):
        # r(s,a) = sum_r r * sum_s' p(s',r|s,a), eq. (7-5).
        return sum(p * REWARD[s2] for s2, p in self.p_next(s, action).items())

    def _moves(self, action):
        delta = DELTA[action]

        if self.scenario == "slip":
            perpendicular = (DELTA[LEFT], DELTA[RIGHT]) if delta[1] == 0 else (DELTA[UP], DELTA[DOWN])
            return [(delta, 1 - 2 * SLIP), (perpendicular[0], SLIP), (perpendicular[1], SLIP)]

        if self.scenario == "wind":
            return [(delta, 1 - WIND), ((delta[0] + 1, delta[1]), WIND)]

        return [(delta, 1.0)]

    @staticmethod
    def _move(s, delta):
        row = min(max(s // 3 + delta[0], 0), 2)
        col = min(max(s % 3 + delta[1], 0), 2)

        return row * 3 + col

    def render(self):
        if self.render_mode != "human":
            return

        cells = list(CELL)
        cells[self.s] = "A"
        print("\n".join(" ".join(cells[i:i + 3]) for i in range(0, 9, 3)))


def non_terminal():
    return [s for s in range(9) if s not in (GOAL, BOMB)]


if __name__ == "__main__":
    env = GridWorld()

    for name in SCENARIOS:
        env.scenario = name
        mass = [sum(env.p_next_r(s, a).values()) for s in non_terminal() for a in range(4)]

        assert torch.allclose(torch.tensor(mass), torch.ones(len(mass)))
        print(f"{name:<14} sum of p(s',r|s,a) = 1, r(s=start, right) = {env.reward(START, RIGHT):+.2f}")
