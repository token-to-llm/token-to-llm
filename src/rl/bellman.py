"""Bellman equations on the grid world (textbook section 7.1.5, eq. (7-16) to (7-23))."""

import torch

from envs.grid_world import BOMB, DOWN, GAMMA, RIGHT, GridWorld, non_terminal

torch.manual_seed(0)

STATE = 4  # (2,2)


def format_state(s):
    return f"({s // 3 + 1},{s % 3 + 1})"


def q_from_v(env, value, s):
    # eq. (7-18): q(s,a) = r(s,a) + gamma * sum_s' p(s'|s,a) v(s')
    return torch.tensor([env.reward(s, a) + GAMMA * sum(p * value[s2] for s2, p in env.p_next(s, a).items())
                         for a in range(4)])


def solve(env, policy=None):
    # Repeated substitution of eq. (7-19); policy=None uses the max of eq. (7-23).
    value = torch.zeros(9)
    for _ in range(200):
        for s in non_terminal():
            q = q_from_v(env, value, s)
            value[s] = float(q.max()) if policy is None else float((policy * q).sum())

    return value


def evaluate_policy(env, actions):
    # Expected return of a deterministic policy, evaluated on the true model.
    value = torch.zeros(9)
    for _ in range(200):
        for i, s in enumerate(non_terminal()):
            value[s] = q_from_v(env, value, s)[actions[i]]

    return value


def sample_episode(env):
    # eq. (7-6): tau = (S1, A1, R2, S2, A2, R3, S3). Indices start at 1 and the reward
    # lags the action by one step -- R_{t+1} is what the environment returns after A_t.
    env.reset()
    env.s = STATE

    trace = [f"S1={format_state(STATE)}"]
    running_return = 0.0
    for t, a in enumerate([DOWN, RIGHT], start=1):
        s2, r, is_stop, _, _ = env.step(a)
        trace += [f"A{t}={'down' if a == DOWN else 'right'}", f"R{t + 1}={r:+.0f}", f"S{t + 1}={format_state(s2)}"]
        running_return = r + GAMMA * running_return

    return ", ".join(trace), running_return


if __name__ == "__main__":
    env = GridWorld("slip")

    joint = env.p_next_r(STATE, RIGHT)
    marginal = env.p_next(STATE, RIGHT)
    print(f"eq. (7-3) to (7-5),  s = {format_state(STATE)}, a = right (slip 0.1)")
    print("  p(s',r|s,a) = " + ", ".join(f"{format_state(s2)},{r:+.0f}:{p:.2f}" for (s2, r), p in joint.items()))
    print("  p(s'|s,a)   = " + ", ".join(f"{format_state(s2)}:{p:.2f}" for s2, p in marginal.items()))
    print(f"  r(s,a)      = {env.reward(STATE, RIGHT):+.2f}")

    trace, running_return = sample_episode(GridWorld("deterministic"))
    print(f"eq. (7-6),  deterministic, from {format_state(STATE)}:  tau = ({trace}), G1 = {running_return:+.2f}")

    uniform = torch.full((4,), 0.25)
    value, value_star = solve(env, uniform), solve(env)
    q, q_star = q_from_v(env, value, STATE), q_from_v(env, value_star, STATE)
    print(f"eq. (7-17)/(7-18),  uniform policy: q(2,2,.) = {[round(x, 2) for x in q.tolist()]}, "
          f"v(2,2) = {float((uniform * q).sum()):.3f}, v(start) = {value[0]:.3f}")
    print(f"eq. (7-22)/(7-23),  optimal: q*(2,2,.) = {[round(x, 2) for x in q_star.tolist()]}, "
          f"v*(2,2) = {float(q_star.max()):.3f}, v*(start) = {value_star[0]:.3f}")

