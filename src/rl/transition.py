"""Transition probability and reward function (textbook section 7.1.3, eq. (7-3) to (7-5))."""

from envs.grid_world import DOWN, LEFT, RIGHT, UP, GridWorld, non_terminal

STATE = 4  # (2,2)
ACTION_NAME = {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}


def format_state(s):
    return f"({s // 3 + 1},{s % 3 + 1})"


def format_joint(joint):
    return ", ".join(f"{format_state(s2)} with r={r:+.0f}: p={p:.2f}" for (s2, r), p in joint.items())


def format_marginal(marginal):
    return ", ".join(f"{format_state(s2)}: p={p:.2f}" for s2, p in marginal.items())


if __name__ == "__main__":
    env = GridWorld("slip")

    # eq. (7-3): the joint distribution p(s',r|s,a) for one state-action pair.
    print(f"eq. (7-3)  s = {format_state(STATE)}, a = right, scenario 'slip'")
    print(f"  p(s',r|s,a) = {format_joint(env.p_next_r(STATE, RIGHT))}")

    # eq. (7-4): summing the joint over r leaves the transition probability p(s'|s,a).
    # In 'slip' the agent moves right 80% of the time and slides to the perpendicular
    # cells 10% each, so three landing cells carry 0.80 / 0.10 / 0.10.
    print(f"eq. (7-4)  p(s'|s,a) = {format_marginal(env.p_next(STATE, RIGHT))}")
    print(f"  probabilities sum to {sum(env.p_next(STATE, RIGHT).values()):.2f}")

    # eq. (7-5): r(s,a) is a scalar, the probability-weighted mean of the rewards.
    # The reward depends only on the landing cell, so moving right from (1,2) hits the
    # bomb with p = 0.8 and averages 0.8 * (-5) + 0.2 * (-1) = -4.20.
    print("eq. (7-5)  r(s,a) = sum_r r * sum_s' p(s',r|s,a)")
    for s, a in [(STATE, RIGHT), (1, RIGHT), (7, DOWN), (3, UP)]:
        print(f"  r({format_state(s)}, {ACTION_NAME[a]}) = {env.reward(s, a):+.2f}")

    # Unlike the old implementation there is no reward jitter: r(s,a) is deterministic,
    # which is what lets the Bellman equations be written with a plain r(s,a).
    print(f"  total probability over all (s,a) = "
          f"{max(abs(sum(env.p_next_r(s, a).values()) - 1) for s in non_terminal() for a in range(4)):.2e} deviation")
