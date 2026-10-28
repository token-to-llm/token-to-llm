"""Why comparing actions through V is harder (textbook section 7.3, eq. (7-30)/(7-31))."""

import torch

from envs.grid_world import DOWN, GAMMA, LEFT, RIGHT, UP, GridWorld, non_terminal

torch.manual_seed(0)

ACTION_NAME = {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}


def format_state(s):
    return f"({s // 3 + 1},{s % 3 + 1})"


def q_from_v(env, value, s):
    # eq. (7-30): one step of lookahead needs r(s,a) and p(s'|s,a) for every action.
    return torch.tensor([env.reward(s, a) + GAMMA * sum(p * value[s2] for s2, p in env.p_next(s, a).items())
                         for a in range(4)])


def solve(env):
    value = torch.zeros(9)
    for _ in range(200):
        for s in non_terminal():
            value[s] = float(q_from_v(env, value, s).max())

    return value


def greedy_actions(env, value):
    return {s: int(q_from_v(env, value, s).argmax()) for s in non_terminal()}


def evaluate_policy(env, actions):
    value = torch.zeros(9)
    for _ in range(200):
        for s in non_terminal():
            value[s] = q_from_v(env, value, s)[actions[s]]

    return value


def average(value):
    return float(value[non_terminal()].mean())


if __name__ == "__main__":
    true_env = GridWorld("wind")
    value_star = solve(true_env)
    action_star = greedy_actions(true_env, value_star)

    # A model that ignores the wind: the planner believes the transition is deterministic.
    wrong_env = GridWorld("deterministic")
    action_wrong = greedy_actions(wrong_env, solve(wrong_env))

    print("comparing actions through V needs the model, comparing through Q does not")
    print(f"  model size: p(s'|s,a) has {len(non_terminal()) ** 2 * 4} entries, "
          f"Q has {len(non_terminal()) * 4}")
    print("  cost per state: V needs 4 actions x 3 successors = 12 multiply-adds, Q needs 3 comparisons")
    print()

    print(f"scenario 'wind', v*(start) = {value_star[0]:.3f}")
    for s in non_terminal():
        mark = "" if action_star[s] == action_wrong[s] else "   <- wrong action"
        print(f"  s={format_state(s)}  correct a={ACTION_NAME[action_star[s]]:<5} "
              f"model-wrong a={ACTION_NAME[action_wrong[s]]:<5}{mark}")

    wrong_states = [s for s in non_terminal() if action_star[s] != action_wrong[s]]
    lost = evaluate_policy(true_env, action_wrong)

    print()
    print(f"model error: {len(wrong_states)}/7 states pick the wrong action "
          f"{[format_state(s) for s in wrong_states]}")
    print(f"  average v falls from {average(value_star):.3f} to {average(lost):.3f}")
