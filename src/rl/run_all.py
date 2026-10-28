"""Cross-check every tabular method against the exact solution (textbook sections 7.3 to 7.5).

Control methods are judged on the return of the greedy policy they end up with.
Value-estimation methods are judged on how close V is to the true v_pi of the
behaviour policy they were given.
"""

import torch

from base import uniform_policy
from dp import policy_evaluation, policy_improvement, value_iteration
from envs.grid_world import GAMMA, GridWorld, non_terminal
from mc import MCControl, MCValue
from q_learning import QLearning
from sarsa import ExpectedSarsa, Sarsa
from td_lambda import TDLambda

torch.manual_seed(0)

TOLERANCE = 0.01
EPISODES = {"deterministic": 20000, "slip": 50000, "wind": 20000}


def exact_solution(scenario):
    env = GridWorld(scenario)
    value, policy = [0.0] * 9, [0] * 9

    for _ in range(200):
        policy_evaluation(env, policy, value)
        policy, _ = policy_improvement(env, policy, value)

    value_from_vi, _ = value_iteration(GridWorld(scenario))
    assert all(abs(value[s] - value_from_vi[s]) < 1e-6 for s in non_terminal())

    return env, policy, value[0]


def policy_value(env, actions):
    value = [0.0] * 9
    policy_evaluation(env, actions, value)

    return value[0]


def uniform_policy_value(env):
    # eq. (7-19) with pi(a|s) = 1/4 for every action.
    value = [0.0] * 9
    for _ in range(2000):
        for s in non_terminal():
            value[s] = sum(env.reward(s, a) + GAMMA * sum(p * value[s2] for s2, p in env.p_next(s, a).items())
                           for a in range(4)) / 4

    return value


def main():
    for scenario in ("deterministic", "slip", "wind"):
        env, policy, value_star = exact_solution(scenario)
        episodes = EPISODES[scenario]

        print(f"\n[{scenario}] optimal policy " + " ".join(f"{s}:{policy[s]}" for s in non_terminal())
              + f"  V(start) = {value_star:.3f}")

        print("  control methods: return of the greedy policy, evaluated on the true model")
        control = []

        for name, cls in (("Q-learning", QLearning), ("SARSA", Sarsa), ("expected SARSA", ExpectedSarsa)):
            torch.manual_seed(0)
            control.append((name, cls(alpha=0.1).train(GridWorld(scenario), episodes=episodes).q_table))

        torch.manual_seed(0)
        control.append(("MC control", MCControl(epsilon=0.1).train(GridWorld(scenario), episodes=3000).q_table))

        for name, q_table in control:
            actions = q_table.argmax(1).tolist()
            value = policy_value(env, actions)
            mark = "ok" if abs(value - value_star) <= TOLERANCE * abs(value_star) else "off"

            print(f"    {name:<15} {mark:<4} V={value:.3f}  "
                  + " ".join(f"{s}:{actions[s]}" for s in non_terminal()))

        print("  value estimation: max |V - v_pi| against the exact v_pi of the uniform policy")
        targets = uniform_policy_value(env)
        estimates = []

        torch.manual_seed(0)
        estimates.append(("MC estimate", MCValue().train(GridWorld(scenario), episodes=5000).value_table.tolist()))

        for lam in (0.0, 0.5, 1.0):
            torch.manual_seed(0)
            agent = TDLambda(lam=lam, alpha=0.005).train(GridWorld(scenario), episodes=50000)
            estimates.append((f"TD(lambda={lam})", agent.value_table.tolist()))

        for name, value in estimates:
            # A constant step size leaves a small bias, so allow 0.4 -- well below the
            # spread between the scenarios' own values.
            error = max(abs(value[s] - targets[s]) for s in non_terminal())
            mark = "ok" if error < 0.4 else "off"

            print(f"    {name:<17} {mark:<4} max error {error:.3f}  V(start)={value[0]:.3f} vs {targets[0]:.3f}")


if __name__ == "__main__":
    main()
