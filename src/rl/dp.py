"""Dynamic programming (textbook section 7.3, algorithm "iterative policy evaluation")."""

from envs.grid_world import GAMMA, GridWorld, non_terminal

THETA = 1e-6
STATE_SIZE, ACTION_SIZE = 9, 4


def q_from_v(env, value, s, a, gamma=GAMMA):
    # eq. (7-28): q(s,a) = sum_{s',r} p(s',r|s,a) (r + gamma * V(s'))
    return sum(prob * (r + gamma * value[s2]) for (s2, r), prob in env.p_next_r(s, a).items())


def policy_evaluation(env, policy, value, theta=THETA, gamma=GAMMA):
    # In-place iteration until the largest change drops below theta.
    while True:
        delta = 0.0

        for s in non_terminal():
            old = value[s]
            value[s] = q_from_v(env, value, s, policy[s], gamma)
            delta = max(delta, abs(old - value[s]))

        if delta < theta:
            return value


def policy_improvement(env, policy, value, gamma=GAMMA):
    # eq. (7-31): pi'(s) = argmax_a q(s,a); eq. (7-33) is the greedy form.
    new_policy = list(policy)
    stable = True

    for s in non_terminal():
        q = [q_from_v(env, value, s, a, gamma) for a in range(ACTION_SIZE)]

        new_policy[s] = max(range(ACTION_SIZE), key=q.__getitem__)
        stable = stable and new_policy[s] == policy[s]

    return new_policy, stable


def policy_iteration(env, theta=THETA, gamma=GAMMA):
    # Alternate evaluation and improvement until the policy stops changing.
    value, policy, rounds = [0.0] * STATE_SIZE, [0] * STATE_SIZE, 0

    while True:
        rounds += 1

        policy_evaluation(env, policy, value, theta, gamma)
        policy, stable = policy_improvement(env, policy, value, gamma)

        if stable:
            return policy, value, rounds


def value_iteration(env, theta=THETA, gamma=GAMMA):
    # eq. (7-23): fold improvement into evaluation by taking the max over actions.
    value, rounds = [0.0] * STATE_SIZE, 0

    while True:
        rounds += 1
        delta = 0.0

        for s in non_terminal():
            old = value[s]
            value[s] = max(q_from_v(env, value, s, a, gamma) for a in range(ACTION_SIZE))
            delta = max(delta, abs(old - value[s]))

        if delta < theta:
            return value, rounds


if __name__ == "__main__":
    for scenario in ("deterministic", "slip"):
        env = GridWorld(scenario)

        policy, value, pi_rounds = policy_iteration(env)
        value_from_vi, vi_rounds = value_iteration(env)
        policy_from_vi, _ = policy_improvement(env, [0] * STATE_SIZE, value_from_vi)

        assert policy == policy_from_vi

        print(f"[{scenario}] policy iteration {pi_rounds} rounds, value iteration {vi_rounds} rounds")
        print(f"[{scenario}] optimal policy {policy}, V(start) = {value[0]:.2f}")
