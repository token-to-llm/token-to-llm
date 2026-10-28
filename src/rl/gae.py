"""Generalized advantage estimation, pure functions (textbook section 7.8.2, eq. (7-101) to (7-103))."""

import torch

torch.manual_seed(0)


def deltas(rewards, values, gamma, is_stop):
    # eq. (7-102): delta_t = R_t + gamma V(S_{t+1}) - V(S_t).
    # values has one extra trailing entry, the bootstrap value after the last step.
    rewards = torch.as_tensor(rewards, dtype=torch.float)
    values = torch.as_tensor(values, dtype=torch.float)
    is_stop = torch.as_tensor(is_stop, dtype=torch.float)

    length = rewards.numel()
    next_values = torch.zeros(length)
    next_values[:values.numel() - 1] = values[1:]

    return rewards + gamma * next_values * (1.0 - is_stop) - values[:length]


def gae_recursive(delta, gamma, lam):
    # eq. (7-103), recursive form: A_t = delta_t + gamma * lambda * A_{t+1}.
    delta = torch.as_tensor(delta, dtype=torch.float)
    advantages = torch.zeros_like(delta)
    running = torch.zeros(())
    decay = gamma * lam

    for t in reversed(range(delta.numel())):
        running = delta[t] + decay * running
        advantages[t] = running

    return advantages


def gae_explicit(delta, gamma, lam):
    # eq. (7-103), expanded form: A_t = sum_l (gamma * lambda)^l delta_{t+l}.
    # Same quantity as the recursive version; kept to show they agree.
    delta = torch.as_tensor(delta, dtype=torch.float)
    length = delta.numel()
    advantages = torch.zeros_like(delta)

    for t in range(length):
        weight = 1.0
        for offset in range(length - t):
            advantages[t] += weight * delta[t + offset]
            weight *= gamma * lam

    return advantages


def returns_from_advantages(advantages, values):
    # eq. (7-101): A(s,a) = q(s,a) - v(s), so the value target is A + V.
    return torch.as_tensor(advantages, dtype=torch.float) + torch.as_tensor(values, dtype=torch.float)


def format_vector(x):
    return "[" + " ".join(f"{float(v):.3f}" for v in x) + "]"


if __name__ == "__main__":
    gamma = 0.9
    rewards = [1.0, 1.0, 1.0, 1.0, 0.0]
    values = [0.5, 0.6, 0.4, 0.3, 0.0]
    is_stop = [0, 0, 0, 0, 1]
    delta = deltas(rewards, values, gamma, is_stop)

    for lam in (0.0, 0.5, 1.0):
        print(f"lambda={lam:<4} delta={format_vector(delta)}  A={format_vector(gae_recursive(delta, gamma, lam))}")

    gap = max(float((gae_recursive(delta, gamma, lam) - gae_explicit(delta, gamma, lam)).abs().max())
              for lam in (0.0, 0.5, 1.0))
    print(f"recursive vs expanded form, max difference = {gap:.2e}")

    # lambda = 0 collapses to the one-step TD error, eq. (7-102).
    print(f"lambda=0: max |A - delta| = {float((gae_recursive(delta, gamma, 0.0) - delta).abs().max()):.2e}")

    # lambda = 1 collapses to the Monte Carlo advantage G_t - V(S_t), eq. (7-7).
    running, monte_carlo = 0.0, [0.0] * len(rewards)
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        monte_carlo[t] = running
        if is_stop[t]:
            running = 0.0

    mc_advantage = torch.tensor(monte_carlo) - torch.tensor(values)
    print(f"lambda=1: A={format_vector(gae_recursive(delta, gamma, 1.0))}  "
          f"G - V={format_vector(mc_advantage)}")
    print(f"  max difference = {float((gae_recursive(delta, gamma, 1.0) - mc_advantage).abs().max()):.2e}")
