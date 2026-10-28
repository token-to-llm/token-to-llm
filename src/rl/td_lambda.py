"""Multi-step TD and TD(lambda) (textbook section 7.5.4, eq. (7-52) and (7-53))."""

import torch

from base import ValueAgent, evaluate, greedy_from_v, uniform_policy
from envs.grid_world import GAMMA, MAX_STEP, GridWorld, non_terminal

torch.manual_seed(0)


def sample_episode(env, behaviour, max_step=MAX_STEP):
    # eq. (7-7): one episode. states[t] is S_t, the state the t-th action is taken from,
    # and rewards[t] is R_{t+1}, what the environment pays for that step. next_state is
    # the state after the last action: terminal, or alive if the step limit was hit.
    states, rewards, next_state, is_done = [], [], None, False

    s, _ = env.reset()
    for _ in range(max_step):
        s2, r, is_stop, is_cut, _ = env.step(behaviour(s))

        states.append(s)
        rewards.append(r)

        s = s2
        if is_stop or is_cut:
            is_done = is_stop
            break

    return states, rewards, s, is_done


def lambda_return(rewards, next_values, lam, gamma=GAMMA):
    # eq. (7-53): G_t^lambda = (1 - lambda) sum_{n>=1} lambda^(n-1) G_t^(n), truncated at
    # the episode length. The truncated last term is the Monte Carlo tail and carries no
    # bootstrap. lambda = 0 leaves the one-step target R + gamma V(S'); lambda = 1
    # leaves the Monte Carlo return.
    length = len(rewards)
    returns = torch.zeros(length)

    for t in range(length):
        total = 0.0
        partial_return = 0.0

        for n in range(1, length - t + 1):
            partial_return += gamma ** (n - 1) * rewards[t + n - 1]

            if n < length - t:
                weight = (1 - lam) * lam ** (n - 1)
                bootstrap = gamma ** n * next_values[t + n - 1]
            else:
                weight = lam ** (length - t - 1)
                bootstrap = 0.0

            total += weight * (partial_return + bootstrap)

        returns[t] = total

    return returns


class TDLambda(ValueAgent):
    def __init__(self, lam=0.0, state_size=9, gamma=GAMMA, alpha=0.1, behaviour=None):
        super().__init__(state_size, gamma=gamma, alpha=alpha, behaviour=behaviour)

        self.lam = lam

    def train(self, env, episodes=2000, max_step=MAX_STEP):
        for _ in range(episodes):
            states, rewards, next_state, is_done = sample_episode(env, self.behaviour, max_step)

            # V of the successor states. A terminal successor has no value; a truncated
            # one is still alive, so it keeps its estimate.
            next_values = [float(self.value_table[s]) for s in states[1:]]
            next_values.append(0.0 if is_done else float(self.value_table[next_state]))

            returns = lambda_return(rewards, next_values, self.lam, self.gamma)

            for t, s in enumerate(states):
                # eq. (7-52). The same state may be visited twice in one episode, so
                # update once per visit rather than once per state.
                self.value_table[s] += self.alpha * (returns[t] - self.value_table[s])

        return self

    def policy(self, env):
        return lambda s: greedy_from_v(env, self.value_table, s)


def monte_carlo_value(env, episodes=2000):
    # Reference for lambda = 1: the plain average of sampled returns.
    returns = torch.zeros(9)
    counts = torch.zeros(9)

    for _ in range(episodes):
        states, rewards, _, _ = sample_episode(env, uniform_policy())
        running_return = 0.0

        for t in reversed(range(len(rewards))):
            running_return = GAMMA * running_return + rewards[t]
            returns[states[t]] += running_return
            counts[states[t]] += 1

    return returns / counts.clamp(min=1)


if __name__ == "__main__":
    values = {}
    for lam in (0.0, 0.5, 1.0):
        torch.manual_seed(0)
        agent = TDLambda(lam=lam, alpha=0.005).train(GridWorld("slip"), episodes=50000)
        values[lam] = agent.value_table.clone()

        print(f"lambda = {lam:.1f}  V(start) = {agent.value_table[0]:.2f}  "
              f"V = {[round(v, 2) for v in agent.value_table.tolist()]}")

    # On a single trajectory the two endpoints are exact, not merely close.
    states, rewards, next_state, is_done = sample_episode(GridWorld("slip"), uniform_policy())
    next_values = [float(values[0.0][s]) for s in states[1:]]
    next_values.append(0.0 if is_done else float(values[0.0][next_state]))

    one_step = lambda_return(rewards, next_values, 0.0)
    one_step_target = torch.tensor([rewards[t] + GAMMA * next_values[t] for t in range(len(rewards))])
    monte_carlo = lambda_return(rewards, next_values, 1.0)
    monte_carlo_target = torch.tensor([sum(GAMMA ** k * rewards[t + k] for k in range(len(rewards) - t))
                                       for t in range(len(rewards))])

    print(f"one episode: max |lambda=0 - one-step target| = {float((one_step - one_step_target).abs().max()):.2e}")
    print(f"one episode: max |lambda=1 - Monte Carlo return| = {float((monte_carlo - monte_carlo_target).abs().max()):.2e}")

    print(f"training: max |lambda=1 - Monte Carlo estimate| = "
          f"{float((values[1.0] - monte_carlo_value(GridWorld('slip'))).abs().max()):.3f}")

    env = GridWorld("slip")
    torch.manual_seed(0)
    agent = TDLambda(lam=0.5, alpha=0.005).train(env, episodes=50000)
    average_return, win_rate = evaluate(agent.policy(env), env, episodes=100)

    print(f"lambda = 0.5, one-step lookahead on V: return {average_return:.2f}, win rate {win_rate:.0%}")
