"""CartPole wrapper (textbook section 7.1.3, figure 7.2(d))."""

import gymnasium as gym
import torch


class CartPole:
    def __init__(self, render=False, seed=0):
        self.env = gym.make("CartPole-v1", render_mode="human" if render else None)
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)

        self.observation_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

    def reset(self):
        observation, _ = self.env.reset()
        return torch.as_tensor(observation, dtype=torch.float32)

    def step(self, action):
        observation, reward, terminated, truncated, _ = self.env.step(action)
        # is_stop: the pole fell, the state is dead. is_cut: the 500 step limit was hit,
        # the state is still alive. Only is_stop may cut the bootstrap target.
        is_stop, is_cut = terminated, truncated

        return torch.as_tensor(observation, dtype=torch.float32), reward, is_stop, is_cut
