import gym
import numpy as np


class MiniGridCostWrapper(gym.Wrapper):
    def __init__(self, env, cost=0.9/1024):
        super(MiniGridCostWrapper, self).__init__(env)
        self.cost = cost

    def reset(self):
        return super(MiniGridCostWrapper, self).reset()

    def step(self, action):
        obs, reward, done, info = super(MiniGridCostWrapper, self).step(action)
        shaped_reward = 1 - self.cost if reward > 0 else -self.cost
        return obs, shaped_reward, done, info


class StepCostWrapper(gym.Wrapper):
    def __init__(self, env, cost=0.001):
        super(StepCostWrapper, self).__init__(env)
        self.cost = cost

    def reset(self):
        return super(StepCostWrapper, self).reset()

    def step(self, action):
        obs, reward, done, info = super(StepCostWrapper, self).step(action)
        reward = reward - self.cost
        return obs, reward, done, info


class NoopCostWrapper(gym.Wrapper):
    def __init__(self, env, cost=0.01):
        super(NoopCostWrapper, self).__init__(env)
        self.cost = cost

    def reset(self):
        return super(NoopCostWrapper, self).reset()

    def step(self, action):
        obs, reward, done, info = super(NoopCostWrapper, self).step(action)
        if "noop" in info and info["noop"]:
            reward = reward - self.cost
        return obs, reward, done, info


class DQNRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_scale=100, step_cost=1):
        super(DQNRewardWrapper, self).__init__(env)
        self.reward_scale = reward_scale
        self.step_cost = step_cost

    def reset(self):
        return super(DQNRewardWrapper, self).reset()

    def step(self, action):
        obs, reward, done, info = super(DQNRewardWrapper, self).step(action)
        reward = reward * self.reward_scale
        reward = reward - self.step_cost
        return obs, reward, done, info
