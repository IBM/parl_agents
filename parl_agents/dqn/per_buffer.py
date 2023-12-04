"""
PER buffer
"""
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
import random

from .segment_tree import SumSegmentTree, MinSegmentTree


class PriorityReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device,
        alpha=0.6,      # default from baseline learn()
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True
    ):
        """
        alpha: full prioritization
        """
        super(PriorityReplayBuffer, self).__init__(buffer_size,
                                                   observation_space,
                                                   action_space,
                                                   device,
                                                   n_envs,
                                                   optimize_memory_usage,
                                                   handle_timeout_termination)
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < self.buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self.pos
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def sample(self, batch_size, beta, env=None):
        # env is for normalizing obs and reward
        # beta: to what degree to use importance weights 1 for full correction
        # return ReplayBufferSamples and weights, batch index
        batch_inds = np.array(self._sample_proportional(batch_size))       # this was list not np array, dim ok?

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.size()) ** (-beta)

        for idx in batch_inds:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self.size()) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        buffer_samples = self._get_samples(batch_inds, env)
        return buffer_samples, self.to_torch(weights), batch_inds

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.size() - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def update_priorities(self, batch_inds, priorities):
        assert len(batch_inds) == len(priorities)
        for idx, priority in zip(batch_inds, priorities):
            assert priority > 0
            assert 0 <= idx < self.size()
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
