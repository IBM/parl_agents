"""
modify ReplayBuffer such that
it returns the samples from the last to the first in 1 episode

In Tabular Q learning,
we update table per 1 episode,
while doing so this modification allows updating the sample
from the last (goal) to the first

during test, didn't observe difference between this deterministic sampling
and the basic random sampling
"""

from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np

class ReplayBufferEpisode(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device,
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True
    ):
        super(ReplayBufferEpisode, self).__init__(buffer_size,
                                                  observation_space,
                                                  action_space,
                                                  device,
                                                  n_envs,
                                                  optimize_memory_usage,
                                                  handle_timeout_termination)

    def sample(self, batch_size: int, env):
        # this batch size is 1 epsidoe length in Tabular Q learning
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.arange(upper_bound, upper_bound-batch_size, -1)
        return self._get_samples(batch_inds, env=env)
