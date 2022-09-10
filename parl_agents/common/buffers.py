from typing import Optional, Generator
import numpy as np

from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.buffers import RolloutBuffer


class OptionRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        """
        option rollout buffer doesn't full during rollout like flatPPO
        (1) we should sample batches up to the buffer limit
        (2) batch_size is adjusted to be bounded by buffer limit
        """
        if self.full:
            buffer_limit = self.buffer_size
        else:
            buffer_limit = self.pos
        indices = np.random.permutation(buffer_limit * self.n_envs)     # n_envs is 1

        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = buffer_limit * self.n_envs

        start_idx = 0
        while start_idx < buffer_limit * self.n_envs - 1:   
            # 1 less ensures the batch size is at least 2, currently self.n_envs is 1
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
