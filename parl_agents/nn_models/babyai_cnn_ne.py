"""
Collection of NN feature extractors for training agents

We took model implementations from
https://github.com/mila-iqia/babyai/blob/dyth-babyai-v1.1/babyai/model.py

BaybAIFullyObsCNN won't use
  * English instructions
  * memory (rnn)
  * FiLM, so they are removed from the code.

It may not be the best architecture for our use case, but
we will retain this architecture consistently over experiments.
"""
import gym
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from parl_agents.nn_models.utils import initialize_parameters

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor
)

# nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
# nn.ReLU(),
# nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
# nn.ReLU(),
# nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
# nn.ReLU(),
# nn.Flatten(),


class BabyAIFullyObsCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 64,
    ):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=features_dim, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(features_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=features_dim, out_channels=features_dim, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(features_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Flatten()
        )

        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

        self.apply(initialize_parameters)       # Initialize parameters correctly

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations.float()
        x = self.cnn(x)
        x = self.linear(x)
        return x



class BabyAIFullyObsCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 64,
    ):
        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=features_dim, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(features_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=features_dim, out_channels=features_dim, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(features_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Flatten()
        )

        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

        self.apply(initialize_parameters)       # Initialize parameters correctly

    def forward(self, observations: th.Tensor) -> th.Tensor:

        x = observations.float()
        x = self.cnn(x)
        x = self.linear(x)
        return x


class BabyAIFullyObsSmallCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 64,
    ):
        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=features_dim, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(features_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Flatten()
        )

        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

        self.apply(initialize_parameters)       # Initialize parameters correctly

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # offsets = th.Tensor([0, self.max_value, 2 * self.max_value]).to(observations.device)
        # x = (observations + offsets[None, :, None, None]).long()
        # x = self.embedding(x).sum(1).permute(0, 3, 1, 2)

        x = observations.float()
        x = self.cnn(x)
        x = self.linear(x)
        return x
