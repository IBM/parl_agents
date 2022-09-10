import gym
from gym.spaces.utils import flatdim

import torch as th

from stable_baselines3.common.policies import BasePolicy


class QTable(BasePolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule=None,
            gamma=0.99,
    ):
        super(QTable, self).__init__(
            observation_space,
            action_space,
            features_extractor_class = None,        # Q learning takes state numbers directly
            optimizer_class = None      # Q learning doesn't use optimizer
        )
        self.num_states = flatdim(observation_space)
        self.num_actions = flatdim(action_space)        # this assumes the environment maps action index to operator
        self.q_net = th.zeros(size=(self.num_states, self.num_actions), requires_grad=False)
        self.gamma=gamma
        self.learning_rate = 1.0    # this is for monitoring purpose, actual number is updated outside

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        :param obs: tensor of discrete state observations used as index to the row
        :return: stacked rows for the action values
        """
        return self.q_net[obs, :]

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        """
        :param observation: 1d tensor of discrete state observations
        :param deterministic: greedy action selection
        :return: tensor of greedy actions
        """
        q_values = self.forward(observation)
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def step(self, replay_data, learning_rate):
        # learning_rate is decaying over time
        observations = replay_data.observations
        actions = replay_data.actions.long()
        next_observations = replay_data.next_observations

        # Compute the next Q-values using the target network
        next_q_values = self.forward(next_observations.squeeze())
        # Follow greedy policy: use the one with the highest value
        next_q_values, _ = next_q_values.max(dim=1)
        # Avoid potential broadcast issue
        next_q_values = next_q_values.reshape(-1, 1)
        # 1-step TD target
        target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # Get current Q-values estimates
        current_q = self.forward(observations.squeeze())
        # Retrieve the q-values for the actions from the replay buffer
        current_q = th.gather(current_q, dim=1, index=actions)

        delta_q = target_q - current_q
        for ind in range(len(observations)):
            self.q_net[observations[ind], actions[ind]] = current_q[ind] + learning_rate * delta_q[ind]
        # self.q_net[observations.squeeze(), actions] = current_q + learning_rate * delta_q

        return delta_q




