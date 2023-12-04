"""
DQN or DDQN with PER Buffer
"""
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, is_vectorized_observation, polyak_update, safe_mean
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.dqn import DQN

from .per_buffer import PriorityReplayBuffer


class DQN_PER(DQN):
    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        double_dqn: bool = False,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 50000,
        batch_size: Optional[int] = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[ReplayBuffer] = PriorityReplayBuffer,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,      # pass alpha through dict(alpha=)
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=None,
        prioritized_replay_eps=1e-6,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        print_log = True,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,    # call setup model in init
    ):
        assert "alpha" in replay_buffer_kwargs, "pass alpha to create PER buffer"
        self.double_dqn = double_dqn
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self.prioritized_replay_eps = prioritized_replay_eps
        super(DQN_PER, self).__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class,
            replay_buffer_kwargs,       # must pass alpha
            optimize_memory_usage,
            target_update_interval,
            exploration_fraction,
            exploration_initial_eps,
            exploration_final_eps,
            max_grad_norm,
            tensorboard_log,
            create_eval_env,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model
        )
        self.print_log = True

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv] = None,
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ):
        self._setup_beta_schedule(total_timesteps,
                                  self.prioritized_replay_beta_iters,
                                  self.prioritized_replay_beta0)  # this could go inside _setup_model() but put it here
        return super()._setup_learn(total_timesteps,
                                    eval_env,
                                    callback,
                                    eval_freq,
                                    n_eval_episodes,
                                    log_path,
                                    reset_num_timesteps,
                                    tb_log_name)
    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "DQN-PER",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ):

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps > 0 else rollout.episode_timesteps
                self.train(gradient_steps, self.batch_size, self.prioritized_replay_eps)

        callback.on_training_end()

        return self

    def _setup_beta_schedule(self, total_timesteps, prioritized_replay_beta_iters, prioritized_replay_beta0):
        if prioritized_replay_beta_iters is None:
            end_fraction = 1.0
        else:
            end_fraction = prioritized_replay_beta_iters / total_timesteps
        self.beta_schedule = get_linear_fn(start=prioritized_replay_beta0,
                                           end=1.0,
                                           end_fraction=end_fraction)

    def _update_beta_rate(self):
        if self.print_log:
            self.logger.record("train/beta_rate", self.beta_schedule(self._current_progress_remaining))
        return self.beta_schedule(self._current_progress_remaining)

    # buffer interface is different (need beta) for PER,
    def train(self, gradient_steps: int, batch_size: int = 100, prioritized_replay_eps: float=1e-6) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)
        beta = self._update_beta_rate()

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data, weights, batch_inds = self.replay_buffer.sample(batch_size, beta, self._vec_normalize_env)

            with th.no_grad():
                if not self.double_dqn:
                    # Compute the next Q-values using the target network
                    next_q_values = self.q_net_target(replay_data.next_observations)
                    # Follow greedy policy: use the one with the highest value
                    next_q_values, _ = next_q_values.max(dim=1)
                else:
                    # obtain best action from q_net and use it for getting next_q_values
                    q_net_values = self.q_net(replay_data.observations)
                    _, best_actions = q_net_values.max(dim=1)
                    next_q_values = self.q_net_target(replay_data.next_observations)
                    next_q_values = next_q_values.gather(1, best_actions.unsqueeze(-1))

                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values, reduce=False, reduction=None)    # TODO check reduction
            weights = th.unsqueeze(weights, dim=-1)     # weights[:, None]
            loss = th.mul(loss, weights)
            loss = th.mean(loss)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            with th.no_grad():
                td_errors = current_q_values - target_q_values  # target_q_values has no gradient
                td_errors = td_errors.detach().cpu().numpy()
                new_priorities = np.abs(td_errors) + prioritized_replay_eps # td_errors from torch to np
                self.replay_buffer.update_priorities(batch_inds, new_priorities)    # batch_idxes are np array

        # Increase update counter
        self._n_updates += gradient_steps
        if self.print_log:
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/loss", np.mean(losses))

    def _dump_logs(self):
        if self.print_log:
            super()._dump_logs()