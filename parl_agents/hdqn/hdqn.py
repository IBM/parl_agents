"""
Implementation of H-DQN by Kullkarni, Narasimhan, Saeedi, Tenenbaum (https://arxiv.org/abs/1604.06057)

HDQN = DDQN + DDQN+PER for meta and lower level control agents
Two policy objects one for higher level (DDQM) and the other for lower level (DDQN+PER)
        high level
        self.action_space = gym.spaces.Discrete(subgoal_task.num_subgoals)

        lower level -- repeat init process since it's hard to separate rollout/training
        control state space is (S, G) and action space is from env
        sb3 doesn't support spaces.Tuple --> make discretre
        Don't observation_space_low = gym.spaces.Tuple((env.observation_space, self.action_space))
        Do observation_space_low = gym.spaces.Discrete(self.observation_space.n * self.action_space.n)
        action_space_low = env.action_space

TODO
The policy network architecture is limited to MLP.
potential fix: subgoal as one more channel to CNN -- probably no fix
maybe one need custom normalizer?
"""
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable
import time
import warnings

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import get_policy_from_name
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.utils import (
    get_schedule_fn,
    update_learning_rate,
    get_linear_fn,
    is_vectorized_observation,
    polyak_update,
    safe_mean
)
from stable_baselines3.common.vec_env import (
    VecEnv,
    VecMonitor,
    is_vecenv_wrapped
)
from stable_baselines3.dqn.policies import DQNPolicy, MlpPolicy, CnnPolicy

from parl_annotations.annotated_tasks import SubgoalTask

from parl_agents.dqn.per_buffer import PriorityReplayBuffer


class HDQN(BaseAlgorithm):
    def __init__(
        self,
        # HDQN
        subgoal_task: SubgoalTask,
        env: Union[GymEnv, str],
        # DQN meta level
        policy = MlpPolicy,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 50000,
        learning_starts: int = 20000,
        batch_size: Optional[int] = 128,
        tau: float = 1.0,
        gamma: float = 1.0,
        # train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = 1,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 2000,
        exploration_fraction: float = 0.02,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.02,
        exploration_mid_fraction: float = 0.02,
        exploration_mid_eps: float = 1.0,
        max_grad_norm: float = 10,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        # DQN low level
        policy_low: Optional[CnnPolicy, MlpPolicy] = CnnPolicy,
        learning_rate_low: Union[float, Schedule] = 1e-4,
        buffer_size_low: int = 50000,
        learning_starts_low: int = 20000,
        batch_size_low: Optional[int] = 128,
        tau_low: float = 1.0,
        gamma_low: float = 1.0,
        # train_freq_low: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps_low: int = 1,
        replay_buffer_kwargs_low: Optional[Dict[str, Any]] = None,
        target_update_interval_low: int = 2000,
        # exploration_fraction_low: float = 0.02,   # adaptive scheduling == 1 - average_success_rate
        exploration_initial_eps_low: float = 1.0,
        exploration_final_eps_low: float = 0.02,
        max_grad_norm_low: float = 10,
        policy_kwargs_low: Optional[Dict[str, Any]] = None,
        # intrinsic reward
        reward_subtask: float = 100.0,
        reward_step: float = -1.0,
        # misc
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        verbose: int = 1,
        seed: Optional[int] = None,  # better to get seed from gym env and use it here too
        device: Union[th.device, str] = "auto",
        supported_action_spaces=(gym.spaces.Discrete,),
        action_noise=None,
        remove_time_limit_termination: bool = False,
        render: bool = False
    ):
        assert "alpha" in replay_buffer_kwargs_low, "pass alpha to create PER buffer in DQN low"
        # meta agent
        super(HDQN, self).__init__(
            policy=policy,
            env=env,
            policy_base=DQNPolicy,
            learning_rate=learning_rate,        # this must be Schedule fn
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=False,
            create_eval_env=create_eval_env,        # pass eval_env from outise
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=False,
            sde_sample_freq=-1,
            supported_action_spaces=supported_action_spaces
        )
        # meta state space is S, action space is subgoals
        self.num_subgoalsteps = 0
        self.action_space = gym.spaces.Discrete(subgoal_task.num_subgoals)

        # control agent -- repeat init process since it's hard to separate rollout/training
        # control state space is (S, G) and action space is from env
        # sb3 doesn't support spaces.Tuple --> make discretre
        # self.observation_space_low = gym.spaces.Tuple((env.observation_space, self.action_space))
        self.observation_space_low = gym.spaces.Discrete(self.observation_space.n * self.action_space.n)
        self.action_space_low = env.action_space

        if isinstance(policy_low, str):
            self.policy_class_low = get_policy_from_name(DQNPolicy, policy)
        else:
            self.policy_class_low = policy_low
        self.policy_kwargs_low = {} if policy_kwargs_low is None else policy_kwargs_low
        self.policy_low = None

        self.learning_rate_low = learning_rate_low
        self.lr_schedule_low = None  # type: Optional[Schedule]
        self._n_updates_low = 0  # type: int

        # Off policy algorithms
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.gradient_steps = gradient_steps
        self.action_noise = action_noise
        self.optimize_memory_usage = optimize_memory_usage
        self.replay_buffer_class = ReplayBuffer
        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {}
        self.replay_buffer_kwargs = replay_buffer_kwargs
        self._episode_storage = None

        self.buffer_size_low = buffer_size_low
        self.batch_size_low = batch_size_low
        self.learning_starts_low = learning_starts_low
        self.tau_low = tau_low
        self.gamma_low = gamma_low
        self.gradient_steps_low = gradient_steps_low

        self.replay_buffer_class_low = PriorityReplayBuffer
        self.replay_buffer_kwargs_low = replay_buffer_kwargs_low
        self._episode_storage_low = None

        self.remove_time_limit_termination = remove_time_limit_termination

        self.replay_buffer = None  # type: Optional[ReplayBuffer]
        self.replay_buffer_low = None  # type: Optional[ReplayBuffer]

        # DQN meta
        self.exploration_mid_fraction = exploration_mid_fraction
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_mid_eps = exploration_mid_eps
        self.exploration_final_eps = exploration_final_eps

        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        self.exploration_schedule = None
        self.q_net, self.q_net_target = None, None

        # DQN low
        self.exploration_initial_eps_low = exploration_initial_eps_low
        self.exploration_final_eps_low = exploration_final_eps_low
        self.exploration_fraction_low = exploration_mid_fraction        # TODO fix

        self.target_update_interval_low = target_update_interval_low
        self.max_grad_norm_low = max_grad_norm_low
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate_low = 0.0
        self.exploration_schedule_low = None
        self.q_net_low, self.q_net_target_low = None, None

        # HDQN
        self.subgoal_task = subgoal_task
        self.reward_subtask = reward_subtask
        self.reward_step = reward_step

        self.render = render

        # setup model
        self._setup_model()

    def tuple_to_discrete_obs(self, obs, subgoal):
        state_n = self.observation_space.n
        discrete_obs = obs + (subgoal) * state_n
        try:
            assert discrete_obs < self.action_space.n * state_n
        except:
            exit()
        return discrete_obs      # subgoal index starts from 0

    def _setup_model(self) -> None:
        self._setup_model_OffPolicyAlgorithm()
        self._setup_model_DQN()

    def _setup_model_OffPolicyAlgorithm(self) -> None:
        # self._setup_lr_schedule()
        self.lr_schedule = get_schedule_fn(self.learning_rate)
        self.lr_schedule_low = get_schedule_fn(self.learning_rate_low)

        self.set_random_seed(self.seed)
        self.action_space_low.seed(self.seed)

        # create replay buffer meta
        if self.replay_buffer is None:
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                self.device,
                optimize_memory_usage=self.optimize_memory_usage,
                **self.replay_buffer_kwargs,
            )
        # create replay buffer low
        if self.replay_buffer_low is None:
            self.replay_buffer_low = self.replay_buffer_class_low(
                self.buffer_size_low,
                self.observation_space_low,
                self.action_space_low,
                self.device,
                optimize_memory_usage=self.optimize_memory_usage,
                **self.replay_buffer_kwargs_low,
            )
        # create policy meta
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        # create policy low
        self.policy_low = self.policy_class_low(  # pytype:disable=not-instantiable
            self.observation_space_low,
            self.action_space_low,
            self.lr_schedule_low,
            **self.policy_kwargs_low,  # pytype:disable=not-instantiable
        )
        self.policy_low = self.policy_low.to(self.device)

    def _setup_model_DQN(self):
        # self._create_aliases()
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target
        self.q_net_low = self.policy_low.q_net
        self.q_net_target_low = self.policy_low.q_net_target

        # exploration meta schedule
        self.exploration_schedule = get_exploration_piecewise(
            self.exploration_initial_eps,
            self.exploration_mid_eps,
            self.exploration_final_eps,
            self.exploration_mid_fraction,
            self.exploration_fraction - self.exploration_mid_fraction
        )
        # exploration low schedule
        self.exploration_schedule_low = get_exploration_value(
            self.exploration_initial_eps_low,
            self.exploration_final_eps,
            self.exploration_fraction_low
        )

    def _setup_beta_schedule(self, total_timesteps, prioritized_replay_beta_iters, prioritized_replay_beta0):
        if prioritized_replay_beta_iters is None:
            end_fraction = 1.0
        else:
            end_fraction = prioritized_replay_beta_iters / total_timesteps
        self.beta_schedule = get_linear_fn(prioritized_replay_beta0, 1.0, end_fraction)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "HDQN",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        # PER params
        # prioritized_replay_alpha=0.6, # from dict in init
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=None,
        prioritized_replay_eps=1e-6,
    ):
        total_timesteps, callback = super(HDQN, self)._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        self._setup_beta_schedule(total_timesteps, prioritized_replay_beta_iters, prioritized_replay_beta0)

        callback.on_training_start(locals(), globals())

        # HDQN Algorithm
        episode_rewards, episode_lens = [], []
        ep_success, ep_fail = 0, 0

        while self.num_timesteps < total_timesteps:

            callback.on_rollout_start()
            assert self._last_obs is not None, "No previous observation was provided"
            ep_done = False
            rollout_interrupted = False
            episode_reward, episode_timesteps = 0.0, 0

            while not (ep_done or rollout_interrupted):     # 1 episodes
                subgoal_reward = 0.0
                begin_state_meta = self._last_obs
                subgoal_index = self._sample_subgoal(self.learning_starts, self._last_obs)
                subgoal_reached = self.subgoal_task.subgoal_test(subgoal_index.item(), self._last_obs.item())

                done, infos = np.array([False]), np.array([dict()])      # num_env is always 1
                while not (subgoal_reached or ep_done):
                    tuple_to_discrete = self.tuple_to_discrete_obs(self._last_obs.item(), subgoal_index.item())
                    current_state_low = np.array([tuple_to_discrete])
                    action = self._sample_action(self.learning_starts_low, current_state_low)

                    new_obs, reward, done, infos = self.env.step(action)
                    self.num_timesteps += 1
                    episode_timesteps += 1

                    dones_maxep = np.array([info['TimeLimit.truncated'] for info in infos]).reshape((1,))
                    if done[0]:
                        ep_done = True
                        if not dones_maxep[0]:
                            ep_success += 1
                        else:
                            ep_fail += 1

                    callback.update_locals(locals())

                    self._update_info_buffer(infos, done)

                    if callback.on_step() is False:
                        rollout_interrupted = True
                        break

                    episode_reward += reward.item()
                    subgoal_reward += reward.item()
                    subgoal_reached = self.subgoal_task.subgoal_test(subgoal_index.item(), new_obs.item())
                    intrinsic_reward = self.reward_step
                    if subgoal_reached:
                        intrinsic_reward += self.reward_subtask
                    intrinsic_reward = np.array([intrinsic_reward])

                    self._store_transition_low(self.replay_buffer_low, action, new_obs, subgoal_index,
                                               intrinsic_reward, done, infos)

                    self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                    self._on_step()

                    # HDQN updates params every step
                    if self.num_timesteps > self.learning_starts_low:
                        callback.on_training_start(locals(), globals())
                        if self.replay_buffer_low.size() > self.batch_size_low:
                            self.train_low(self.gradient_steps_low, self.batch_size_low, prioritized_replay_eps)
                        if self.num_timesteps > self.learning_starts and self.replay_buffer.size() > self.batch_size:
                            self.train_meta(self.gradient_steps, self.batch_size)
                        callback.on_training_end()

                # subgoal_reached or episode terminated
                if not rollout_interrupted:
                    self.num_subgoalsteps += 1
                    subgoal_reward = np.array([subgoal_reward])
                    self._store_transition(self.replay_buffer, subgoal_index,
                                           begin_state_meta, new_obs, subgoal_reward, done, infos)

            # finish 1 episode by reaching goal or reaching max len
            callback.on_rollout_end()
            if not rollout_interrupted:
                self._episode_num += 1
                episode_lens.append(episode_timesteps)
                episode_rewards.append(episode_rewards)

            if log_interval is not None and self._episode_num % log_interval == 0:
                self._dump_logs()

        return self

    def _sample_subgoal(self, learning_starts, observation):
        if self.num_timesteps < learning_starts:
            subgoal = np.array([self.action_space.sample()])
        else:
            subgoal, _ = self.predict_subgoal(observation, deterministic=False)
        return subgoal

    def _sample_action(self, learning_starts, observation):
        if self.num_timesteps < learning_starts:
            action = np.array([self.action_space_low.sample()])
        else:
            action, _ = self.predict(observation, deterministic=False)
        return action

    def predict_subgoal(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, mask, deterministic)
        return action, state

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        exploration_rate = self.exploration_rate_low
        observation_space = self.observation_space_low
        action_space = self.action_space_low
        policy = self.policy_low

        if not deterministic and np.random.rand() < exploration_rate:
            if is_vectorized_observation(maybe_transpose(observation, observation_space), observation_space):
                n_batch = observation.shape[0]
                action = np.array([action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(action_space.sample())
        else:
            action, state = policy.predict(observation, state, mask, deterministic)
        return action, state

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        if self.num_timesteps % self.target_update_interval_low == 0:
            polyak_update(self.q_net_low.parameters(), self.q_net_target_low.parameters(), self.tau_low)

        if self.num_subgoalsteps % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        fail_rate = 1 - safe_mean(self.ep_success_buffer)
        self.exploration_rate_low = self.exploration_schedule_low(self._current_progress_remaining, fail_rate)

        logger.record("rollout/exploration rate", self.exploration_rate)
        logger.record("rollout/exploration rate low", self.exploration_rate_low)

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        begin_state_meta: np.ndarray,
        new_obs: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        if done and infos[0].get("terminal_observation") is not None:
            next_obs = infos[0]["terminal_observation"]
            next_obs = np.array([next_obs])
        else:
            next_obs = new_obs

        replay_buffer.add(
            begin_state_meta,
            next_obs,
            buffer_action,
            reward,
            done,
            infos,
        )
        self._last_obs = new_obs

    def _store_transition_low(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: np.ndarray,
        subgoal_index: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        self._last_original_obs, new_obs_ = self._last_obs, new_obs
        obs_low = np.array([self.tuple_to_discrete_obs(self._last_original_obs.item(), subgoal_index.item())])

        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        if done and infos[0].get("terminal_observation") is not None:
            next_obs = infos[0]["terminal_observation"]
            next_obs = np.array([next_obs])
        else:
            next_obs = new_obs_

        next_obs_low = np.array([self.tuple_to_discrete_obs(next_obs.item(), subgoal_index.item())])

        replay_buffer.add(
            obs_low,
            next_obs_low,
            buffer_action,
            reward,
            done,
            infos,
        )
        self._last_obs = new_obs

    # buffer interface is different (need beta) for PER,
    def train_low(self, gradient_steps: int, batch_size: int = 100, prioritized_replay_eps: float=1e-6) -> None:
        q_net = self.q_net_low
        q_net_target = self.q_net_target_low
        replay_buffer = self.replay_buffer_low
        gamma = self.gamma_low
        policy = self.policy_low
        max_grad_norm = self.max_grad_norm_low

        # Update learning rate according to schedule
        logger.record("train/learning_rate_low", self.lr_schedule_low(self._current_progress_remaining))
        optimizers = policy.optimizer
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.lr_schedule_low(self._current_progress_remaining))
        beta = self._update_beta_rate()

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data, weights, batch_inds = replay_buffer.sample(batch_size, beta, self._vec_normalize_env)

            with th.no_grad():
                # obtain best action from q_net and use it for getting next_q_values
                q_net_values = q_net(replay_data.observations)
                _, best_actions = q_net_values.max(dim=1)
                next_q_values = q_net_target(replay_data.next_observations)
                next_q_values = next_q_values.gather(1, best_actions.unsqueeze(-1))

                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values, reduce=False, reduction=None)
            weights = th.unsqueeze(weights, dim=-1)     # weights[:, None]
            loss = th.mul(loss, weights)
            loss = th.mean(loss)
            losses.append(loss.item())

            # Optimize the policy
            policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            policy.optimizer.step()

            with th.no_grad():
                td_errors = current_q_values - target_q_values  # target_q_values has no gradient
                td_errors = td_errors.detach().cpu().numpy()
                new_priorities = np.abs(td_errors) + prioritized_replay_eps # td_errors from torch to np
                replay_buffer.update_priorities(batch_inds, new_priorities)    # batch_idxes are np array

        # Increase update counter
        self._n_updates_low += gradient_steps

        logger.record("train/n_updates_low", self._n_updates_low, exclude="tensorboard")
        logger.record("train/loss_low", np.mean(losses))

    def train_meta(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # obtain best action from q_net and use it for getting next_q_values
                q_net_values = self.q_net(replay_data.observations)
                _, best_actions = q_net_values.max(dim=1)
                next_q_values = self.q_net_target(replay_data.next_observations)
                # next_q_values = next_q_values[:, best_actions]
                next_q_values = next_q_values.gather(1, best_actions.unsqueeze(-1))

                # obtain best action from target network
                # # Compute the next Q-values using the target network
                # next_q_values = self.q_net_target(replay_data.next_observations)
                # # Follow greedy policy: use the one with the highest value
                # next_q_values, _ = next_q_values.max(dim=1)

                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps
        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/loss", np.mean(losses))

    def _update_beta_rate(self):
        logger.record("train/beta_rate", self.beta_schedule(self._current_progress_remaining))
        return self.beta_schedule(self._current_progress_remaining)

    def _excluded_save_params(self) -> List[str]:
        return super(HDQN, self)._excluded_save_params() + ["q_net", "q_net_target", "q_net_low", "q_net_target_low"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer", "policy_low", "policy_low.optimizer"]
        return state_dicts, []

    def _dump_logs(self) -> None:
        time_elapsed = time.time() - self.start_time
        fps = int(self.num_timesteps / (time_elapsed + 1e-8))
        logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        logger.record("time/fps", fps)
        logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")

        if len(self.ep_success_buffer) > 0:
            logger.record("rollout/success rate", safe_mean(self.ep_success_buffer))

        logger.dump(step=self.num_timesteps)

    def show_parameters(self):
        print("buffer_size:{}".format(self.buffer_size))
        print("buffer_size_low:{}".format(self.buffer_size_low))
        print("learning_rate:{}".format(self.learning_rate))
        print("learning_rate_low:{}".format(self.learning_rate_low))
        print("batch_size:{}".format(self.batch_size))
        print("batch_size_low:{}".format(self.batch_size_low))
        print("exploration_fraction:{}".format(self.exploration_fraction))
        print("exploration_mid_fraction:{}".format(self.exploration_mid_fraction))
        print("exploration_mid_eps:{}".format(self.exploration_mid_eps))
        print("exploration_fraction_low:{}".format(self.exploration_fraction_low))


def get_exploration_piecewise(eps_begin, eps_mid, eps_end, mid_fraction, end_fraction) -> Schedule:
    assert mid_fraction + end_fraction < 1.0, "each fraction is the ratio"
    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) < eps_mid:
            return max(eps_end, eps_begin + (1 - progress_remaining) * (eps_mid - eps_begin) / mid_fraction)
        elif (1 - progress_remaining) < mid_fraction + end_fraction:
            return max(eps_end, eps_mid + (1 - progress_remaining - mid_fraction) * (eps_end - eps_mid) / end_fraction)
        else:
            return eps_end
    return func


def get_exploration_value(eps_begin, eps_end, end_fraction) -> Schedule:
    def func(progress_remaining: float, value: float) -> float:
        if (1-progress_remaining) > end_fraction:
            return eps_end
        return min(eps_begin, max(eps_end, value))
    return func

def hdqn_evaluate_policy(
    model,
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.env_util import is_wrapped
    from stable_baselines3.common.monitor import Monitor

    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"
        is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    else:
        is_monitor_wrapped = is_wrapped(env, Monitor)

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    episode_rewards, episode_lengths = [], []
    while len(episode_rewards) < n_eval_episodes:
        done = False
        last_state_meta = env.reset()
        last_obs = last_state_meta
        episode_reward = 0.0
        episode_length = 0

        while not done:
            subgoal_index, _ = model.predict_subgoal(last_state_meta, deterministic=False)
            subgoal_reached = model.subgoal_task.subgoal_test(subgoal_index.item(), last_state_meta.item())

            while not (subgoal_reached or done):
                tuple_to_discrete = model.tuple_to_discrete_obs(last_obs.item(), subgoal_index.item())
                current_state_low = np.array([tuple_to_discrete])
                action, _ = model.predict(current_state_low, deterministic=False)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                if callback is not None:
                    callback(locals(), globals())
                episode_length += 1
                subgoal_reached = model.subgoal_task.subgoal_test(subgoal_index.item(), obs.item())
                last_obs = obs
            last_state_meta = last_obs

        if is_monitor_wrapped:
            # Do not trust "done" with episode endings.
            # Remove vecenv stacking (if any)
            if isinstance(env, VecEnv):
                info = info[0]
            if "episode" in info.keys():
                # Monitor wrapper includes "episode" key in info if environment
                # has been wrapped with it. Use those rewards instead.
                episode_rewards.append(info["episode"]["r"])
                episode_lengths.append(info["episode"]["l"])
        else:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward