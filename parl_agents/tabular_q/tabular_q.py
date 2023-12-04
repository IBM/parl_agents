from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import gym
import torch as th

from stable_baselines3.common import logger
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.utils import get_linear_fn, is_vectorized_observation

from parl_agents.tabular_q.policies import QTable
from parl_agents.tabular_q.buffers import ReplayBufferEpisode


class TabularQ(OffPolicyAlgorithm):
    def __init__(
            self,
            policy: Type[BasePolicy],       # class for creating Q table
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule],      # get initial learning rate and modify to callable
            buffer_size: int = 2048,        # larger than 1 full episode length
            learning_starts: int = 0,
            batch_size: int = 0,            # don't use batch_size, take full episode per update
            tau: float = 0.005,
            gamma: float = 0.99,
            replay_buffer_class: Optional[ReplayBuffer] = ReplayBufferEpisode,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            policy_kwargs: Dict[str, Any] = None,
            exploration_fraction: float = 0.1,
            exploration_initial_eps: float = 1.0,
            exploration_final_eps: float = 0.05,
            tensorboard_log: Optional[str] = None,
            verbose: int = 0,
            create_eval_env: bool = False,
            seed: Optional[int] = None,
    ):
        # choose optimal learning_rate 1.0 if environment is deterministic, constant scheduler will be invoked
        # otherwise pass Schedule object to TabularQ get_linear_fn(start, end, end_fration) in SB3
        super(TabularQ, self).__init__(
            policy=policy,
            env=env,
            policy_base=QTable,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=(1, "episode"),
            gradient_steps=0,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=dict(gamma=gamma),
            device='cpu',
            verbose=verbose,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=False
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.exploration_rate = 0.0
        self.exploration_schedule = None
        self.q_net = None

        # Linear schedule will be defined in `_setup_model()`
        self._setup_model()

    def _setup_model(self) -> None:
        super(TabularQ, self)._setup_model()        # where is the place creating buffer? by default it is None and use base class to create object
        self.q_net = self.policy.q_net
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration rate", self.exploration_rate)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "TabularQ",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:
        return super(TabularQ, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _update_learning_rate(self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer] = None) -> None:
        self.policy.learning_rate = self.lr_schedule(self._current_progress_remaining)  # update learning rate
        self.logger.record("train/learning_rate", self.policy.learning_rate)

    def train(self, gradient_steps, batch_size) -> None:
        # gradient_steps is episode length (total number of samples generated during rollout)
        self._update_learning_rate()    # updae learning rate in linear scheduling

        batch_size = gradient_steps
        # this replay buffer returns samples collected from last to the first during rollout
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
        with th.no_grad():
            delta_q = self.policy.step(replay_data, self.learning_rate)
        self._n_updates += gradient_steps
        self.replay_buffer.reset()

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", delta_q.max().item())   # this is not loss, but the TD error

    def predict(
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

    def _excluded_save_params(self) -> List[str]:
        return super(TabularQ, self)._excluded_save_params() + ["q_net"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy"]
        return state_dicts, []
