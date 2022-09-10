from abc import abstractmethod
from collections import deque, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
import time
import os

import gym
import numpy as np
import torch as th

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import (
    GymEnv, MaybeCallback, Schedule, TrainFreq, TrainFrequencyUnit
)
from stable_baselines3.common.utils import (
    safe_mean,
    get_latest_run_id,
    get_linear_fn,
    configure_logger,
    should_collect_more_steps,
)

from parl_agents.dqn.ddqn import DDQN
from stable_baselines3.dqn.policies import DQNPolicy
from parl_agents.common.parl_base import PaRLBase
from parl_annotations.annotated_tasks import AnnotatedTask


class HplanDQN(PaRLBase):
    """ Extend PaRLBase with option agent being DQN in SB3 """
    def __init__(
        self,
        # args for BaseAlgorithm
        parl_policy: Union[Type[BasePolicy], None],
        option_agent_policy: Type[DQNPolicy],         # TODO DQN policy modification needed?
        env: Union[GymEnv, str, None],
        parl_policy_learning_rate: Union[float, Schedule],
        option_policy_learning_rate: Union[float, Schedule],
        # additional args for PaRLBase
        parl_task: AnnotatedTask,
        max_episode_len: int,
        # kwargs for BaseAlgorithm
        parl_policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        support_multi_env: bool = False,
        create_eval_env: bool = False,  # 1 eval gym env for PaRL task
        monitor_wrapper: bool = False,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
        # additional kwargs for PaRLBase
        is_on_policy: bool = False,
        is_markovian_reward: bool = True,
        option_termination_reward: float = 1.0,
        option_step_cost: float = 1.0,
        option_penalty_cost: float = 1.0,
        option_terminal_cost: float = 0.0,
        reward_scale: float = 1.0,
        use_intrinsic_reward = True,
        # additional kwargs for DQN
        buffer_size: int = 10240,
        learning_starts: int = 1024,
        batch_size: int = 256,
        tau: float = 1.0,           # 1 for hard update, Polyak
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (2048, "step"),
        gradient_steps: int = 10,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 8000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.01,
        max_grad_norm: float = 10,
        option_policy_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # PaRL base algorithm
        super().__init__(
            parl_policy=parl_policy,
            env=env,
            parl_policy_learning_rate=parl_policy_learning_rate,
            parl_task=parl_task,
            max_episode_len=max_episode_len,
            parl_policy_kwargs=parl_policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            create_eval_env=create_eval_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            supported_action_spaces=supported_action_spaces,
            is_on_policy=is_on_policy,
            is_markovian_reward=is_markovian_reward,
            option_termination_reward=option_termination_reward,
            option_step_cost=option_step_cost,
            option_penalty_cost=option_penalty_cost,
            option_terminal_cost=option_terminal_cost,
            reward_scale=reward_scale,
            buffer_size=buffer_size,
            buffer_kwargs=replay_buffer_kwargs,
            use_intrinsic_reward = use_intrinsic_reward            
        )

        self.option_agent_policy = option_agent_policy
        self.option_policy_learning_rate = option_policy_learning_rate
        self.option_policy_kwargs = option_policy_kwargs

        # init for DQN parameters; store here first and create agent later
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.replay_buffer_class = replay_buffer_class
        self.replay_buffer_kwargs = replay_buffer_kwargs
        self.optimize_memory_usage = optimize_memory_usage
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.max_grad_norm = max_grad_norm

    def _setup_model(self):
        # set up PaRL model, high level policy, feature extractor
        super()._setup_model()

        print("{}._setup_model()".format(self.__class__.__name__))
        self._convert_train_freq()

        if self.planning_task.name not in self.option_agents:
            self.create_option_agent(self.planning_task.name)

    def create_option_agent(self, option_name):
        """
            perform __init__, and _setup_model for DDQN
            manually add steps in DQN, off policy, base classes
        """
        tensorboard_log_agent = self.tensorboard_log

        self.option_agents[option_name] = DDQN(
            policy = self.option_agent_policy,
            env = self.env,
            learning_rate = self.option_policy_learning_rate,
            buffer_size = self.buffer_size,
            learning_starts = self.learning_starts,
            batch_size = self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            replay_buffer_class=self.replay_buffer_class,
            replay_buffer_kwargs=self.replay_buffer_kwargs,
            optimize_memory_usage=self.optimize_memory_usage,
            target_update_interval=self.target_update_interval,
            exploration_fraction=self.exploration_fraction,
            exploration_initial_eps=self.exploration_initial_eps,
            exploration_final_eps=self.exploration_final_eps,
            max_grad_norm=self.max_grad_norm,
            tensorboard_log=tensorboard_log_agent,
            create_eval_env=False,
            policy_kwargs=self.option_policy_kwargs,
            verbose=self.verbose,
            seed=self.seed,
            device=self.device,
            _init_setup_model=False,
        )
        option_agent = self.option_agents[option_name]

        # manually do _setup model
        option_agent._setup_lr_schedule()
        self._add_option_buffer(option_name)            
        option_agent.replay_buffer = self.sample_buffer[option_name]
        
        option_agent.policy = self.option_agent_policy(
            option_agent.observation_space,
            option_agent.action_space, 
            option_agent.lr_schedule, 
            **self.option_policy_kwargs)
        option_agent.policy = option_agent.policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        option_agent._convert_train_freq()

        option_agent._create_aliases()      # option_agent.q_net; option_agent.q_enet_target
        option_agent.exploration_schedule = get_linear_fn(
            option_agent.exploration_initial_eps,
            option_agent.exploration_final_eps,
            option_agent.exploration_fraction,
        )

    def _setup_learn_option_agent(
        self,
        option_agent,
        total_timesteps: int,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ):
        """
            DQN has _setup_learn() defined OffPolicyAlgorithm and BaseAlgorithm
            adapt it to avoid environment resets
        """
        # OffPolicyAlgorithm -- don't optimize memory usage and don't truncate last trajectory

        # BaseAlgorithm
        option_agent.start_time = time.time()
        if option_agent.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            option_agent.ep_info_buffer = deque(maxlen=100)
            option_agent.ep_success_buffer = deque(maxlen=100)

        if option_agent.action_noise is not None:
            option_agent.action_noise.reset()

        if reset_num_timesteps:
            option_agent.num_timesteps = 0
            option_agent._episode_num = 0
        else:
            total_timesteps += option_agent.num_timesteps
        option_agent._total_timesteps = total_timesteps
        option_agent._num_timesteps_at_start = option_agent.num_timesteps

        if not option_agent._custom_logger:
            option_agent._logger = configure_logger(self.verbose,
                                                    self.tensorboard_log, tb_log_name, reset_num_timesteps)
        # don't create eval env for option agent!
        return total_timesteps

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        option_total_timesteps: int = 1e5,
    ):
        """ call _setup_learn from PaRLBase and BaseAlgorithm
            Then, call _setup_learn for each option agent, but don't use DQN._setup_learn() """

        self.option_total_timesteps = option_total_timesteps

        # _setup_learn from PaRLBase, which also calls BaseAlgorithm
        total_timesteps, callback = super()._setup_learn(total_timesteps,
                                                         eval_env,
                                                         callback,
                                                         eval_freq,
                                                         n_eval_episodes,
                                                         log_path,
                                                         reset_num_timesteps,
                                                         tb_log_name)

        # init timestep counter and logger per option, don't touch eval env and no reset
        for option_name, option_agent in self.option_agents.items():
            self._setup_learn_option_agent(
                option_agent,
                total_timesteps=option_total_timesteps,
                tb_log_name=tb_log_name + "_" + str(self.latest_run_id) + "/" + option_name,
                reset_num_timesteps=reset_num_timesteps
            )
        return total_timesteps, callback

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        option_total_timesteps: int = 1e5,
    ):

        tb_log_name = self.__class__.__name__
        self.latest_run_id = get_latest_run_id(self.tensorboard_log, tb_log_name) + 1

        # call setup method in BaseAlgorithm
        total_timesteps, callback = self._setup_learn(total_timesteps,
                                                      eval_env,
                                                      callback,
                                                      eval_freq,
                                                      n_eval_episodes,
                                                      eval_log_path,
                                                      reset_num_timesteps,
                                                      tb_log_name,
                                                      option_total_timesteps)
        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            self.collected_option_names = []
            self.collected_pl_states = []

            episode_timesteps, n_episodes, continue_training = self.collect_rollouts(
                self.env,
                callback=callback,
                train_freq=self.train_freq,
                log_interval=log_interval,
            )
            if not continue_training:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    # print(self.collected_option_names)
                    # print(self.collected_pl_states)
                    self.train(gradient_steps=gradient_steps)

        callback.on_training_end()
        return self

    def collect_rollouts(self, env, callback, train_freq, log_interval):
        assert self._last_obs is not None
        assert self._last_pl_state is not None
        assert env.num_envs == 1                            # state mapping already assumed num env is 1

        num_collected_steps, num_collected_episodes = 0, 0

        callback.on_rollout_start()

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            self._last_pl_state = self.state_map(self._last_obs)    # In MiniGrid, we read pl sate from env directly
            self.pl_state_option_init = self._last_pl_state     # remember what was the starting pl state
            self.collected_pl_states.append(self.pl_state_option_init)  # record pl states during rollout

            self.option_stepwise_rewards = defaultdict(list)    # record intrinsic rewards during rollout
            t_start_option = time.time()
            step_count_option = 0

            print("starting_state:{}".format(self._last_pl_state))
            cur_option = cur_agent = None
            if self.planning_task.goal_reached(self.pl_state_option_init):
                cur_option_name = self.planning_task.name
                print("goal_reached:{}".format(cur_option_name))
            else:
                pl_state_str = str(self.pl_state_option_init)
                if pl_state_str not in self.parl_policy:
                    self.planning_task.initial_state = self.pl_state_option_init
                    plan_as_policy = self.planner.solve(self.planning_task)
                    assert plan_as_policy is not None
                    self.step_parl_policy(plan_as_policy)
                plan_action = self.forward_parl_policy(self.pl_state_option_init, get_first=True)
                # print("s={}->a={}".format(pl_state_str, plan_action))
                cur_option = self.strips_options[self.option_name2ind[plan_action]]
                cur_option_name = cur_option.name

            self.collected_option_names.append(cur_option_name)
            if cur_option_name in self.option_agents:
                cur_agent = self.option_agents[cur_option_name]
            else:
                self.create_option_agent(cur_option_name)
                cur_agent = self.option_agents[cur_option_name]
                self._setup_learn_option_agent(cur_agent, self.option_total_timesteps, True,
                                               self.__class__.__name__ + "_" + str(self.latest_run_id) + "/" + cur_option_name)
            cur_buffer = self.sample_buffer[cur_option_name]

            cur_agent.policy.set_training_mode(False)
            continue_rollout_cur_option = True

            while continue_rollout_cur_option:
                # Select action randomly or according to policy
                cur_agent._last_obs = self._last_obs
                actions, buffer_actions = cur_agent._sample_action(cur_agent.learning_starts, None, env.num_envs)

                new_obs, rewards, dones, infos = env.step(actions)

                new_pl_state = self.state_map(new_obs)
                self.num_timesteps += env.num_envs
                cur_agent.num_timesteps += self.n_envs  # increment agent counter
                num_collected_steps += 1
                step_count_option += 1

                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return num_collected_steps * env.num_envs, num_collected_episodes, False

                self._update_info_buffer(infos, dones)

                # decide done and info dictionary for option and update it to ep_info/success_buffer in agent
                option_termination_info = self.decide_option_termination(cur_option, new_pl_state, dones, infos)
                option_infos = option_termination_info['option_infos']
                option_dones = option_termination_info['option_dones']
                continue_rollout_cur_option = option_termination_info['continue_rollout_cur_option']
                # print(option_termination_info)

                # basic intrinsic reward for option agent
                if cur_option_name == self.planning_task.name:
                    option_rewards = self.goal_option_shape_zero_reward(option_infos)
                else:
                    option_rewards = self.plan_option_intrinsic_reward(cur_option,
                                                                       self.pl_state_option_init,
                                                                       new_pl_state,
                                                                       rewards,
                                                                       option_infos)
                self.option_stepwise_rewards[cur_option_name].extend(option_rewards)

                # epside information dictionary for option episode termination
                if option_dones[0]:
                    option_infos[0]["episode"] = {
                        "r": round(sum(self.option_stepwise_rewards[cur_option_name]), 6),
                        "l": len(self.option_stepwise_rewards[cur_option_name]),
                        "t": round(time.time()- t_start_option, 6)
                    }
                cur_agent._update_info_buffer(option_infos, option_dones)

                cur_agent._last_obs = self._last_obs
                # cur_agent._last_original_obs = self._last_original_obs
                cur_agent._store_transition(cur_buffer, buffer_actions, new_obs, option_rewards, option_dones, option_infos)
                self._last_obs = new_obs
                self._last_pl_state = new_pl_state

                cur_agent._update_current_progress_remaining(cur_agent.num_timesteps, cur_agent._total_timesteps)

                cur_agent._on_step()  # DQN agent also checks polyak update

                # dump log when episode finishes
                for idx, done in enumerate(dones):
                    if done:
                        # Update stats
                        num_collected_episodes += 1
                        self._episode_num += 1
                        if log_interval is not None and self._episode_num % log_interval == 0:
                            self.dump_logs(self, None)

                for idx, option_done in enumerate(option_dones):
                    if option_done:
                        # Update stats
                        cur_agent._episode_num += 1

                        if log_interval is not None and cur_agent._episode_num % log_interval == 0:
                            self.dump_logs(cur_agent, cur_option_name)

        callback.on_rollout_end()

        return num_collected_steps * env.num_envs, num_collected_episodes, True

    def dump_logs(self, agent, agent_name=None) -> None:
        if agent_name is None:
            name_str = ""
        else:
            name_str = agent_name + "/"

        time_elapsed = time.time() - agent.start_time
        fps = int((agent.num_timesteps - agent._num_timesteps_at_start) / (time_elapsed + 1e-8))
        agent.logger.record(name_str + "time/fps", fps)
        agent.logger.record(name_str + "time/episodes", agent._episode_num)
        agent.logger.record(name_str + "time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        agent.logger.record(name_str + "time/total_timesteps", agent.num_timesteps, exclude="tensorboard")

        if len(agent.ep_info_buffer) > 0 and len(agent.ep_info_buffer[0]) > 0:
            agent.logger.record(name_str + "rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            agent.logger.record(name_str + "rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))

        if len(agent.ep_success_buffer) > 0:
            agent.logger.record(name_str + "rollout/ep_success", safe_mean([1 if is_success else 0 for is_success in agent.ep_success_buffer]))
        else:
            agent.logger.record(name_str + "rollout/ep_success", 0)
        # Pass the number of timesteps for tensorboard
        agent.logger.dump(step=agent.num_timesteps)

    def train(self, gradient_steps):
        for option_name in sorted(set(self.collected_option_names)):
            cur_agent = self.option_agents[option_name]

            if cur_agent.num_timesteps > cur_agent.batch_size and cur_agent.num_timesteps > cur_agent.learning_starts:
                cur_agent.train(gradient_steps=gradient_steps, batch_size=cur_agent.batch_size)

    def _convert_train_freq(self) -> None:
        """ from SB3 off_policy_algorithm
        Convert `train_freq` parameter (int or tuple)
        to a TrainFreq object.
        """
        if not isinstance(self.train_freq, TrainFreq):
            train_freq = self.train_freq

            # The value of the train frequency will be checked later
            if not isinstance(train_freq, tuple):
                train_freq = (train_freq, "step")

            try:
                train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))
            except ValueError:
                raise ValueError(f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!")

            if not isinstance(train_freq[0], int):
                raise ValueError(f"The frequency of `train_freq` must be an integer and not {train_freq[0]}")

            self.train_freq = TrainFreq(*train_freq)

    @classmethod
    def load(
        cls,
        path,
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        model = cls(env=env, **kwargs)
        super(HplanDQN, model)._setup_model()
        # this is the last step before option agent creation in _setup_model
        model._convert_train_freq()     

        # here, load instead of creating option agents
        valid_option_names = set([option.name for option in model.strips_options])
        valid_option_names.add(model.planning_task.name)

        for f in os.listdir(path):
            if f.startswith("model--") and f.endswith(".zip"):
                option_name = f.replace("model--", "").replace(".zip", "")
                assert option_name in valid_option_names
                print("start loading {}".format(option_name))

                agent = DDQN.load(
                    path=os.path.join(path, f),
                    env=model.env,
                    device=model.device
                )

                # internally loading will _setup_model() so take buffer from agent
                model.option_agents[option_name] = agent
                model.sample_buffer[option_name] = agent.replay_buffer

        return model
