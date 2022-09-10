from abc import abstractmethod
from collections import deque, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
import time
import os

import gym
import numpy as np
import torch as th

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import (
    obs_as_tensor,
    safe_mean,
    get_schedule_fn,
    get_latest_run_id,
    configure_logger
)
from stable_baselines3.ppo import PPO

from parl_agents.common.parl_base import PaRLBase
from parl_agents.common.policies import ActorCriticOptionPolicy, ActorCriticSeparatePolicy
from parl_annotations.annotated_tasks import AnnotatedTask


class HplanPPO(PaRLBase):
    """ Extend PaRLBase with option agent being PPO in SB3 """
    def __init__(
        self,
        # args for BaseAlgorithm
        parl_policy: Union[Type[BasePolicy], None],
        option_agent_policy: Union[Type[ActorCriticOptionPolicy], Type[ActorCriticSeparatePolicy]],
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
        create_eval_env: bool = False,       # 1 eval gym env for PaRL task
        monitor_wrapper: bool = False,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
        # additional kwargs for PaRLBase
        is_on_policy: bool = True,
        is_markovian_reward: bool = True,
        option_termination_reward: float = 1.0,
        option_step_cost: float = 1.0,
        option_penalty_cost: float = 1.0,
        option_terminal_cost: float = 0.0,
        reward_scale: float = 1.0,
        use_intrinsic_reward = True,
        # additional kwargs for PPO
        n_steps: int = 1024,                # this step applies to each option
        gamma: float = 1.0,
        batch_size: int = 32,               # In PPO, batch is per option
        n_epochs: int = 20,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: float = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        option_policy_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # check batch and buffer args parameter for PPO in advance
        assert batch_size > 1
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.buffer_size = self.n_steps * 1       # override buffer size for PPO
        assert self.buffer_size % batch_size == 0
        self.buffer_kwargs = dict(gamma=gamma, gae_lambda=gae_lambda)

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
            option_terminal_cost = option_terminal_cost,
            reward_scale=reward_scale,
            buffer_size=self.buffer_size,
            buffer_kwargs=self.buffer_kwargs,
            use_intrinsic_reward = use_intrinsic_reward
        )
        self.option_agent_policy = option_agent_policy
        self.option_policy_learning_rate = option_policy_learning_rate

        # init for PPO based on SB3 PPO class
        self.n_steps = n_steps
        assert self.n_steps >= self.max_episode_len
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.option_policy_kwargs = option_policy_kwargs

    def _setup_model(self):
        # setup PaRL model, high level policy, feature extractor
        super()._setup_model()

        print("{}._setup_model()".format(self.__class__.__name__))

        # create option agent and its NN model for default option

        if self.planning_task.name not in self.option_agents:
            self.create_option_agent(self.planning_task.name)

    def create_option_agent(self, option_name):
        """
            perform __init__, and __setup_model for RL agent code
            with better control on
            reset, policy, buffer creation and hyper parameter setup

            manually add steps appearing in
            PPO class, op/off-policy class, and base class
        """
        tensorboard_log_agent = self.tensorboard_log

        self.option_agents[option_name] = PPO(
            policy = self.option_agent_policy,
            env = self.env,
            learning_rate = self.option_policy_learning_rate,
            n_steps = self.n_steps,
            batch_size = self.batch_size,
            n_epochs = self.n_epochs,
            gamma = self.gamma,
            gae_lambda = self.gae_lambda,
            clip_range = self.clip_range,
            clip_range_vf = self.clip_range_vf,
            normalize_advantage = self.normalize_advantage,
            ent_coef = self.ent_coef,
            vf_coef = self.vf_coef,
            max_grad_norm = self.max_grad_norm,
            use_sde = False,
            sde_sample_freq = -1,
            target_kl = self.target_kl,
            tensorboard_log = tensorboard_log_agent,
            create_eval_env = False,                # 1 eval gym env for PaRL task; no additional env for option agent
            policy_kwargs = self.option_policy_kwargs,
            verbose = self.verbose,
            seed = self.seed,
            device = self.device,
            _init_setup_model = False)

        option_agent = self.option_agents[option_name]

        # manually do _setup model and avoid resetting env and setting random seeds again
        option_agent._setup_lr_schedule()
        self._add_option_buffer(option_name)        # create rollout buffer from PaRLBase
        option_agent.rollout_buffer = self.sample_buffer[option_name]

        option_agent.policy = self.option_agent_policy(
            option_agent.observation_space,
            option_agent.action_space, 
            option_agent.lr_schedule, 
            **self.option_policy_kwargs)
        option_agent.policy = option_agent.policy.to(self.device)

        # Initialize schedules for policy/value clipping in _setup model code for PPO.
        option_agent.clip_range = get_schedule_fn(option_agent.clip_range)
        if option_agent.clip_range_vf is not None:
            if isinstance(option_agent.clip_range_vf, (float, int)):
                assert option_agent.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"
            option_agent.clip_range_vf = get_schedule_fn(option_agent.clip_range_vf)

    def _setup_learn_option_agent(
        self,
        option_agent,
        total_timesteps: int,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ):
        """
            PPO has _setup_learn() defined only in BaseAlgorithm
            adapt it to avoid environment resets
        """
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
        tb_log_name: str = "",
        option_total_timesteps: int = 1e5,
    ):
        """ call _setup_learn from PaRLBase and BaseAlgorithm
            Then, call _setup_learn for each option agent, but don't use PPO._setup_learn() """
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
        log_interval: int = 1,
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
        # self.option_log_save_path = os.path.join(tensorboard_log, f"{tb_log_name}_{latest_run_id + 1}")

        iteration = 0
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

            n_rollout_steps = max(self.max_episode_len, self.n_steps)
            continue_training = self.collect_rollouts(self.env, callback, n_rollout_steps)
            if not continue_training:
                break

            iteration += 1
            # print("DBG::iteration={}".format(iteration))
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            for option_name in sorted(self.collected_option_names):
                option_agent = self.option_agents[option_name]
                option_agent._update_current_progress_remaining(option_agent.num_timesteps, self.option_total_timesteps)

            if log_interval is not None and iteration % log_interval == 0:
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                    if len(self.ep_success_buffer) > 0:
                        self.logger.record("rollout/ep_success", safe_mean([1 if is_success else 0
                                                                            for is_success in self.ep_success_buffer]))
                    else:
                        self.logger.record("rollout/ep_success", 0)
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            for option_name in sorted(set(self.collected_option_names)):
                option_agent = self.option_agents[option_name]
                option_agent.logger.record("{}/time/iterations".format(option_name), iteration, exclude="tensorboard")
                if len(option_agent.ep_info_buffer) > 0 and len(option_agent.ep_info_buffer[0]) > 0:
                    option_agent.logger.record("{}/rollout/ep_rew_mean".format(option_name),
                                       safe_mean([ep_info["r"] for ep_info in option_agent.ep_info_buffer]))
                    option_agent.logger.record("{}/rollout/ep_len_mean".format(option_name),
                                       safe_mean([ep_info["l"] for ep_info in option_agent.ep_info_buffer]))
                if len(option_agent.ep_success_buffer) > 0:
                    option_agent.logger.record("{}/rollout/ep_success".format(option_name), safe_mean([1 if is_success else 0
                                                                                  for is_success in option_agent.ep_success_buffer]))
                else:
                    option_agent.logger.record("{}/rollout/ep_success".format(option_name), 0)
                option_agent.logger.record("{}/time/time_elapsed".format(option_name), int(time.time() - option_agent.start_time), exclude="tensorboard")
                option_agent.logger.record("{}/time/total_timesteps".format(option_name), option_agent.num_timesteps, exclude="tensorboard")
                option_agent.logger.dump(step=option_agent.num_timesteps)

            # print(self.collected_option_names)
            # print(self.collected_pl_states)
            self.train()

        callback.on_training_end()
        return self

    def collect_rollouts(self, env, callback, n_rollout_steps):
        """ adapted from SB3 on_policy_algorithm to handle PaRL rollout"""
        assert self._last_obs is not None
        assert self._last_pl_state is not None
        assert env.num_envs == 1                            # state mapping already assumed num env is 1
        n_steps = 0

        for option_name in self.sample_buffer:
            self.sample_buffer[option_name].reset()         # on policy method cleans up buffer

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
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
            # from option_policy, it sees as if the episode was just began now
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            continue_rollout_cur_option = True

            while continue_rollout_cur_option and n_steps < n_rollout_steps:

                with th.no_grad():
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    actions, values, log_probs = cur_agent.policy.forward(obs_tensor)
                actions = actions.cpu().numpy()

                new_obs, rewards, dones, infos = env.step(actions)
                new_pl_state = self.state_map(new_obs)

                step_count_option += 1
                cur_agent.num_timesteps += self.n_envs      # increment agent counter
                self.num_timesteps += self.n_envs           # increment global counter

                callback.update_locals(locals())
                if callback.on_step() is False:
                    return False

                self._update_info_buffer(infos, dones)
                n_steps += 1

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

                if isinstance(self.action_space, gym.spaces.Discrete):
                    actions = actions.reshape(-1, 1)    # Reshape in case of discrete action

                # Handle timeout by bootstraping with value function
                # see GitHub issue #633 https://github.com/DLR-RM/stable-baselines3/issues/633
                for idx, done in enumerate(option_dones):
                    if (
                        done
                        and option_infos[idx].get("terminal_observation") is not None
                        and option_infos[idx].get("TimeLimit.truncated", False)
                    ):
                        terminal_obs = self.policy.obs_to_tensor(option_infos[idx]["terminal_observation"])[0]
                        with th.no_grad():
                            terminal_value = self.policy.predict_values(terminal_obs)[0]
                        option_rewards[idx] += self.gamma * terminal_value

                # In case of PPO, we don't need to store planning state transitions
                cur_buffer.add(self._last_obs, actions, option_rewards, self._last_episode_starts, values, log_probs)
                self._last_obs = new_obs
                self._last_episode_starts = option_dones
                self._last_pl_state = new_pl_state

            with th.no_grad():
                # Compute value for the last timestep
                values = cur_agent.policy.predict_values(obs_as_tensor(new_obs, self.device))
            cur_buffer.compute_returns_and_advantage(last_values=values, dones=dones)


        callback.on_rollout_end()

        return True

    def train(self):
        for option_name in sorted(set(self.collected_option_names)):
            arg_batch_size = self.batch_size
            buffer_used = self.sample_buffer[option_name].size()
            if buffer_used < 2:     # len 1 sample returns nan during advantage normalization!
                continue
            self.batch_size = min(self.batch_size, buffer_used)
            self.option_agents[option_name].train()
            self.batch_size = arg_batch_size

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
        super(HplanPPO, model)._setup_model()

        # here, load instead of creating option agents
        valid_option_names = set([option.name for option in model.strips_options])
        valid_option_names.add(model.planning_task.name)

        for f in os.listdir(path):
            if f.startswith("model--") and f.endswith(".zip"):
                option_name = f.replace("model--", "").replace(".zip", "")
                assert option_name in valid_option_names
                print("start loading {}".format(option_name))

                agent = PPO.load(
                    path=os.path.join(path, f),
                    env=model.env,
                    device=model.device
                )

                # internally loading will _setup_model() so take buffer from agent
                model.option_agents[option_name] = agent
                model.sample_buffer[option_name] = agent.rollout_buffer

        return model
