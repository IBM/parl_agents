from abc import abstractmethod
from collections import deque
from typing import Any, Iterable, Dict, List, Optional, Tuple, Type, Union
from random import choice
import os

import gym
import numpy as np
import torch as th

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer

from stable_baselines3.common.policies import BasePolicy
from parl_annotations.annotated_tasks import AnnotatedTask
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback
from stable_baselines3.common.save_util import (
    load_from_zip_file,
    recursive_getattr,
    recursive_setattr,
    save_to_zip_file
)
from stable_baselines3.common.utils import (
    get_schedule_fn,
)

from parl_annotations.pyperplan_planner import PyperplanPlanner
from parl_agents.common.buffers import OptionRolloutBuffer


class PaRLBase(BaseAlgorithm):
    """
    PaRLBase defines basic interface and common methods for PaRL algorithms
        _setup_learn, learn, train should be implemented in subclasses
    """
    def __init__(
        self,
        # args for BaseAlgorithm
        parl_policy: Union[Type[BasePolicy], None],
        env: Union[GymEnv, str, None],
        parl_policy_learning_rate: Union[float, Schedule],
        # additional args for PaRLBase
        parl_task: AnnotatedTask,
        max_episode_len: int,
        # kwargs for BaseAlgorithm
        parl_policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        support_multi_env: bool = False,     # assume n_env is 1
        create_eval_env: bool = False,      # create eval env outside of agent code
        monitor_wrapper: bool = False,      # wrap everything outside before creating agent
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
        # additional kwargs for PaRLBase
        is_on_policy: bool = True,
        is_markovian_reward: bool = True,
        option_termination_reward: float = 1.0,
        option_step_cost: float = 0.9,
        option_penalty_cost: float = 0.9,
        option_terminal_cost: float = 0.0,
        reward_scale: float = 1.0,
        buffer_size: int = 10000,           # total buffer size
        buffer_kwargs: Dict[Any, Any] = dict(gamma=1.0, gae_lambda=0.95),
        use_intrinsic_reward = True
    ):
        # init base attributes
        super().__init__(
            parl_policy,
            env,
            parl_policy,
            parl_policy_learning_rate,
            parl_policy_kwargs,
            tensorboard_log,
            verbose,
            device,
            support_multi_env,
            create_eval_env,
            monitor_wrapper,
            seed,
            use_sde,
            sde_sample_freq,
            supported_action_spaces
        )
        # alias in SB3 policy can predict something
        self.parl_policy_class = self.policy_class
        self.parl_policy_kwargs = self.policy_kwargs
        self.parl_policy_learning_rate = self.learning_rate

        # parl_task holds unwrapped env
        # if state mapping sees the internal states of env, this task binds to env passed to __init__
        self.parl_task = parl_task
        self._last_pl_state = None
        self.max_episode_len = max_episode_len
        # self.env.set_attr(attr_name="max_episode_len", value=self.max_episode_len)

        self.is_on_policy = is_on_policy
        self.is_markovian_reward = is_markovian_reward

        self.option_termination_reward = option_termination_reward
        self.option_step_cost = option_step_cost
        self.option_penalty_cost = option_penalty_cost
        self.option_terminal_cost = option_terminal_cost
        self.reward_scale = reward_scale

        self.sample_buffer = None
        self.buffer_size = buffer_size
        self.buffer_kwargs = buffer_kwargs

        self.option_info_buffer = None
        self.option_success_buffer = None

        self.use_intrinsic_reward = use_intrinsic_reward

    def _setup_model(self):
        print("{}._setup_model()".format("PaRLBase"))
        # set random seed at higher level agent only once
        self.set_random_seed(self.seed)

        self._setup_parl_task(self.parl_task)

        # cannot determine features extractor here\
        self._setup_parl_policy()

        # place holder for option agent objects and buffers
        self.sample_buffer = {}
        self.option_agents = {}

    def _setup_parl_task(self, parl_task):
        self.planning_task = parl_task.planning_task
        self.state_map = parl_task.rl_obs_to_pl_state
        self.strips_options = parl_task.strips_options
        self.option_name2ind = {option.name: ind for ind, option in enumerate(parl_task.strips_options)}
        self.planner = PyperplanPlanner(search_alg="astar2")
        self.num_planning_facts = len(parl_task.planning_facts)

    def _setup_parl_policy(self):
        # setup high level policy, this is just a dictionary if we don't use NN
        if self.parl_policy_class is None:
            self.parl_policy = dict()
            self.lr_schedule = None     # lr schedule for high-level agent
        else:
            # parl_policy_kwargs take care all required args for high level NN policy
            self.parl_policy = self.parl_policy_class(**self.parl_policy_kwargs)
            assert isinstance(self.parl_policy, BasePolicy)
            self.parl_policy = self.parl_policy.to(self.device)
            self.lr_schedule = get_schedule_fn(self.parl_policy_learning_rate)        

    def forward_parl_policy(self, pl_state, get_first=True):
        """ forward/appling parl_policy at pl_state.
            self.parl_policy.forward() will override this if it is NN """
        state_str = str(pl_state)
        if state_str not in self.parl_policy:
            return None

        if get_first:
            return self.parl_policy[state_str][0]
        else:
            return choice(self.parl_policy[state_str])

    def step_parl_policy(self, plan_as_policy):
        """ learning/recording parl_policy.
            self.parl_policy.optimzier.step() will override this if it is NN """
        for pl_state, pl_op in plan_as_policy:
            state_str = str(pl_state)
            if state_str not in self.parl_policy:
                self.parl_policy[state_str] = []
            if pl_op.name not in self.parl_policy[state_str]:
                self.parl_policy[state_str].append(pl_op.name)

    def _create_basic_buffer(self):
        if self.is_on_policy:
            return OptionRolloutBuffer(buffer_size=self.buffer_size, observation_space=self.observation_space,
            action_space=self.action_space, device=self.device,
            gae_lambda=self.buffer_kwargs["gae_lambda"],
            gamma=self.buffer_kwargs["gamma"], n_envs=self.n_envs)
        else:
            return ReplayBuffer(buffer_size=self.buffer_size, 
            observation_space=self.observation_space,
            action_space=self.action_space, 
            device=self.device,
            n_envs=self.n_envs)

    def _add_option_buffer(self, option_name):
        self.sample_buffer[option_name] = self._create_basic_buffer()

    @abstractmethod
    def create_option_agent(self, option_name):
        """ create RL agent for option learning """

    def decide_option_termination(self, option, new_pl_state, dones, infos):
        assert len(dones) == 1 and len(infos) == 1
        info, done = infos[0], dones[0]
        option_info = dict()
        option_info["terminal_observation"] = info.get("terminal_observation")
        option_done, continue_rollout_cur_option = False, True
        if done:  # gym env episode done
            if option is None: #and option.name == self.planning_task.name:
                if "is_success" in info and info["is_success"]:
                    option_done = True  # goal option reaching RL goal, current goal option done, reset env
                    option_info["is_success"] = True
                    option_info["Timelimit.truncated"] = False
                    continue_rollout_cur_option = False
                else:
                    option_done = True  # global episode is done so option ep is also done
                    option_info["is_success"] = False
                    option_info["Timelimit.truncated"] = True
                    continue_rollout_cur_option = False
            else:  # plan option rollout
                if self.check_option_termination(option, new_pl_state):
                    option_done = True  # plan option terminated, move to next option
                    option_info["is_success"] = True
                    option_info["Timelimit.truncated"] = False
                    continue_rollout_cur_option = False
                else:
                    option_done = True  # ep is done so option local ep is also done
                    option_info["is_success"] = False
                    option_info["Timelimit.truncated"] = True
                    continue_rollout_cur_option = False
        else:  # gym env continues but option ep can be done
            if option is None: # and option.name == self.planning_task.name:
                option_done = False  # goal option reaching RL goal, current goal option done, reset env
                continue_rollout_cur_option = True
            else:
                if self.check_option_termination(option, new_pl_state):
                    option_done = True  # plan option terminated, move to next option
                    option_info["is_success"] = True
                    option_info["Timelimit.truncated"] = False
                    continue_rollout_cur_option = False
                else:
                    option_done = False  # goal option reaching RL goal, current goal option done, reset env
                    continue_rollout_cur_option = True
        return dict(option_infos=[option_info],
                    option_dones=np.array([option_done]),
                    continue_rollout_cur_option=continue_rollout_cur_option)

    @staticmethod
    def check_option_termination(option, pl_state):
        if pl_state.issuperset(option.term_set) and pl_state.isdisjoint(option.del_set):
            return True
        else:
            return False

    def plan_option_intrinsic_reward(self, option, init_pl_state, new_pl_state, rewards, option_infos):
        assert len(option_infos) == 1 and len(rewards) == 1

        # option termination reward - step cost
        if not self.use_intrinsic_reward:
            return self.goal_option_shape_zero_reward(option_infos)
    
        shaped_reward = 0

        if self.is_markovian_reward:
            ref_pl_state = self._last_pl_state      # compute reward relative to the previous state
        else:
            ref_pl_state = init_pl_state            # compute reward relative to the first state

        # check context violation
        non_context_facts =  option.init_set | option.term_set
        max_context_dist = self.num_planning_facts - len(non_context_facts)
        context_dist = self.parl_task.dist_states(ref_pl_state, new_pl_state, non_context_facts)
        if max_context_dist > 0:
            shaped_reward -= self.option_penalty_cost * context_dist / max_context_dist

        # check terminated, if not, count violated pl_state variables
        missing_pos = option.term_set - new_pl_state          # they should appear
        spurious_neg = new_pl_state & option.del_set          # they shouldn't appear
        termset_dist = len(missing_pos) + len(spurious_neg)
        termset_dist_max = len(option.term_set) + len(option.del_set)
        shaped_reward -= self.option_terminal_cost * ( termset_dist / termset_dist_max )

        is_success = option_infos[0].get("is_success")
        if is_success:
            shaped_reward += self.option_termination_reward

        # unit step cost
        shaped_reward -= self.option_step_cost * (1.0 / self.max_episode_len)  # step cost

        shaped_reward *= self.reward_scale
        return [shaped_reward]

    def goal_option_shape_zero_reward(self, option_infos):
        assert len(option_infos) == 1
        is_success = option_infos[0].get("is_success")
        if is_success:
            shaped_reward = self.option_termination_reward
        else:
            shaped_reward = 0
        shaped_reward *= self.reward_scale
        return [shaped_reward]

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
        **kwargs
    ):
        total_timesteps, callback = super()._setup_learn(total_timesteps,
                                                         eval_env,
                                                         callback,
                                                         eval_freq,
                                                         n_eval_episodes,
                                                         log_path,
                                                         reset_num_timesteps,
                                                         tb_log_name)
        if self._last_obs is not None:
            self._last_pl_state = self.state_map(self._last_obs)
        return total_timesteps, callback

    def save(self, path, exclude= None, include=None) -> None:
        for option_name in self.option_agents:
            agent = self.option_agents[option_name]
            agent_path = os.path.join(path, "model--" + option_name + ".zip")
            agent.save(agent_path, exclude, include)
