import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict

import gym
import numpy as np

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped

from parl_agents.common.parl_base import PaRLBase


def evaluate_parl_policy(
        model: PaRLBase,
        env: Union[gym.Env, VecEnv],
        parl_task,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    adapt stable_baselines3.common.evaluation.evaluate_policy for PaRL

    this is rollout in hplan_ppo. TODO refactor rollout? can be common for DDQN/PPO?
    """

    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    # option_episode_rewards = defaultdict(list)
    # option_episode_lengths = defaultdict(list)

    assert n_envs == 1
    episode_counts = 0
    episode_count_targets = n_eval_episodes

    while episode_counts < episode_count_targets:
        # start evaluating PaRL model by rollingout 1 episode

        current_rewards = np.zeros(n_envs)
        current_lengths = np.zeros(n_envs, dtype="int")
        observations = env.reset()
        # pl_state = model.state_map(observations[0])
        pl_state = parl_task.rl_obs_to_pl_state(observations[0])

        states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        done = False
        collected_pl_states = []
        collected_option_names = []
        pl_dead_end = False
        while not done:

            collected_pl_states.append(pl_state)
            option_not_exist = False
            cur_option = None
            cur_agent = None
            if parl_task.planning_task.goal_reached(pl_state):
                cur_option_name = model.planning_task.name
            else:
                pl_state_str = str(pl_state)
                if pl_state_str not in model.parl_policy:
                    parl_task.planning_task.initial_state = pl_state
                    plan_as_policy = model.planner.solve(parl_task.planning_task)

                    # evalutor should not update symbolic policy of agent
                    # if plan_as_policy is not None:
                    #     model.step_parl_policy(plan_as_policy)
                    # else:
                    if plan_as_policy is None:
                        pl_dead_end = True

                # during evaluation, if agent encounters deadend, use goal option
                if not pl_dead_end:
                    # during evaluation it maybe possible not to have option created
                    plan_action = model.forward_parl_policy(pl_state, get_first=True)
                    if plan_action is None:
                        cur_option = None  # no strips option obect for goal option
                        cur_option_name = model.planning_task.name
                    else:
                        # model.strips_options is parl_task.strips_options
                        # train task and eval task should have identical options
                        cur_option = model.strips_options[model.option_name2ind[plan_action]]
                        cur_option_name = cur_option.name
                else:
                    cur_option = None       # no strips option obect for goal option
                    cur_option_name = model.planning_task.name
                pl_dead_end = False

            if cur_option_name in model.option_agents:
                cur_agent = model.option_agents[cur_option_name]
            else:
                # use goal option as emergency policy during evaluation
                cur_option_name = model.planning_task.name
                cur_agent = model.option_agents[cur_option_name]
            collected_option_names.append(cur_option_name)

            # start unrolling options
            option_episode_starts = np.ones((env.num_envs,), dtype=bool)
            continue_rollout_cur_option = True
            while continue_rollout_cur_option:
                # sample transition from the current option agent
                actions, states = cur_agent.predict(observations,
                                                    state=states,
                                                    episode_start=option_episode_starts,
                                                    deterministic=deterministic)
                observations, rewards, dones, infos = env.step(actions)
                current_rewards += rewards
                current_lengths += 1
                # pl_state = model.state_map(observations[0])
                pl_state = parl_task.rl_obs_to_pl_state(observations[0])

                reward = rewards[0]
                done = dones[0]
                info = infos[0]
                episode_starts[0] = done

                if callback is not None:
                    callback(locals(), globals())

                if done:
                    if is_monitor_wrapped:
                        if "episode" in info.keys():
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            episode_counts += 1
                    else:
                        episode_rewards.append(current_rewards[0])
                        episode_lengths.append(current_lengths[0])
                        episode_counts += 1
                        # reset current episode rewards and length
                        current_rewards = np.zeros(n_envs)
                        current_lengths = np.zeros(n_envs, dtype="int")

                # decide_option_termination
                option_done, option_info = False, dict()
                if done:        # done from environment
                    hrl_ep_done = True
                    if cur_option is None: # and cur_option_name == model.planning_task.name:
                        # due to Monitor Wrapper and TerminationWrapper we can check is_success, TimeLlimit
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
                        if model.check_option_termination(cur_option, pl_state):
                            option_done = True  # plan option terminated, move to next option
                            option_info["is_success"] = True
                            option_info["Timelimit.truncated"] = False
                            continue_rollout_cur_option = False
                        else:
                            option_done = True  # ep is done so option local ep is also done
                            option_info["is_success"] = False
                            option_info["Timelimit.truncated"] = True
                            continue_rollout_cur_option = False
                # gym env continues but option ep can be done
                else:
                    if cur_option is None: # and option.name == self.planning_task.name:
                        option_done = False  # goal option reaching RL goal, current goal option done, reset env
                        continue_rollout_cur_option = True
                    else:
                        if model.check_option_termination(cur_option, pl_state):
                            option_done = True  # plan option terminated, move to next option
                            option_info["is_success"] = True
                            option_info["Timelimit.truncated"] = False
                            continue_rollout_cur_option = False
                        else:
                            option_done = False  # goal option reaching RL goal, current goal option done, reset env
                            continue_rollout_cur_option = True

                # if option_done:
                #     option_episode_rewards[cur_option_name].append(option_current_rewards)
                #     option_episode_lengths[cur_option_name].append(option_current_length)

                if render:
                    env.render()

        # print("Evaluation: {}/{}".format(episode_counts+1, episode_count_targets))
        # s = iter(colected_pl_states)
        # a = iter(collected_option_names)
        # print("num_options:{}".format(len(collected_option_names)))
        # while True:
        #     try:
        #         print(next(str(s)))
        #         print(next(a))
        #     except:
        #         break

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward