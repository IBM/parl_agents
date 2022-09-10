import argparse
import os
from functools import partial
import pathlib

import gym
import gym_minigrid
from gym_minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
import torch

import parl_minigrid
from parl_minigrid.envs.wrappers import EpisodeTerminationWrapper

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.utils import get_latest_run_id

from stable_baselines3.dqn.policies import DQNPolicy
from parl_agents.common.callbacks import TerminationSecCallback, CustumEvalCallback
from parl_agents.nn_models.babyai_cnn import BabyAIFullyObsSmallCNN, BabyAIFullyObsSmallCNNDict
from parl_agents.wrappers.step_cost import MiniGridCostWrapper
from parl_agents.wrappers.obs_wrapper import AugmentDiscreteToImg
from parl_agents.hplan_dqn.hplan_dqn2 import HplanDQN2
from parl_minigrid.annotations.strips.annotated_task import MazeRoomsAnnotatedTask
from parl_agents.common.evaluation import evaluate_parl_policy


if __name__ == "__main__":
    print("cuda.is_available:{}".format(torch.cuda.is_available()))
    print("cuda.device_count:{}".format(torch.cuda.device_count()))
    if torch.cuda.is_available():
        print("cuda.current_device:{}".format(torch.cuda.current_device()))
        print("cuda.get_device_name:{}".format(torch.cuda.get_device_name(torch.cuda.current_device())))
    else:
        print("no cuda device")

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="DDQN")
    parser.add_argument("--env", default="MazeRooms-8by8-DoorKey-v0")
    parser.add_argument("--fe", type=str, default="dqn_single")        
    parser.add_argument("--tfroot", type=str, default="results")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ir", type=int, default=1)
    parser.add_argument("--modelroot", type=str, default="models")
    parser.add_argument("--load", type=int, default=0)
    args = parser.parse_args()

    TS = 20000000
    EF = 10240
    N_TS=1000
    N_ES=100

    task_env = gym.make(args.env, train_mode=True, max_steps=1024, num_train_seeds=N_TS, num_test_seeds=N_ES)
    parl_task_temp = MazeRoomsAnnotatedTask(task_env)
    total_num_options = len(parl_task_temp.strips_options) + 1

    train_env = gym.make(args.env, train_mode=True, max_steps=1024, num_train_seeds=N_TS, num_test_seeds=N_ES)
    train_env = FullyObsWrapper(train_env)
    train_env = ImgObsWrapper(train_env)
    train_env = AugmentDiscreteToImg(train_env, discrete_feature_size=total_num_options)     # here
    train_env = MiniGridCostWrapper(train_env)
    train_env = EpisodeTerminationWrapper(train_env)
    train_env = Monitor(train_env)

    tensorboard_log = os.path.join(os.path.dirname(__file__), args.tfroot, "seed-{}".format(args.seed), args.env)
    pathlib.Path(tensorboard_log).mkdir(parents=True, exist_ok=True)
    tb_log_name = args.agent

    parl_task = MazeRoomsAnnotatedTask(train_env)

    BabyAIFE = BabyAIFullyObsSmallCNNDict
    policy_kwargs = dict(
        features_extractor_class = BabyAIFE,
        features_extractor_kwargs = dict(features_dim=128),
        net_arch = [64, 64],
        normalize_images = False
    )
    model = HplanDQN2(
        parl_policy=None,
        option_agent_policy=DQNPolicy,
        env=train_env,
        parl_policy_learning_rate=1,
        option_policy_learning_rate=0.0005,
        device="auto",
        parl_task=parl_task,
        max_episode_len=1024,
        parl_policy_kwargs=None,
        tensorboard_log=tensorboard_log,
        verbose=1,
        option_termination_reward=1.0,
        option_step_cost=0.9,                  # 0.00008789062
        option_penalty_cost=0.0,       # 0.9*0.9/1024,
        option_terminal_cost=0.0,
        reward_scale=1,
        use_intrinsic_reward=False,
        # DDQN
        buffer_size=102400,         # 100 episodes
        learning_starts=1024,
        batch_size=256,
        tau=1.0,
        gamma=0.90,
        train_freq=(1, 'step'),     # due to current option unrolling, roll out loop expires when one escape from option
        gradient_steps=64,         # this will be effectively 1024, 1 step, 1 update
        target_update_interval=4096,    # this guarantees at least 2 episode
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        max_grad_norm=10,
        option_policy_kwargs=policy_kwargs
    )


    print(model.__class__.__name__)
    for k, v in model.__dict__.items():
        print("{}:{}".format(k, v))

    model._setup_model()

    model.learn(
        total_timesteps=TS,
        eval_env=None,
        log_interval=1,      # per episode
        tb_log_name=tb_log_name,            # this is tf log name
        option_total_timesteps=TS
        )

