import argparse
import os
from functools import partial
import pathlib

import gym
import gym_minigrid
from gym_minigrid.wrappers import ImgObsWrapper
import torch

import parl_minigrid
from parl_minigrid.envs.wrappers import FullyObsWrapper, EpisodeTerminationWrapper

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.dqn.policies import DQNPolicy

from parl_agents.dqn.ddqn import DDQN
from parl_agents.common.callbacks import TerminationSecCallback
from parl_agents.nn_models.babyai_cnn_ne import BabyAIFullyObsCNN, BabyAIFullyObsSmallCNN
from parl_agents.wrappers.step_cost import MiniGridCostWrapper, DQNRewardWrapper


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
    parser.add_argument("--tfroot", type=str, default="results")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--modelroot", type=str, default="models")
    args = parser.parse_args()

    MAX_STEPS = 2048
    TS = int(1e7)
    EF = 2048*100
    N_TS=int(1e6)
    N_ES=1000
    EVAL_S=100
    TIME_LIMIT_HR=23



    train_env = gym.make(args.env, train_mode=True, max_steps=MAX_STEPS, num_train_seeds=N_TS, num_test_seeds=N_ES)
    train_env = FullyObsWrapper(train_env)
    train_env = ImgObsWrapper(train_env)
    # train_env = MiniGridCostWrapper(train_env)
    # train_env = DQNRewardWrapper(train_env, reward_scale=100, step_cost=0)
    train_env = EpisodeTerminationWrapper(train_env)
    train_env = Monitor(train_env)

    eval_env =  gym.make(args.env, train_mode=False, max_steps=MAX_STEPS, num_train_seeds=N_TS, num_test_seeds=N_ES)
    eval_env = FullyObsWrapper(eval_env)
    eval_env= ImgObsWrapper(eval_env)
    # eval_env = MiniGridCostWrapper(eval_env)
    # train_env = DQNRewardWrapper(train_env, reward_scale=100, step_cost=0)
    eval_env = EpisodeTerminationWrapper(eval_env)
    eval_env = Monitor(eval_env)

    # set up path variables
    # utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)
    # save_path = os.path.join(tensorboard_log, f"{tb_log_name}_{latest_run_id + 1}")
    #   under this directory, store tfevent
    tensorboard_log = os.path.join(os.path.dirname(__file__), args.tfroot, "seed-{}".format(args.seed), args.env)
    pathlib.Path(tensorboard_log).mkdir(parents=True, exist_ok=True)
    tb_log_name = args.agent

    model_root = os.path.join(os.path.dirname(__file__), args.modelroot)
    model_dir = "--".join([args.agent, args.env, "seed={}".format(args.seed)])
    model_trial = get_latest_run_id(model_root, model_dir)
    model_path = os.path.join(model_root, model_dir + "_{}".format(model_trial))
    model_path = os.path.join(model_path, "model.zip")

    term_callback = TerminationSecCallback(verbose=0, time_limit=3600*TIME_LIMIT_HR)
    eval_callback = EvalCallback(eval_env, eval_freq=EF, n_eval_episodes=EVAL_S,  deterministic=False)

    BabyAIFE = BabyAIFullyObsCNN
    if args.env == "MazeRooms-8by8-DoorKey-v0":
        BabyAIFE = BabyAIFullyObsSmallCNN

    policy_kwargs = dict(
        features_extractor_class = BabyAIFE,
        features_extractor_kwargs = dict(features_dim=128),
        net_arch = [128, 128],
        normalize_images = False
    )
    # https://arxiv.org/pdf/2012.08621.pdf, https://arxiv.org/pdf/2006.12122.pdf
    print("create model:{}".format(model_path))
    model = DDQN(
        policy=DQNPolicy,
        env=train_env,
        learning_rate=0.000681590954892754,
        buffer_size=409600,
        learning_starts=7876,
        batch_size=128,
        tau=1.0,
        gamma=0.9514622035384503,
        train_freq=(2862, 'step'),
        gradient_steps=142,
        target_update_interval=8230,
        exploration_fraction=0.3312761633788077,       # if we load, this will change...; so set TS high and finish by time
        exploration_initial_eps=1.0,
        exploration_final_eps=0.189177300078208,
        max_grad_norm=10,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        verbose=0
    )

    print(model.__class__.__name__)
    for k, v in model.__dict__.items():
        print("{}:{}".format(k, v))

    model.learn(
        total_timesteps=TS,
        callback=[term_callback, eval_callback],
        eval_freq=EF,  # per sample
        n_eval_episodes=EVAL_S,
        eval_env=None,
        log_interval=1,      # per episode
        tb_log_name=tb_log_name            # this is tf log name
        )

    print("save model:{}".format(model_path))
    model.save(path=model_path)
