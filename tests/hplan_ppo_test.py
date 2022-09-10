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
from stable_baselines3.ppo.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_latest_run_id

from parl_agents.common.policies import ActorCriticSeparatePolicy, ActorCriticOptionPolicy
from parl_agents.common.callbacks import TerminationSecCallback, CustumEvalCallback
from parl_agents.nn_models.babyai_cnn import BabyAIFullyObsCNN, BabyAIFullyObsSmallCNN
from parl_agents.hplan_ppo.hplan_ppo import HplanPPO
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
    parser.add_argument("--agent", default="HplanPPO")
    parser.add_argument("--env", default="MazeRooms-2by2-Locked-v0")
    parser.add_argument("--fe", type=str, default="ac_duel")
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

    train_env = gym.make(args.env, train_mode=True, max_steps=1024, num_train_seeds=N_TS, num_test_seeds=N_ES)
    train_env = FullyObsWrapper(train_env)
    train_env = ImgObsWrapper(train_env)
    train_env = EpisodeTerminationWrapper(train_env)
    train_env = Monitor(train_env)

    eval_env = gym.make(args.env, train_mode=False, max_steps=1024, num_train_seeds=N_TS, num_test_seeds=N_ES)
    eval_env = FullyObsWrapper(eval_env)
    eval_env = ImgObsWrapper(eval_env)
    eval_env = EpisodeTerminationWrapper(eval_env)
    eval_env = Monitor(eval_env)

    tensorboard_log = os.path.join(os.path.dirname(__file__), args.tfroot, "seed-{}".format(args.seed), args.env)
    pathlib.Path(tensorboard_log).mkdir(parents=True, exist_ok=True)
    tb_log_name = args.agent

    model_root = os.path.join(os.path.dirname(__file__), args.modelroot)
    model_dir = "--".join([args.agent, args.env, "seed={}".format(args.seed), 
                             "fe={}".format(args.fe), "ir={}".format(args.ir)])
    model_trial = get_latest_run_id(model_root, model_dir)
    model_path = os.path.join(model_root, model_dir + "_{}".format(model_trial))
    if args.load == 1:
        next_model_path = os.path.join(model_root, model_dir + "_{}".format(model_trial+1))

    term_callback = TerminationSecCallback(verbose=0, time_limit=3600 * 23)
    eval_callback = CustumEvalCallback(eval_env, eval_freq=EF, n_eval_episodes=20,
                                       deterministic=False, evaluator=evaluate_parl_policy)

    parl_task = MazeRoomsAnnotatedTask(train_env)

    BabyAIFE = BabyAIFullyObsCNN
    if args.env == "MazeRooms-8by8-DoorKey-v0":
        BabyAIFE = BabyAIFullyObsSmallCNN

    # ActorCriticPolicy
    ac_option_policy_kwargs = dict(
        features_extractor=None,  # ActorCriticOptionPolicy this is mandatory arg
        features_extractor_class=BabyAIFE,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        normalize_images=False
    )

    # ActorCriticOptionPolicy
    fe_env = DummyVecEnv([lambda: train_env])  # done in base_class.py
    fe_env = VecTransposeImage(fe_env)

    acop_option_policy_kwargs = dict(
        features_extractor=BabyAIFE(fe_env.observation_space),  # shared FE needs to know this wrapped space
        features_extractor_class=BabyAIFE,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        normalize_images=False
    )

    # ActorCriticSeparatePolicy
    acsp_option_policy_kwargs = dict(
        policy_features_extractor=BabyAIFE(fe_env.observation_space),
        # shared FE needs to know this wrapped space
        value_features_extractor=BabyAIFE(fe_env.observation_space),
        # shared FE needs to know this wrapped space
        features_extractor_class=BabyAIFE,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        normalize_images=False
    )

    option_policy_kwargs = ac_option_policy_kwargs
    option_agent_policy = ActorCriticPolicy
    if args.fe == 'ac_shared':  # AC shared FE
        option_policy_kwargs = acop_option_policy_kwargs
        option_agent_policy = ActorCriticOptionPolicy
    elif args.fe == 'ac_duel':  # AC separate
        option_policy_kwargs = acsp_option_policy_kwargs
        option_agent_policy = ActorCriticSeparatePolicy
        option_policy_kwargs['policy_features_extractor'] = None
        option_policy_kwargs['value_features_extractor'] = None
    elif args.fe == 'ac_duel_shared':  # AC separate shared FE
        option_policy_kwargs = acsp_option_policy_kwargs
        option_agent_policy = ActorCriticSeparatePolicy
    print("option_policy_kwargs")
    for k, v in option_policy_kwargs.items():
        print("{}:{}".format(k, v))

    if args.load == 0:
        print("create model:{}".format(model_path))
        model = HplanPPO(
            parl_policy=None,
            option_agent_policy=option_agent_policy,
            env=train_env,
            parl_policy_learning_rate=1,
            option_policy_learning_rate=2.5e-4,
            #
            device="auto",
            parl_task=parl_task,
            max_episode_len=1024,
            parl_policy_kwargs=None,
            tensorboard_log=tensorboard_log,
            verbose=1,
            option_termination_reward=1.0,
            option_step_cost=0.09,                  # 0.00008789062     10% of 0.9
            option_penalty_cost=0.00079101562,       # 0.9*0.9/1024,     90% of 0.9 div by max len
            option_terminal_cost=0.0,
            use_intrinsic_reward=True if args.ir == 1 else False,
            #
            n_steps=2048,  # buffer size can go larger like 1000 or higher
            batch_size=256,  # order of 10s
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,  # make more random actions 0.01 ~ 0.0001
            vf_coef=0.5,  # value estimate increases with more reward/ loss decreases when stable
            max_grad_norm=0.5,
            option_policy_kwargs=option_policy_kwargs,
        )
    else:
        print("load model:{}".format(model_path))
        load_kwargs = dict(
            parl_policy=None,
            option_agent_policy=option_agent_policy,
            parl_policy_learning_rate=1,
            option_policy_learning_rate=2.5e-4,
            #
            device="auto",
            parl_task=parl_task,
            max_episode_len=1024,
            parl_policy_kwargs=None,
            tensorboard_log=tensorboard_log,
            verbose=1,
            option_termination_reward=1.0,
            option_step_cost=0.09,                  # 0.00008789062     10% of 0.9
            option_penalty_cost=0.00079101562,       # 0.9*0.9/1024,     90% of 0.9 div by max len
            option_terminal_cost=0.0,
            use_intrinsic_reward=True if args.ir == 1 else False,
            #
            n_steps=2048,  # buffer size can go larger like 1000 or higher
            batch_size=256,  # order of 10s
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,  # make more random actions 0.01 ~ 0.0001
            vf_coef=0.5,  # value estimate increases with more reward/ loss decreases when stable
            max_grad_norm=0.5,
            option_policy_kwargs=option_policy_kwargs            
        )
        model = HplanPPO.load(path=model_path, env=train_env, **load_kwargs)

    print(model.__class__.__name__)
    for k, v in model.__dict__.items():
        print("{}:{}".format(k, v))

    model._setup_model()

    model.learn(
        total_timesteps=TS,
        # callback=[term_callback, eval_callback],
        #eval_freq=1024,  # per sample
        #n_eval_episodes=5,
        eval_env=None,  # pass this through eval_callback
        log_interval=1,  # per iteration
        option_total_timesteps=TS     # this arg doesn't play any role since rollout is done by PaRLBase
    )

    if args.load == 1:
        model_path = next_model_path
    print("save models to {}".format(model_path))
    model.save(path=model_path)