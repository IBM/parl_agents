import argparse
import os
import time

import gym
import gym_minigrid
from gym_minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
import torch
import numpy as np

import parl_minigrid
from parl_minigrid.envs.wrappers import EpisodeTerminationWrapper

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean

from parl_agents.common.policies import ActorCriticSeparatePolicy
from parl_agents.nn_models.babyai_cnn import BabyAIFullyObsSmallCNN
from parl_agents.hplan_ppo.hplan_ppo import HplanPPO
from parl_minigrid.annotations.strips.annotated_task import MazeRoomsAnnotatedTask
from parl_agents.tuning.utils import TuneObjectiveCallback

import optuna
from optuna.integration.skopt import SkoptSampler
from optuna.pruners import BasePruner, MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import BaseSampler, RandomSampler, TPESampler

N_TS=1
N_ES=1


def sample_params(trial: optuna.Trial):
    option_step_cost = trial.suggest_uniform("option_step_cost", 0, 0.9)
    option_penalty_cost = trial.suggest_loguniform("option_penalty_cost", 1e-6, 0.9)
    return dict(option_step_cost=option_step_cost, option_penalty_cost=option_penalty_cost)


def create_env(args):
    train_env = gym.make(args.env, train_mode=True, max_steps=1024, num_train_seeds=N_TS, num_test_seeds=N_ES)
    train_env = FullyObsWrapper(train_env)
    train_env = ImgObsWrapper(train_env)
    train_env = EpisodeTerminationWrapper(train_env)
    train_env = Monitor(train_env)    
    parl_task = MazeRoomsAnnotatedTask(train_env)
    return train_env, parl_task


class ParameterTuner:
    def __init__(self, args, hyper_params):
        self.study_name = " ".join([args.study_name, args.env])
        self.n_trials = args.trials
        self.n_timesteps = args.timesteps
        self.hyper_params = hyper_params
        self.args = args

    def objective(self, trial: optuna.Trial) -> float:
        train_env, parl_task = create_env(self.args)
        self.hyper_params.update(sample_params(trial))

        acsp_option_policy_kwargs = dict(
            policy_features_extractor=None,
            value_features_extractor=None,
            features_extractor_class=BabyAIFullyObsSmallCNN,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
            normalize_images=False
        )
        option_policy_kwargs = acsp_option_policy_kwargs
        option_agent_policy = ActorCriticSeparatePolicy
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
            tensorboard_log=None,
            verbose=0,
            option_termination_reward=1.0,
            option_terminal_cost=0.0,
            use_intrinsic_reward=True,
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
            **self.hyper_params
        )
        model._setup_model()
        model.trial = trial
        tune_obj_call_back = TuneObjectiveCallback(trial)
        try:
            model.learn(total_timesteps=self.n_timesteps, callback=tune_obj_call_back)
            model.env.close()
        except AssertionError as e:
            model.env.close()
            print(e)
            raise optuna.exceptions.TrialPruned()

        is_pruned = tune_obj_call_back.is_pruned
        mean_ep_length = tune_obj_call_back.mean_ep_length

        del model.env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return mean_ep_length

    def hyperparam_optimization(self):
        self.tensorboard_log = None

        sampler = TPESampler(n_startup_trials=2, seed=0)
        pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=0)
        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            study_name=self.study_name,
            load_if_exists=True,
            direction="minimize",
        )
        try:
            study.optimize(self.objective, n_trials=self.n_trials)
        except KeyboardInterrupt:
            pass

        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        print("Value: ", trial.value)
        print("Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        report_name = (
            f"report_{self.study_name}_steps-{self.n_timesteps}_{int(time.time())}.csv"
        )
        log_path = os.path.join(os.path.dirname(__file__), report_name)
        # Write report
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        study.trials_dataframe().to_csv(log_path)



if __name__ == "__main__":
    print("cuda.is_available:{}".format(torch.cuda.is_available()))
    print("cuda.device_count:{}".format(torch.cuda.device_count()))
    if torch.cuda.is_available():
        print("cuda.current_device:{}".format(torch.cuda.current_device()))
        print("cuda.get_device_name:{}".format(torch.cuda.get_device_name(torch.cuda.current_device())))
    else:
        print("no cuda device")


    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", type=str, default="hplanppo tuning")    
    parser.add_argument("--env", type=str, default="MazeRooms-8by8-DoorKey-v0")
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()

    hyper_params = {}
    exp_manager = ParameterTuner(args, hyper_params)
    exp_manager.hyperparam_optimization()