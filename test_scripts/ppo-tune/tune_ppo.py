import argparse
import os
import time

import gym
import gym_minigrid
from gym_minigrid.wrappers import ImgObsWrapper
import torch
import numpy as np

import parl_minigrid
from parl_minigrid.envs.wrappers import FullyObsWrapper, EpisodeTerminationWrapper

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean

from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import ActorCriticPolicy
from parl_agents.common.policies import ActorCriticSeparatePolicy
from parl_agents.nn_models.babyai_cnn_ne import BabyAIFullyObsSmallCNN
from parl_agents.tuning.utils import TuneObjectiveCallback

import optuna
from optuna.integration.skopt import SkoptSampler
from optuna.pruners import BasePruner, MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import BaseSampler, RandomSampler, TPESampler

N_TS = int(1e3)
N_ES = 100


def sample_params(trial: optuna.Trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [4, 16, 32, 64, 128, 256, 512])
    n_epochs = trial.suggest_int("n_epochs", 10, 100, 10)
    gamma = trial.suggest_loguniform("gamma", 0.9, 1.0)
    ent_coef = trial.suggest_loguniform("ent_coef", 1e-4, 1e-2)   # make more random actions 0.01 ~ 0.0001
    vf_coef = trial.suggest_uniform("vf_coef", 0.1, 0.9)
    max_grad_norm = trial.suggest_uniform("max_grad_norm", 0.1, 10)
    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm
    }

def create_env(args):
    train_env = gym.make(args.env, train_mode=True, max_steps=2048, num_train_seeds=N_TS, num_test_seeds=N_ES)
    train_env = FullyObsWrapper(train_env)
    train_env = ImgObsWrapper(train_env)
    train_env = EpisodeTerminationWrapper(train_env)
    train_env = Monitor(train_env)
    return train_env


class ParameterTuner:
    def __init__(self, args, hyper_params):
        self.study_name = "--".join([args.study_name, args.env])
        self.n_trials = args.trials
        self.n_timesteps = args.timesteps
        self.hyper_params = hyper_params
        self.args = args

    def objective(self, trial: optuna.Trial) -> float:
        train_env = create_env(self.args)
        self.hyper_params.update(sample_params(trial))

        policy_kwargs = dict(
            features_extractor_class=BabyAIFullyObsSmallCNN,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=[dict(pi=[128, 128], vf=[128, 128])],
            normalize_images=False
        )

        model = PPO(
            policy=ActorCriticPolicy,
            env=train_env,
            policy_kwargs=policy_kwargs,
            # learning_rate=2.5e-4,
            n_steps=2048,
            # batch_size=256,
            # n_epochs=10,
            # gamma=0.99,
            gae_lambda=0.95,
            # ent_coef=0.01,
            # vf_coef=0.5,
            # max_grad_norm=0.5,
            tensorboard_log=None,
            verbose=0,
            **self.hyper_params
        )
        model._setup_model()
        model.trial = trial
        tune_obj_call_back = TuneObjectiveCallback(trial)
        try:
            model.learn(total_timesteps=self.n_timesteps,
                        callback=tune_obj_call_back)
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

        sampler = TPESampler(n_startup_trials=5, seed=int(time.time()))
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=0)
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
        # os.makedirs(os.path.dirname(log_path), exist_ok=True)
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
    parser.add_argument("--study-name", type=str, default="ppo-tuning")
    parser.add_argument("--env", type=str, default="MazeRooms-8by8-DoorKey-v0")
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()

    hyper_params = {}
    exp_manager = ParameterTuner(args, hyper_params)
    exp_manager.hyperparam_optimization()
