"""
Extending SB3 Policy classes to support PaRL agents

* In HRL architecture, PaRL agent holds feature extractor objects and
  the option agent policy network will share it.
* In HRL architecture, high level NN policy interface can be different from
  option agent
"""
import collections
import copy
from inspect import trace
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor
)
from stable_baselines3.common.type_aliases import Schedule

from parl_agents.nn_models.utils import initialize_parameters


class ActorCriticOptionPolicy(ActorCriticPolicy):
    """
    Only Two changes,
    (1) use provided feature_extractor object
    (2) use initialize_parameters for initializing parameters
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        features_extractor: Optional[BaseFeaturesExtractor],       # if this is None, then this is ActorCriticPolicy
        net_arch: List[Union[int, Dict[str, List[int]]]],
        # default kwargs works
        activation_fn: Type[nn.Module] = nn.Tanh,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = False,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # cannot call super class method since it will attempt to create feature extractor
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super(ActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            features_extractor=features_extractor,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        # Default network architecture, from stable-baselines
        assert net_arch is not None     # net_arch = [dict(pi=[128, 64], vf=[128, 64])]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = False

        if self.features_extractor is None:
            self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        for module in [self.mlp_extractor, self.action_net, self.value_net]:
            module.apply(initialize_parameters)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(feature_extractor=self.feature_extractor)
        )
        return data


class ActorCriticSeparatePolicy(ActorCriticPolicy):
    """
    Change on top ActorCriticOptionPolicy
    when features_extractor object is given, use it to share across options policy nets

    policy and value train separate features_extractors
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        feature_scale: int = 1,         # multy this by feature dim of features extractor dimension; 3x3 rooms it is 2
        policy_features_extractor: Optional[BaseFeaturesExtractor]= None,  # if this is None, then this is ActorCriticPolicy
        value_features_extractor: Optional[BaseFeaturesExtractor]=None,  # if this is None, then this is ActorCriticPolicy
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = False,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = False,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super(ActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            features_extractor=policy_features_extractor,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = False

        self.policy_features_extractor = self.features_extractor
        self.value_features_extractor = value_features_extractor

        if policy_features_extractor is None:
            self.policy_features_extractor = features_extractor_class(observation_space=self.observation_space, **self.features_extractor_kwargs)
        else:
            self.policy_features_extractor = self.features_extractor
        self.features_dim = self.policy_features_extractor.features_dim
        # self.default_features_dim = self.features_dim * feature_scale      # larger layout needs to multiply numbers to match dim

        if value_features_extractor is None:
            self.value_features_extractor = features_extractor_class(observation_space=self.observation_space, **self.features_extractor_kwargs)
        else:
            self.value_features_extractor = value_features_extractor

        self.normalize_images = normalize_images        # using custom extract_feature won't call preprocess_obs
        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:

        actor_arch, critic_arch = self.net_arch[0]['pi'], self.net_arch[0]['vf']
        last_layer_dim_pi = self.features_dim
        last_layer_dim_vf= self.features_dim

        self.policy_features_extractor = self.policy_features_extractor.to(self.device)
        policy_net = []
        for pi_layer_size in actor_arch:
            policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
            policy_net.append(self.activation_fn())
            last_layer_dim_pi = pi_layer_size
        latent_dim_pi = last_layer_dim_pi        
        self.policy_net = nn.Sequential(*policy_net).to(self.device)

        if isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_features_extractor = self.value_features_extractor.to(self.device)
        value_net = []
        for vf_layer_size in critic_arch:
            value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
            value_net.append(self.activation_fn())
            last_layer_dim_vf = vf_layer_size
        value_net.append(nn.Linear(last_layer_dim_vf, 1))
        self.value_net = nn.Sequential(*value_net).to(self.device)

        for module in [self.policy_net, self.action_net, self.value_net]:
            module.apply(initialize_parameters)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        obs = obs.float()
        policy_features = self.policy_features_extractor(obs)
        latent_pi = self.policy_net(policy_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        value_features = self.value_features_extractor(obs)
        values = self.value_net(value_features)

        return actions, values, log_prob

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        policy_features = self.policy_features_extractor(obs)
        latent_pi = self.policy_net(policy_features)
        return self._get_action_dist_from_latent(latent_pi)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """

        policy_features = self.policy_features_extractor(obs)
        value_features = self.value_features_extractor(obs)

        latent_pi = self.policy_net(policy_features)
        values = self.value_net(value_features)

        try:
            distribution = self._get_action_dist_from_latent(latent_pi)
        except Exception as e:
            import traceback, sys
            print("DBG::show nan tensors?")
            print(obs)
            print(policy_features)
            print(value_features)
            print(latent_pi)
            print(values)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            exit(1)
        log_prob = distribution.log_prob(actions)

        return values, log_prob, distribution.entropy()

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        value_features = self.value_features_extractor(obs)
        values = self.value_net(value_features)
        return values

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                feature_scale=1,
                policy_feature_extractor=self.policy_feature_extractor,
                value_feature_extractor=self.value_feature_extractor
            )
        )
        return data

