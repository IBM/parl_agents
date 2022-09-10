import gym
from gym import error, spaces, utils


class AugmentDiscreteToImg(gym.core.ObservationWrapper):
    """
    take image as obs
    augment blank discrete feature with value 0
    """
    def __init__(self, env, discrete_feature_name="label", discrete_feature_size=2):
        super().__init__(env)
        self.discrete_feature_name = discrete_feature_name
        self.discrete_feature_size = discrete_feature_size

        self.observation_space = spaces.Dict(
            {'image': env.observation_space,
             discrete_feature_name: spaces.Discrete(discrete_feature_size)
             })

    def observation(self, obs):
        return {
            'image': obs,
            self.discrete_feature_name: 0
        }