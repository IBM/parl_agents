import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


def make_env(env_id, seed, kwargs, wrappers, wrapper_kwargs):
    env = gym.make(env_id, **kwargs)
    np_seed = env.seed(seed)        # list of seeds, the first one is the main seed
    if isinstance(np_seed, list):
        np_seed = np_seed[0]

    set_random_seed(np_seed)
    if wrappers:
        for wrap in wrappers:
            env = wrap(env, **wrapper_kwargs)
    # env = Monitor(env)
    return env