import numpy as np
import optuna
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean


class TuneObjectiveCallback(BaseCallback):
    def __init__(self, trial: optuna.Trial, verbose=0):
        super(TuneObjectiveCallback, self).__init__(verbose)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.mean_ep_length = np.inf

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            self.mean_ep_length = safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
        else:
            self.mean_ep_length = np.inf
        self.eval_idx += 1
        self.trial.report(self.mean_ep_length, self.eval_idx)
        if self.trial.should_prune():
            self.is_pruned = True
            return False
        return True


# class TrialEvalCallback(EvalCallback):
#     """
#     Callback used for evaluating and reporting a trial.
#     from stable-baselines3 rl zoo
#     it accepts any evaluator and reports best mean reward, and maximize it
#     """
#     def __init__(
#         self,
#         eval_env: VecEnv,
#         trial: optuna.Trial,
#         n_eval_episodes: int = 5,
#         eval_freq: int = 10000,
#         deterministic: bool = True,
#         verbose: int = 0,
#         evaluator = None
#     ):

#         super(TrialEvalCallback, self).__init__(
#             eval_env=eval_env,
#             n_eval_episodes=n_eval_episodes,
#             eval_freq=eval_freq,
#             deterministic=deterministic,
#             verbose=verbose,
#         )
#         self.trial = trial
#         self.eval_idx = 0
#         self.is_pruned = False
#         self.mean_ep_length = np.inf

#         if evaluator is None:
#             self.evaluator = evaluate_policy
#         else:
#             self.evaluator = evaluator

#     def _on_step(self) -> bool:
#         if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
#             self.evaluate_model()
#             self.eval_idx += 1
#             # report best mean reward, and maximize it
#             self.trial.report(self.best_mean_reward, self.eval_idx)
#             # Prune trial if need
#             if self.trial.should_prune():
#                 self.is_pruned = True
#                 return False
#         return True

#     def evaluate_model(self) -> bool:
#         if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
#             # Sync training and eval env if there is VecNormalize

#             # Reset success rate buffer
#             self._is_success_buffer = []

#             episode_rewards, episode_lengths = self.evaluator(
#                 self.model,
#                 self.eval_env,
#                 n_eval_episodes=self.n_eval_episodes,
#                 render=self.render,
#                 deterministic=False,
#                 return_episode_rewards=True,
#                 warn=self.warn,
#                 callback=self._log_success_callback,
#             )

#             if self.log_path is not None:
#                 self.evaluations_timesteps.append(self.num_timesteps)
#                 self.evaluations_results.append(episode_rewards)
#                 self.evaluations_length.append(episode_lengths)

#                 kwargs = {}
#                 # Save success log if present
#                 if len(self._is_success_buffer) > 0:
#                     self.evaluations_successes.append(self._is_success_buffer)
#                     kwargs = dict(successes=self.evaluations_successes)

#                 np.savez(
#                     self.log_path,
#                     timesteps=self.evaluations_timesteps,
#                     results=self.evaluations_results,
#                     ep_lengths=self.evaluations_length,
#                     **kwargs,
#                 )

#             mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
#             mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
#             self.last_mean_reward = mean_reward
#             self.mean_ep_length = mean_ep_length

#             if self.verbose > 0:
#                 print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
#                 print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
#             # Add to current Logger
#             self.logger.record("eval/mean_reward", float(mean_reward))
#             self.logger.record("eval/mean_ep_length", mean_ep_length)

#             if len(self._is_success_buffer) > 0:
#                 success_rate = np.mean(self._is_success_buffer)
#                 if self.verbose > 0:
#                     print(f"Success rate: {100 * success_rate:.2f}%")
#                 self.logger.record("eval/success_rate", success_rate)

#             if mean_reward > self.best_mean_reward:
#                 if self.verbose > 0:
#                     print("New best mean reward!")
#                 if self.best_model_save_path is not None:
#                     self.model.save(os.path.join(self.best_model_save_path, "best_model"))
#                 self.best_mean_reward = mean_reward
#                 # Trigger callback if needed
#                 if self.callback is not None:
#                     return self._on_event()
#         return True


# def sample_ppo_params(trial: optuna.Trial):
#     """
#     Sampler for PPO hyperparams.
#     :param trial:
#     :return:
#     """
#     # max_episode_len = trial.suggest_categorical("max_episode_len", [256, 512, 1024, 2048])
#     batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
#     n_steps = trial.suggest_categorical("n_steps", [1000, 2000, 4000, 8000])
#     # gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
#     learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-3)
#     ent_coef = trial.suggest_loguniform("ent_coef", 1e-6, 1e-1)
#     clip_range = trial.suggest_loguniform("clip_range", 0.1, 0.2)

#     n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20, 40, 50, 100])
#     # gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
#     max_grad_norm = trial.suggest_loguniform("max_grad_norm", 0.1, 10)
#     vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
#     net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big", "verybig"])

#     # rewards_cat = [-0.1, -0.05, -0.01, -0.05, -0.001]
#     # reward_step = trial.suggest_categorical("reward_step", rewards_cat)
#     # reward_invalid = trial.suggest_categorical("reward_invalid", rewards_cat)
#     # reward_state = trial.suggest_categorical("reward_state", rewards_cat)

#     if batch_size > n_steps:
#         batch_size = n_steps
#     # if max_episode_len > n_steps:
#     #     max_episode_len = n_steps

#     # Independent networks usually work best
#     # when not working with images
#     net_arch = {
#         "small": [dict(pi=[32, 32], vf=[32, 32])],
#         "medium": [dict(pi=[64, 64], vf=[64, 64])],
#         "big":  [dict(pi=[128, 128], vf=[128, 128])],
#         "verybig": [dict(pi=[256, 256], vf=[256, 256])]
#     }[net_arch]

#     return {
#         # "max_episode_len": max_episode_len,
#         "n_steps": n_steps,
#         "batch_size": batch_size,
#         # "gamma": gamma,
#         "learning_rate": learning_rate,
#         "ent_coef": ent_coef,
#         "clip_range": clip_range,
#         "n_epochs": n_epochs,
#         # "gae_lambda": gae_lambda,
#         "max_grad_norm": max_grad_norm,
#         "vf_coef": vf_coef,
#         "policy_kwargs": dict(net_arch=net_arch,),
#         # "reward_step": reward_step,
#         # "reward_invalid": reward_invalid,
#         # "reward_state": reward_state
#     }


# def sample_dqnper_params(trial: optuna.Trial):
#     batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
#     buffer_size = trial.suggest_categorical("buffer_size", [1000, 5000, 10000, 50000, 100000])
#     learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-3)
#     exploration_fraction = trial.suggest_categorical("exploration_fraction", [0.1, 0.2, 0.3, 0.4, 0.5])
#     exploration_final_eps = trial.suggest_loguniform("exploration_final_eps", 0.01, 0.2)
#     net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
#     target_update_interval = trial.suggest_categorical("target_update_interval", [1, 1000, 5000, 10000, 50000])
#     gradient_steps = trial.suggest_int("gradient_steps", 0, 10)

#     # Independent networks usually work best
#     # when not working with images
#     net_arch = {
#         "small": [32, 32],
#         "medium": [64, 64],
#         "big":  [128, 128],
#         "verybig": [256, 256],
#     }[net_arch]

#     return {
#         "batch_size": batch_size,
#         "buffer_size": buffer_size,
#         "learning_rate": learning_rate,
#         "policy_kwargs": dict(net_arch=net_arch),
#         "target_update_interval": target_update_interval,
#         "exploration_fraction": exploration_fraction,
#         "exploration_final_eps": exploration_final_eps,
#         "gradient_steps": gradient_steps
#     }


# def sample_hdqn_params(trial: optuna.Trial):
#     buffer_size = trial.suggest_categorical("buffer_size", [1000, 5000, 10000, 50000, 100000])
#     buffer_size_low = trial.suggest_categorical("buffer_size_low", [1000, 5000, 10000, 50000, 100000])
#     learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
#     learning_rate_low = trial.suggest_loguniform("learning_rate_low", 1e-5, 1e-3)
#     batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
#     batch_size_low = trial.suggest_categorical("batch_size_low", [8, 16, 32, 64, 128])

#     exploration_fraction = trial.suggest_categorical("exploration_fraction", [0.5, 0.6, 0.7, 0.8, 0.9])
#     exploration_mid_fraction = trial.suggest_categorical("exploration_mid_fraction", [0.1, 0.2, 0.3, 0.4, 0.5])
#     exploration_mid_eps = trial.suggest_loguniform("exploration_mid_eps", 0.5, 1.0)

#     # exploration_fraction_low = trial.suggest_categorical("exploration_fraction_low", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
#     net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])

#     # Independent networks usually work best
#     # when not working with images
#     net_arch = {
#         "small": [32, 32],
#         "medium": [64, 64],
#         "big":  [128, 128],
#         "verybig": [256, 256],
#     }[net_arch]

#     return {
#         "buffer_size": buffer_size,
#         "buffer_size_low": buffer_size_low,
#         "learning_rate": learning_rate,
#         "learning_rate_low": learning_rate_low,
#         "batch_size": batch_size,
#         "batch_size_low": batch_size_low,
#         "exploration_fraction": exploration_fraction,
#         "exploration_mid_fraction": exploration_mid_fraction,
#         "exploration_mid_eps": exploration_mid_eps,
#         # "exploration_fraction_low": exploration_fraction_low,
#         "policy_kwargs": dict(net_arch=net_arch),
#         "policy_kwargs_low": dict(net_arch=net_arch),
#     }


# def sample_hrl_plan_dqn_params(trial: optuna.Trial):
#     learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
#     buffer_size = trial.suggest_categorical("buffer_size", [1000, 5000, 10000, 50000, 100000])
#     batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])

#     exploration_fraction = trial.suggest_categorical("exploration_fraction", [0.1, 0.2, 0.3, 0.4, 0.5])
#     exploration_final_eps = trial.suggest_loguniform("exploration_final_eps", 0.01, 0.2)
#     target_update_interval = trial.suggest_categorical("target_update_interval", [1, 1000, 2000, 5000, 10000, 50000])
#     gradient_steps = trial.suggest_int("gradient_steps", 0, 10)
#     net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])

#     # Independent networks usually work best
#     # when not working with images
#     net_arch = {
#         "small": [32, 32],
#         "medium": [64, 64],
#         "big":  [128, 128],
#         "verybig": [256, 256],
#     }[net_arch]

#     return {
#         "buffer_size": buffer_size,
#         "learning_rate": learning_rate,
#         "batch_size": batch_size,
#         "exploration_fraction": exploration_fraction,
#         "exploration_final_eps": exploration_final_eps,
#         "target_update_interval": target_update_interval,
#         "gradient_steps":gradient_steps,
#         "policy_kwargs": dict(net_arch=net_arch)
#     }


# class TuneObjectiveCallback(BaseCallback):
#     """
#     this does not actually evaluate policy but reads stored metric
#     """
#     def __init__(self, trial: optuna.Trial, verbose=0):
#         super(TuneObjectiveCallback, self).__init__(verbose)
#         self.trial = trial
#         self.eval_idx = 0
#         self.is_pruned = False
#         self.mean_ep_length = np.inf

#     def _on_step(self) -> bool:
#         return True

#     def _on_rollout_end(self):
#         if len(self.model.average_action_len_per_ep) >= 100:
#             self.mean_ep_length = safe_mean(self.model.average_action_len_per_ep)
#         else:
#             self.mean_ep_length = np.inf
#         self.eval_idx += 1
#         self.trial.report(self.mean_ep_length, self.eval_idx)
#         if self.trial.should_prune():
#             self.is_pruned = True
#             return False
#         return True

