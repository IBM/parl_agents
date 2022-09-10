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
