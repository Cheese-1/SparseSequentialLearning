from ..AbstractEnvironment import AbstractEnvironment
import numpy as np
import math


class GaussianSparseEnvironment(AbstractEnvironment):


    def __init__(self, params):

        super().__init__(params)
        self.k = params["k"]


        # Action Generation - Mean & Variance
        self.max_val = params["max_val"]
        self.min_val = params["min_val"]

        self.sigma = params["sigma"]

        self.means = np.random.uniform(self.max_val, self.min_val, (self.k, self.d))

        if not isinstance(self.sigma, list):
            self.sigma = [self.sigma] * self.d

        # Reward specification
        safe_globals = {"__builtins__": None}
        safe_globals.update(math.__dict__)

        self.r = eval("lambda x : " + params["r"], safe_globals)
        self.r_sigma = params["r_sigma"]


        # Regret Parameters
        self.trials = params["trials"]

        self.best_action = None
        self.expected_reward_constant_policy = self._maximize_constat_policy()





    def record_regret(self, reward, feature_set):

        # Compute the instantaneous regret
        instantaneous_regret = self.expected_reward_constant_policy - reward

        # record the instantaneous regent and recompute the cumulative regret
        self.regret.append(instantaneous_regret)
        self.cum_regret += instantaneous_regret

    def generate_context(self):

        return np.random.normal(loc=self.means, scale=self.sigma)

    def reveal_reward(self, action):

        return self.r(action) + np.random.normal(loc = 0, scale = self.r_sigma)

    def observe_actions(self):
        return self.k

    def _maximize_constat_policy(self):

        rewards = np.zeros(self.k)

        for _ in range(self.trials):

            contexts = self.generate_context()
            rewards += np.array([self.reveal_reward(c) for c in contexts])

        self.best_action = np.argmax(rewards)
        return max(rewards) / self.trials