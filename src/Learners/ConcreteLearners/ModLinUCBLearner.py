from src.Learners import AbstractLearner
from src.Environments import AbstractEnvironment

import numpy as np

class ModLinUCBLearner(AbstractLearner):

    def __init__(self, T : int, params : dict):
        super().__init__(T, params)

        self.optimal_action = None
        self.action_set = None
        self.d = None
        self.k = None
        self.regressor = None
        self.delta = params["delta"]
        self.regularization = params["regularization"]

        self.b = None
        self.V = None
        self.theta = None

    def run(self, env: AbstractEnvironment, logger):

        # Find the ambient dimension
        self.d = env.get_ambient_dim()
        self.k = env.k

        self.b = np.zeros(self.d).reshape(-1, 1)
        self.V = self.regularization * np.eye(self.d)
        self.theta = np.zeros(self.d)

        for t in range(1, self.T + 1):
            # Generate the next contexts
            contexts = env.generate_context()


            # Determine the next action
            next_action = self._compute_ucb_index(contexts, t)

            # Observe the reward
            reward = env.reveal_reward(contexts[next_action])

            # Record the query
            self.V += contexts[next_action] @ contexts[next_action].T
            self.b += reward * contexts[next_action]
            self.theta = np.linalg.inv(self.V) @ self.b

            env.record_regret(reward, [])
            logger.log(t, env.best_action, next_action, reward, env.regret[-1])

            self.history.append(reward)

    def _compute_ucb_index(self, contexts, time):

        beta = np.sqrt(self.regularization)
        beta += np.sqrt(2 * np.log(1/self.delta) + self.d * (np.log(1 + (self.t-1)/(self.regularization * self.d))))

        V_inv = np.linalg.inv(self.V)

        # Find the action with the best index
        best_index = np.argmax([ct.T @ self.theta + beta * np.sqrt(ct.T @ V_inv @ ct) for ct in contexts])

        return best_index


    def total_reward(self):
        return np.sum(self.history)

    def cum_reward(self):
        return np.cumsum(self.history)

    def select_action(self, context):
        raise Exception("Not implemented since it is unnecessary.")