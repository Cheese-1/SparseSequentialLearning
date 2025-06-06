from src.Learners import AbstractLearner
from src.Environments import AbstractEnvironment

from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import sklearn.gaussian_process.kernels as kernels

class GPUCBLearner(AbstractLearner):

    def __init__(self, T : int, params : dict):
        super().__init__(T, params)

        # General Parameters
        self.d = None

        # GP-UCB Index Parameters
        self.B = params["B"]
        self.delta = params["delta"]
        self.sigma = params["sigma"]
        self.gamma = params["gamma"]

        # Create the kernel
        self._find_kernel(params["kernel"], params["kernel_params"])


        # Create the Gaussian Regressor
        self.gpr = GaussianProcessRegressor(
            kernel = self.kernel,
            alpha = self.sigma ** 2,
            optimizer = None,
            normalize_y=False
        )

        # Record Oracle Queries
        self.xs = []
        self.ys = []

        self.gammas = []

    def run(self, env: AbstractEnvironment, logger):

        # Find the ambient dimension
        self.d = env.get_ambient_dim()

        for t in range(1, self.T + 1):
            # Generate the next contexts
            contexts = env.generate_context()

            # Retrain the GPR
            if t > 1:
                self.gpr.fit(self.xs, self.ys)

                # Determine the next action
                next_action = self._compute_ucb_index(contexts, t)
            else:
                next_action = np.random.randint(0, len(contexts))

            # Observe the reward
            reward = env.reveal_reward(contexts[next_action])

            # Record the query
            self.xs.append(contexts[next_action])
            self.ys.append(reward)

            env.record_regret(reward, [])
            logger.log(t, env.best_action, next_action, reward, env.regret[-1])



    def _compute_ucb_index(self, contexts, time):

        # Compute mu_t and sigma_t
        means_t, stds_t = self.gpr.predict(contexts, return_std = True)

        # Compute gamma_t
        gamma_t = np.log(self.gamma) + (self.d + 1) * np.log(np.log(self.T))


        # Find the action with the best index
        sqrt_beta_t = np.sqrt(2 * self.B + 300 * np.exp(gamma_t) * np.log(time / self.delta) ** 3)
        best_index = np.argmax(means_t + sqrt_beta_t * stds_t)

        self.gammas.append(contexts[np.argmax(stds_t)])

        return best_index


    def _find_kernel(self, name, kernel_params):
        if name == "RBF":
            self.kernel = kernels.RBF(
                length_scale = kernel_params["length"]
            )
        elif name == "Matern":
            self.kernel = kernels.Matern(
                length_scale = kernel_params["length"],
                nu = kernel_params["smoothness"]
            )
        elif name == "Rational Quadratic":
            self.kernel = kernels.RationalQuadratic(
                length_scale = kernel_params["length"],
                alpha = kernel_params["alpha"]
            )
        else:
            raise Exception("Unrecognized kernel name")

    def total_reward(self):
        return np.sum(self.history)

    def cum_reward(self):
        return np.cumsum(self.history)

    def select_action(self, context):
        raise Exception("Not implemented since it is unnecessary.")