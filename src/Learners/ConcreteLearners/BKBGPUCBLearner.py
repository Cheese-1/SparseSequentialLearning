from src.Learners import AbstractLearner
from src.Environments import AbstractEnvironment

from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import sklearn.gaussian_process.kernels as kernels
from scipy.linalg import sqrtm, pinv, inv

class BKBGPUCBLearner(AbstractLearner):

    def __init__(self, T: int, params: dict):
        super().__init__(T, params)

        # General Parameters
        self.d = None

        # GP-UCB Index Parameters
        self.B = params["B"]
        self.delta = params["delta"]
        self.eps = params["epsilon"]
        self.cov_bound = params["cov_bound"]
        self.sigma = params["sigma"]

        self.q = params["q"]

        # Create the kernel
        self._find_kernel(params["kernel"], params["kernel_params"])

        # Record Oracle Queries
        self.xs = np.array([])
        self.ys = np.array([])

        self.S_t = np.array([])

    def run(self, env: AbstractEnvironment, logger):

        # Find the ambient dimension
        self.d = env.get_ambient_dim()

        # Observe a single arm at random initially
        contexts = env.generate_context()
        next_action = np.random.randint(0, len(contexts))

        # Observe the reward
        reward = env.reveal_reward(contexts[next_action])

        # Record the query
        self.xs = np.array([contexts[next_action]])
        self.S_t = np.atleast_2d(np.array([contexts[next_action]]))
        self.ys = np.array([reward])

        env.record_regret(reward, [])
        logger.log(1, env.best_action, next_action, reward, env.regret[-1])


        for t in range(2, self.T + 1):
            # Generate the next contexts
            contexts = env.generate_context()

            next_action = self._compute_ucb_index(contexts, t)

            # Observe the reward
            reward = env.reveal_reward(contexts[next_action])

            # Record the query
            self.xs = np.vstack([self.xs, contexts[next_action].T])
            self.ys = np.append(self.ys, reward)


            env.record_regret(reward, [])
            logger.log(t, env.best_action, next_action, reward, env.regret[-1])

    def _compute_ucb_index(self, contexts, time):

        # Update datasets=
        m = len(self.S_t)

        # Compute auxiliary matrices
        K_S_t= self.kernel(self.S_t, self.S_t)
        assert len(K_S_t) == m and len(K_S_t[0]) == m

        sqrt_pinv = np.array(pinv(sqrtm(K_S_t)))
        assert len(sqrt_pinv) == m and len(sqrt_pinv[0]) == m


        Z_t = self._compute_nystrom_features(self.xs, sqrt_pinv)
        assert len(Z_t) == (time - 1) and len(Z_t[0]) == m

        k_SX_all = self.kernel(self.S_t, contexts)  # shape: (m, K)
        Z_cand = (sqrt_pinv @ k_SX_all).T
        assert len(Z_cand) == len(contexts) and len(Z_t[0]) == m


        V = Z_t.T @ Z_t + self.sigma ** 2 * np.eye(m)
        V_inv = np.linalg.inv(V)
        assert len(V) == m and len(V[0]) == m


        # Compute the posterior quantities
        mu_t = Z_cand.dot(V_inv @ Z_t.T @ self.ys)
        assert len(mu_t) == len(contexts)

        ZTZ = Z_t.T @ Z_t  # (m × m)
        M = ZTZ @ V_inv  # (m × m)
        ZcT = Z_cand.T  # (m × K)
        R = M @ ZcT  # (m × K)
        quad_cand = np.sum(ZcT * R, axis=0)  # (K,)

        k_diag_cand = np.array([
            self.kernel(contexts[[i], :], contexts[[i], :])[0, 0]
            for i in range(contexts.shape[0])
        ])  # (K,)

        var_t = (1.0 / (self.sigma ** 2)) * (k_diag_cand - quad_cand)
        assert len(var_t) == len(contexts)

        alpha = (1 + self.eps) / (1 - self.eps)

        ZTZ = Z_t.T @ Z_t  # (m × m)

        M = ZTZ @ V_inv  # (m × m)
        Zp = Z_t.T  # (m × n)
        R_full = M @ Zp

        quad = np.sum(Zp * R_full, axis=0)  # shape: (n,)

        k_diag_xs = np.array([
            self.kernel(self.xs[[i], :], self.xs[[i], :])[0, 0]
            for i in range(self.xs.shape[0])
        ])  # → (n,)


        observed_var = (1.0 / (self.sigma ** 2)) * (k_diag_xs - quad)


        beta_t = 2 * self.sigma * np.sqrt(alpha * np.log(self.cov_bound * time) * sum(observed_var) + np.log(1 / self.delta)) + (1 + 1 / np.sqrt(1 - self.eps)) * self.sigma * self.B


        # Select the best action
        UCB_indices = np.array(mu_t)  + beta_t * np.sqrt(np.array(var_t))
        best_action = np.argmax(UCB_indices)


        # Select the next inducing subset

        # Compute the mask using the Bernoulli Probabilities
        ps = [np.random.binomial(1,min(self.q * v, 1)) for v in observed_var]
        mask  = np.array(ps)
        # Select the next subset
        self.S_t =  np.vstack([self.xs[mask == 1], contexts[best_action].T])

        return best_action


    def _compute_nystrom_features(self, X, sqrt_pinv):

        X_arr = np.atleast_2d(np.array(X))

        assert X_arr.shape[1] == self.S_t.shape[1]

        k_SX = self.kernel(self.S_t, X_arr)

        return (sqrt_pinv @ k_SX).T

    def _find_kernel(self, name, kernel_params):
        if name == "RBF":
            self.kernel = kernels.RBF(
                length_scale=kernel_params["length"]
            )
        elif name == "Matern":
            self.kernel = kernels.Matern(
                length_scale=kernel_params["length"],
                nu=kernel_params["smoothness"]
            )
        elif name == "Rational Quadratic":
            self.kernel = kernels.RationalQuadratic(
                length_scale=kernel_params["length"],
                alpha=kernel_params["alpha"]
            )
        else:
            raise Exception("Unrecognized kernel name")

    def total_reward(self):
        return np.sum(self.history)

    def cum_reward(self):
        return np.cumsum(self.history)

    def select_action(self, context):
        raise Exception("Not implemented since it is unnecessary.")