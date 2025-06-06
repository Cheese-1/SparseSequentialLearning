from src.Learners import AbstractLearner
from src.Environments import AbstractEnvironment

from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import sklearn.gaussian_process.kernels as kernels
import cvxpy as cp

class SIBOLearner(AbstractLearner):

    def __init__(self, T : int, params : dict):
        super().__init__(T, params)

        # General SI-BO Parameters
        self.m_xi = params["m_xi"]
        self.m_phi = params["m_phi"]
        self.lamb = params["lamb"]
        self.eps = params["eps"]
        self.k = params["k"]
        self.d = None

        # Subspace Transformation matrix
        self.A = None

        # GP-UCB Index Parameters
        self.B = params["B"]
        self.delta = params["delta"]
        self.sigma = params["sigma"]


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

        # Reset rounds
        self.t = 0

        # Sample center means
        centers = []
        for _ in range(self.m_xi // env.observe_actions()):
            contexts = env.generate_context()
            np.random.shuffle(contexts)
            centers.extend(contexts)

        contexts = env.generate_context()
        np.random.shuffle(contexts)
        centers.extend(contexts[:self.m_xi % env.observe_actions()])

        centers = np.array(centers)

        # Compute perturbed directions
        dir_phi = []
        for _ in range(self.m_phi):
            dir_phi.append(np.random.choice([- 1 / np.sqrt(self.m_xi), 1 / np.sqrt(self.m_xi)], size=(env.get_ambient_dim(), self.m_xi)))


        # Compute y
        y = np.zeros(self.m_phi)

        for cen_ind in range(self.m_xi):

            # Computer center Reward
            center_reward = env.reveal_reward(centers[cen_ind])

            # Record Queries

            self.xs.append(centers[cen_ind])
            self.ys.append(center_reward)

            self.t += 1

            env.record_regret(center_reward, [])
            logger.log(self.t, [], [], center_reward, env.regret[-1])

            for phi_ind in range(self.m_phi):
                # Find Perturbed rewards
                shifted_center = centers[cen_ind] + self.eps * dir_phi[phi_ind].T[cen_ind]
                shifted_reward = env.reveal_reward(shifted_center)

                # Modify y
                y[phi_ind] += shifted_reward - center_reward

                # Record queries
                self.xs.append(shifted_center)
                self.ys.append(shifted_reward)

                self.t += 1

                env.record_regret(shifted_reward, [])
                logger.log(self.t, [], [], shifted_reward, env.regret[-1])

        y = y / self.eps


        # Reconstruct the gradient matrix using the Dantzig Selector
        grad_x_dantzig = self._reconstruct_by_dantzig_selector(y, dir_phi)

        # Extract the k principal vectors
        (upper, _ ,_) = np.linalg.svd(grad_x_dantzig)
        self.A = upper[:self.k]

        # Update the query records
        self.xs = [self.A @ x for x in self.xs]

        for t in range(1, self.T - self.m_xi * (self.m_phi + 1) + 1):
            # Generate the next contexts
            contexts = env.generate_context()

            # Retrain the GPR
            self.gpr.fit(self.xs, self.ys)

            # Determine the next action
            next_action = self._compute_ucb_index(contexts, t)

            # Observe the reward
            reward = env.reveal_reward(contexts[next_action])

            # Record the query
            self.xs.append(self.A @ contexts[next_action])
            self.ys.append(reward)

            self.t += 1

            env.record_regret(reward, [])
            logger.log(self.t, env.best_action, next_action, reward, env.regret[-1])



    def _compute_ucb_index(self, contexts, time):

        # Compute mu_t and sigma_t
        means_t, stds_t = self.gpr.predict(contexts @ self.A.T, return_std = True)

        # Compute gamma_t
        if time > 1:
            n_queries = len(self.ys)
            K = self.gpr.kernel_(self.gammas, self.gammas)
            _, logdet = np.linalg.slogdet(np.eye(n_queries) + K / self.sigma ** 2)
            gamma_t = np.e / (np.e - 1) * 0.5 * logdet
        else:
            gamma_t = 0


        # Find the action with the best index
        sqrt_beta_t = np.sqrt(2 * self.B + 300 * gamma_t * np.log(time / self.delta) ** 3)
        best_index = np.argmax(means_t + sqrt_beta_t * stds_t)

        self.gammas.append(contexts[np.argmax(stds_t)])

        return best_index

    def _reconstruct_by_dantzig_selector(self, y, dir_phi):

        M = cp.Variable((self.d, self.m_xi))

        # A(M): vector of trace(M @ Phi_i.T)
        AM = cp.hstack([cp.trace(Phi.T @ M) for Phi in dir_phi])

        residual = y - AM

        # A^*(residual) = sum_i residual[i] * Phi_i
        A_adj_residual = sum(residual[i] * dir_phi[i] for i in range(self.m_phi))

        # Optimization problem
        objective = cp.Minimize(cp.normNuc(M))
        constraints = [cp.norm(A_adj_residual, 'fro') <= self.lamb]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        return M.value

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