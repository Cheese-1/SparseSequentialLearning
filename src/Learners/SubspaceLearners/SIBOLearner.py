from src.Learners import AbstractLearner
from src.Environments import AbstractEnvironment

from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import sklearn.gaussian_process.kernels as kernels
import cvxpy as cp


class SIBOLearner(AbstractLearner):

    def __init__(self, T: int, params: dict):
        """
        Initialize the Subspace Identification Bayesian Optimization (SI-BO) algorithm.

        :param T: the horizon
        :param params: a dictionary of algorithm configuration parameters. The Required keys are:

        • m_xi (int)
            Number of context samples to draw in the subspace-identification (SI) phase.
        • m_phi (int)
            Number of direction samples to draw in the SI phase.
        • lamb (float)
            Constraint parameter for the Dantzig selector used in subspace estimation.
        • eps (float)
            Step-size for updating the subspace transformation.
        • k (int)
            Target dimension of the learned subspace.
        • B (float)
            Upper bound on the complexity of the reward function.
        • delta (float)
            Desired accuracy rate for the GP-UCB acquisition rule.
        • sigma (float)
            Standard deviation of the Gaussian observation noise from the oracle.
        • kernel
            the kernel name
        • kernel_params (dict)
            Keyword arguments for instantiating the chosen GP kernel.
        """

        super().__init__(T, params)

        # General SI-BO Parameters
        self.m_xi = params["m_xi"]          # Number of context samples in the SI phase
        self.m_phi = params["m_phi"]        # Number of direction samples in the SI phase
        self.lamb = params["lamb"]          # The constraint parameter for the Dantzig Selector
        self.eps = params["eps"]            # Step size
        self.k = params["k"]                # Dimension of the subspace
        self.d = None                       # Dimension of the ambient space

        # Subspace Transformation matrix
        self.A = None                       # The learned subspace transformation

        # GP-UCB Index Parameters
        self.B = params["B"]                # Bound on the function complexity of g
        self.delta = params["delta"]        # Accuracy rate
        self.sigma = params["sigma"]        # Standard deviation of the oracle's noise

        # Create the kernel
        self._find_kernel(params["kernel"], params["kernel_params"])

        # Create the Gaussian Regressor
        self.gpr = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.sigma ** 2,
            optimizer=None,
            normalize_y=False
        )

        # Record Oracle Queries
        self.xs = []
        self.ys = []

    def run(self, env: AbstractEnvironment, logger):
        """
        Runs the SI-BO bandit algorithm for the given environment.

        :param env: the bandit environment
        :param logger: handles logging of data
        """

        # Find the ambient dimension
        self.d = env.get_ambient_dim()

        # Reset rounds
        self.t = 0

        # ------------------------------------------------------- #
        # Subspace Identification (SI) Phase                      #
        # ------------------------------------------------------- #

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
            dir_phi.append(np.random.choice([- 1 / np.sqrt(self.m_xi), 1 / np.sqrt(self.m_xi)],
                                            size=(env.get_ambient_dim(), self.m_xi)))

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
        (upper, _, _) = np.linalg.svd(grad_x_dantzig)
        self.A = upper[:self.k]

        # Update the query records
        self.xs = [self.A @ x for x in self.xs]

        # ------------------------------------------------------- #
        # Bayesian Optimization (BO) Phase                        #
        # ------------------------------------------------------- #
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
        """
        Computes the GP-UCB Index function

        :params contexts: a matrix of currently observed context vectors
        :params time: the current round

        :return: The arm that maximizes the UCB_t function.
        """
        # Compute mu_t and sigma_t
        means_t, stds_t = self.gpr.predict(contexts @ self.A.T, return_std=True)

        # Compute gamma_t
        if time > 1:
            n_queries = len(self.ys)
            K = self.gpr.kernel_(self.xs, self.xs)
            _, logdet = np.linalg.slogdet(np.eye(n_queries) + K / self.sigma ** 2)
            gamma_t = np.e / (np.e - 1) * 0.5 * logdet
        else:
            gamma_t = 0

        # Find the action with the best index
        sqrt_beta_t = np.sqrt(2 * self.B + 300 * gamma_t * np.log(time / self.delta) ** 3)
        best_index = np.argmax(means_t + sqrt_beta_t * stds_t)

        return best_index

    def _reconstruct_by_dantzig_selector(self, y, dir_phi):
        """
        Computes the low-rank reconstructed matrix though the Dantzig selector

        :param y: a vector representing the approximated matrix, the curvature, and zero-mean noise.
        :param dir_phi: a list of matrices comprised of the direction vector

        :return: the reconstructed low-rank matrix X_{DS}
        """
        # Define the Matrix
        M = cp.Variable((self.d, self.m_xi))

        # Construct the vector of linear transformations
        AM = cp.hstack([cp.trace(Phi.T @ M) for Phi in dir_phi])


        # Compute the residual relative to the approximated matrix
        residual = y - AM

        # Transform the residual back using the Hermitian adjoint
        A_adj_residual = sum(residual[i] * dir_phi[i] for i in range(self.m_phi))

        # Define the Minimization Objective
        objective = cp.Minimize(cp.normNuc(M))

        # Define the Constraint
        constraints = [cp.norm(A_adj_residual, 'fro') <= self.lamb]

        # Solve the convex optimization problem
        prob = cp.Problem(objective, constraints)
        prob.solve()

        return M.value

    def _find_kernel(self, name, kernel_params):
        """
        Construct the kernel matrix for the specified kernel name with
        the specified kernel parameters

        :param name: the kernel name
        :param kernel_params: a dictionary containing the parameters of the kernel.
        """
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
        """
        Computes the total reward attained by the algorithm.

        :return: the total obtained reward sum_{t=1}^T y_t
        """
        return np.sum(self.ys)

    def cum_reward(self):
        """
        Computes a vector of cumulative reward at each round.

        :return: a vector of cumulative rewards at each round.
        """
        return np.cumsum(self.ys)

    def select_action(self, context):
        """
        Deprecated.
        """
        raise Exception("Not implemented since it is unnecessary.")
