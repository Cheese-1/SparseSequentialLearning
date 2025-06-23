from src.Learners import AbstractLearner
from src.Environments import AbstractEnvironment

import numpy as np
import sklearn.gaussian_process.kernels as kernels
import cvxpy as cp

class SIBKBLearner(AbstractLearner):

    def __init__(self, T : int, params : dict):
        """
        Initialize the SI-BKB algorithm.

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
        • epsilon (float)
            A parameter to determine the alpha parameter, where \alpha = \frac{1+\varepsilon}{1-\varepsilon}.
        • cov_bound (float)
            An upper bound of the kernel
        • q (float)
            A factor regulating the inclusion probability
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
        self.B = params["B"]                    # Bound on the function complexity of g
        self.delta = params["delta"]            # Accuracy rate
        self.sigma = params["sigma"]            # Standard deviation of the oracle's noise
        self.eps = params["epsilon"]            # A parameter to determine the alpha parameter, where \alpha = \frac{1+\varepsilon}{1-\varepsilon}.
        self.cov_bound = params["cov_bound"]    # An upper bound of the kernel

        self.q = params["q"]                    # A factor regulating the inclusion probability


        # Create the kernel
        self._find_kernel(params["kernel"], params["kernel_params"])

        # Record Oracle Queries
        self.xs = []
        self.ys = []
        self.S_t = np.array([])

    def run(self, env: AbstractEnvironment, logger):
        """
        Runs the SI-BKB bandit algorithm for the given environment.

        :param env: the bandit environment
        :param logger: handles logging of data
        """

        # ------------------------------------------------------- #
        # Subspace Identification (SI) Phase                      #
        # ------------------------------------------------------- #

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

        # Convert matrices to arrays
        self.xs = np.array(self.xs @ self.A.T)
        self.ys = np.array(self.ys)

        # ------------------------------------------------------- #
        # Bayesian Optimization (BO) Phase with the BKB algorithm #
        # ------------------------------------------------------- #

        # Observe a single arm at random initially
        contexts = env.generate_context()
        next_action = np.random.randint(0, len(contexts))

        # Observe the reward
        reward = env.reveal_reward(contexts[next_action])

        # Record the query
        self.xs = np.vstack([self.xs, contexts[next_action] @ self.A.T])
        self.S_t = np.atleast_2d(np.array([contexts[next_action] @ self.A.T]))
        self.ys = np.append(self.ys, reward)

        env.record_regret(reward, [])
        logger.log(self.t, env.best_action, next_action, reward, env.regret[-1])

        self.t += 1

        for t in range(2, self.T - self.m_xi * (self.m_phi + 1) + 1):
            # Generate the next contexts
            contexts = env.generate_context()

            next_action = self._compute_ucb_index(contexts @ self.A.T, self.t)

            # Observe the reward
            reward = env.reveal_reward(contexts[next_action])

            # Record the query
            self.xs = np.vstack([self.xs, contexts[next_action] @ self.A.T])
            self.ys = np.append(self.ys, reward)


            self.t += 1

            env.record_regret(reward, [])
            logger.log(self.t, env.best_action, next_action, reward, env.regret[-1])



    def _compute_ucb_index(self, contexts, time):
        """
        Computes the BKB Index function

        :params contexts: a matrix of currently observed context vectors
        :params time: the current round

        :return: The arm that maximizes the UCB_t function.
        """

        # Update datasets
        m = len(self.S_t)

        # Compute auxiliary matrices
        K_S_t= self.kernel(self.S_t, self.S_t)
        assert len(K_S_t) == m and len(K_S_t[0]) == m

        # Symmetrize the matrix to eliminate jitter
        K = (K_S_t + K_S_t.T) / 2
        K += 1e-8 * np.eye(m)

        # Decompose the kernel matrix K into eigenpairs
        eigvals, eigvecs = np.linalg.eigh(K)

        # Clip negative eigenvalues from numerical inaccuracies
        eigvals_clipped = np.clip(eigvals, a_min=0, a_max=None)

        # Compute the pseudo-inverse of the square root of the kernel matrix
        eps = 1e-12
        inv_sqrt_vals = np.array([1 / np.sqrt(l + eps) if l > 0 else 0.0
                                  for l in eigvals_clipped])
        sqrt_pinv = eigvecs @ np.diag(inv_sqrt_vals) @ eigvecs.T
        assert len(sqrt_pinv) == m and len(sqrt_pinv[0]) == m

        # Compute the matrix of Nystrom embeddings.
        Z_t = self._compute_nystrom_features(self.xs, sqrt_pinv)
        assert len(Z_t) == time and len(Z_t[0]) == m

        # Compute the matrix of Nystrom embeddings for the contexts
        k_SX_all = self.kernel(self.S_t, contexts)
        Z_cand = (sqrt_pinv @ k_SX_all).T
        assert len(Z_cand) == len(contexts) and len(Z_t[0]) == m

        # Compute the invertible matrix V
        V = Z_t.T @ Z_t + self.sigma ** 2 * np.eye(m)
        V_inv = np.linalg.inv(V)
        assert len(V) == m and len(V[0]) == m

        # Compute the posterior mean
        mu_t = Z_cand.dot(V_inv @ Z_t.T @ self.ys)
        assert len(mu_t) == len(contexts)


        # Compute the posterior variance
        quad_cand = np.sum(Z_cand.T  * (Z_t.T @ Z_t @ V_inv  @ Z_cand.T) , axis=0)

        k_diag_cand = np.array([
            self.kernel(contexts[[i], :], contexts[[i], :])[0, 0]
            for i in range(contexts.shape[0])
        ])

        var_t = (1.0 / (self.sigma ** 2)) * (k_diag_cand - quad_cand)
        assert len(var_t) == len(contexts)

        # Approximate the maximal information gain
        quad = np.sum(Z_t.T * (Z_t.T @ Z_t @ V_inv @ Z_t.T ), axis=0)

        k_diag_xs = np.array([
            self.kernel(self.xs[[i], :], self.xs[[i], :])[0, 0]
            for i in range(self.xs.shape[0])
        ])
        observed_var = (1.0 / (self.sigma ** 2)) * (k_diag_xs - quad)

        # Compute the beta factor
        alpha = (1 + self.eps) / (1 - self.eps)
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
        """
        Computes the Nystrom embedding for the given context matrix

        :param X: the context matrix
        :param sqrt_pinv: the pseudo-inverse square root kernel matrix of the induced subset

        :return: the Nystrom embeddings for the context matrix.
        """
        X_arr = np.atleast_2d(np.array(X))

        assert X_arr.shape[1] == self.S_t.shape[1]

        k_SX = self.kernel(self.S_t, X_arr)

        return (sqrt_pinv @ k_SX).T


    def _reconstruct_by_dantzig_selector(self, y, dir_phi):
        """
        Computes the low-rank reconstructed matrix though the Dantzig selector

        :param y: a vector representing the approximated matrix, the curvature, and zero-mean noise.
        :param dir_phi: a list of matrices comprised of the direction vector

        :return: the reconstructed low-rank matrix X_\mathrm{DS}
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
        """
        Computes the total reward attained by the algorithm.

        :return: the total obtained reward \sum_{t=1}^T y_t
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