from numpy import matrix, array, dot, sqrt, ones, sign, vstack, cov, mean
from numpy.linalg import inv, multi_dot
from copy import deepcopy

from Decomposition import Cholesky


def least_squares(x: array, y: array):
    """
    Solving the coefficient of a linear system by given input variables.
    :param x: A 2-D array with column as feature dimension and row as sample dimension.
    :param y: The target 1-D array.
    :return: The estimated coefficient array for the linear system.
    """
    return multi_dot([inv(dot(x.T, x)), x.T, y])


def least_squares_nla(A: array, y: array):
    """
    Solving the coefficient of a linear system by given input variables.
    :param A: A 2-D array with column as feature dimension and row as sample dimension, without 1 vector.
    :param y: The target 1-D array.
    :return: The estimated coefficient array for the linear system.
    """
    A = A.reshape(max(A.shape), -1)
    cov_A = cov(A.T)
    sigma = []
    for series in A.T:
        covariance = cov(y.reshape(1, -1), series)
        sigma.append(covariance[0][1])
    sigma = array(sigma).reshape(-1, 1)
    mu_y = mean(y)
    mu_A = mean(A, axis=0)
    b = dot(inv(cov_A), sigma)
    a = mu_y - multi_dot([mu_A, inv(cov_A), sigma])
    coefficient = vstack([a, b])
    return a, b, coefficient


def implied_volatility():
    pass


class PortfolioOptimize(Cholesky):
    def __init__(self, mu, cov):
        super().__init__()
        self.mu = mu
        self.cov = cov

    def min_variance_weights(self, emu: float, rf: float, mu=None, cov=None, solver="cholesky"):
        """
        Compute the weightings of the minimum variance portfolio.
        :param emu: The expected return of a portfolio.
        :param rf: The risk-free rate.
        :param mu: The return array of the risky assets, in 1-D array format.
        :param cov: The covariance matrix, in 2-D array format.
        :param solver: Which solver to use, cholesky or linalg solver.
        :return: The optimal weightings array.
        """
        if not mu:
            mu = deepcopy(self.mu)
        if not cov:
            cov = deepcopy(self.cov)
        mu = mu.reshape(-1, 1)
        if solver == "cholesky":
            x = array(self.compute_single_linear_system(b=mu, A=cov))
            w = (emu - rf) / dot(mu.T, x) * x
        else:
            w = (emu - rf) / multi_dot([mu.T, inv(cov), mu]) * dot(inv(cov), mu)
        return w.reshape(1, -1)[0]

    def min_variance_std(self, emu: float, rf: float, mu=None, cov=None, solver="cholesky"):
        """
        Compute the standard deviation of the minimum variance portfolio.
        :param emu: The expected return of a portfolio.
        :param rf: The risk-free rate.
        :param mu: The return array of the risky assets, in 1-D array format.
        :param cov: The covariance matrix, in 2-D array format.
        :param solver: Which solver to use, cholesky or linalg solver.
        :return: The standard deviation of the minimum variance portfolio.
        """
        if not mu:
            mu = deepcopy(self.mu)
        if not cov:
            cov = deepcopy(self.cov)
        mu = mu.reshape(-1, 1)
        if solver == "cholesky":
            x = array(self.compute_single_linear_system(b=mu, A=cov))
            sigma = (emu - rf) / sqrt(dot(mu.T, x))
        else:
            sigma = (emu - rf) / sqrt(multi_dot([mu.T, inv(cov), mu]))
        return sigma.reshape(1, -1)[0][0]

    def min_var_tangency(self, emu: float, rf: float, mu=None, cov=None, solver="cholesky"):
        """
        Using a tangency portfolio to construct the minimum variance portfolio.
        :param emu: The expected return of a portfolio.
        :param rf: The risk-free rate.
        :param mu: The return array of the risky assets, in 1-D array format.
        :param cov: The covariance matrix, in 2-D array format.
        :param solver: Which solver to use, cholesky or linalg solver.
        :return: The weightings for a tangency portfolio, the cash weightings, the asset weightings.
        """
        if not mu:
            mu = deepcopy(self.mu)
        if not cov:
            cov = deepcopy(self.cov)
        mu = mu.reshape(-1, 1)
        one = ones(mu.shape[0]).reshape(1, -1)

        if solver == "cholesky":
            x = array(self.compute_single_linear_system(b=mu, A=cov))
            w_T = x / dot(one, x)
        else:
            w_T = dot(inv(cov), mu) / multi_dot([one, inv(cov), mu])

        w_cash = 1 - (emu - rf) / dot(mu.T, w_T)
        w = (1 - w_cash) * w_T
        return w_T.reshape(1, -1)[0], w_cash[0][0], w.reshape(1, -1)[0]

    def max_return_weights(self, e_sigma: float, mu=None, cov=None, solver="cholesky"):
        """
        Compute the weightings of the maximum return portfolio.
        :param e_sigma: The expected volatility of a portfolio.
        :param mu: The return array of the risky assets, in 1-D array format.
        :param cov: The covariance matrix, in 2-D array format.
        :param solver: Which solver to use, cholesky or linalg solver.
        :return: The optimal weightings.
        """
        if not mu:
            mu = deepcopy(self.mu)
        if not cov:
            cov = deepcopy(self.cov)
        mu = mu.reshape(-1, 1)
        if solver == "cholesky":
            x = array(self.compute_single_linear_system(b=mu, A=cov))
            w = (e_sigma / sqrt(dot(mu.T, x))) * x
        else:
            w = (e_sigma / sqrt(multi_dot([mu.T, inv(cov), mu]))) * dot(inv(cov), mu)
        w = w.reshape(1, -1)[0]
        return w, 1 - dot(ones(len(w)), w)

    def max_return_mu(self, e_sigma: float, rf: float, mu=None, cov=None, solver="cholesky"):
        """
        Compute the return of the maximum return portfolio.
        :param e_sigma: The expected volatility of a portfolio.
        :param rf: The risk-free rate.
        :param mu: The return array of the risky assets, in 1-D array format.
        :param cov: The covariance matrix, in 2-D array format.
        :param solver: Which solver to use, cholesky or linalg solver.
        :return: The return of the maximum return portfolio.
        """
        if not mu:
            mu = deepcopy(self.mu)
        if not cov:
            cov = deepcopy(self.cov)
        mu = mu.reshape(-1, 1)
        if solver == "cholesky":
            x = array(self.compute_single_linear_system(b=mu, A=cov))
            p_mu = rf + e_sigma * sqrt(dot(mu.T, x))
        else:
            p_mu = rf + e_sigma * sqrt(multi_dot([mu.T, inv(cov), mu]))
        return p_mu.reshape(1, -1)[0][0]

    def max_return_tangency(self, e_sigma: float, mu=None, cov=None, solver="cholesky"):
        """
        Using a tangency portfolio to construct maximum return portfolio.
        :param e_sigma: The expected volatility of a portfolio.
        :param mu: The return array of the risky assets, in 1-D array format.
        :param cov: The covariance matrix, in 2-D array format.
        :param solver: Which solver to use, cholesky or linalg solver.
        :return: The weightings for a tangency portfolio, the cash weightings, the asset weightings.
        """
        if not mu:
            mu = deepcopy(self.mu)
        if not cov:
            cov = deepcopy(self.cov)
        mu = mu.reshape(-1, 1)
        one = ones(mu.shape[0]).reshape(1, -1)

        if solver == "cholesky":
            x = array(self.compute_single_linear_system(b=mu, A=cov))
            w_T = x / dot(one, x)
            w_cash = 1 - e_sigma / sqrt(multi_dot([w_T.T, cov, w_T])) * sign(dot(one, x))
        else:
            w_T = dot(inv(cov), mu) / multi_dot([one, inv(cov), mu])
            w_cash = 1 - e_sigma / sqrt(multi_dot([w_T.T, cov, w_T])) * sign(multi_dot([one, inv(cov), mu]))

        w = (1 - w_cash) * w_T
        return w_T.reshape(1, -1)[0], w_cash[0][0], w.reshape(1, -1)[0]