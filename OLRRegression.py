from numpy import matrix, array, dot, sqrt, ones, sign, vstack, hstack, cov, mean, log, pi, exp
from numpy.linalg import inv, multi_dot
from scipy.stats import norm
from copy import deepcopy

from Decomposition import Cholesky


class OLR(Cholesky):
    def __init__(self, x=None, y=None):
        super().__init__()
        self.x = x
        self.y = y
        self.result_dict = {}
        self.y_hat = None
        self.d1 = None
        self.d2 = None

    def least_squares_with_one(self, x=None, y=None, solver="Cholesky"):
        """
        Solving the coefficient of a linear system by given input variables which includes ones vector.
        :param x: A 2-D array with column as feature dimension and row as sample dimension.
        :param y: The target 1-D array.
        :return: The estimated coefficient array for the linear system.
        """
        if x is None:
            if self.x is None:
                raise ValueError("Please input dependent data.")
            else:
                x = deepcopy(self.x)
        if y is None:
            if self.y is None:
                raise ValueError("Please input dependent data.")
            else:
                y = deepcopy(self.y)

        x, y = x.astype(float), y.astype(float)
        if solver == "Cholesky":
            coefficient = array(self.compute_single_linear_system(b=dot(x.T, y), A=dot(x.T, x)))
        else:
            coefficient = multi_dot([inv(dot(x.T, x)), x.T, y])
        self.result_dict["coefficient"] = coefficient
        return coefficient.reshape(1, -1)[0]

    def least_squares_without_one(self, A=None, y=None, include_intercept=True):
        """
        Solving the coefficient of a linear system by given input variables which do not include ones vector.
        :param A: A 2-D array with column as feature dimension and row as sample dimension, without 1 vector.
        :param y: The target 1-D array.
        :param include_intercept: Whether to add intercept to the input data, default True.
        :return: The estimated coefficient array for the linear system, and the computation result during the process.
        """
        if A is None:
            if self.x is None:
                raise ValueError("Please input dependent data.")
            else:
                A = deepcopy(self.x)
        if y is None:
            if self.y is None:
                raise ValueError("Please input dependent data.")
            else:
                y = deepcopy(self.y)

        A, y = A.astype(float), y.astype(float)
        A = A.reshape(max(A.shape), -1)
        cov_A, mu_A = cov(A.T), mean(A, axis=0)
        mu_y = mean(y)
        a, b, sigma = None, None, None
        if include_intercept:
            A = hstack([ones(A.shape[0]).reshape(-1, 1), A])
            coefficient = array(self.compute_single_linear_system(b=dot(A.T, y), A=dot(A.T, A)))
        else:
            sigma = []
            for series in A.T:
                covariance = cov(y.reshape(1, -1), series)
                sigma.append(covariance[0][1])
            sigma = array(sigma).reshape(-1, 1)
            b = dot(inv(cov_A), sigma)
            a = mu_y - multi_dot([mu_A, inv(cov_A), sigma])
            coefficient = vstack([a, b])
        self.result_dict = {
            "a": a,
            "b": b,
            "sigma": sigma,
            "covariance": cov_A,
            "mu_y": mu_y,
            "mu_A": mu_A,
            "coefficient": coefficient
        }
        return coefficient

    def least_squares_error(self, A=None, y=None):
        """
        The error of the OLR equation.
        :param A: A 2-D array with column as feature dimension and row as sample dimension, without 1 vector.
        :param y: The target 1-D array.
        :return:
        """
        if A is None:
            if self.x is None:
                raise ValueError("Please input dependent data.")
            else:
                A = deepcopy(self.x)
        if y is None:
            if self.y is None:
                raise ValueError("Please input dependent data.")
            else:
                y = deepcopy(self.y)
        coefficient = self.result_dict.get("coefficient")
        A = hstack([ones(A.shape[0]).reshape(-1, 1), A])
        y_hat = dot(A, coefficient.reshape(-1, 1))
        self.y_hat = deepcopy(y_hat)
        error = dot((y.reshape(-1, 1) - self.y_hat).reshape(1, -1), (y.reshape(-1, 1) - self.y_hat).reshape(-1, 1))
        return sqrt(error[0][0])

    @staticmethod
    def compute_error(y_real: array, y_hat: array):
        """
        Compute the error of given true value and estimated value.
        :param y_real: The ture value array.
        :param y_hat: The estimated value array.
        :return: The MSE.
        """
        y_real, y_hat = y_real.reshape(-1, 1), y_hat.reshape(-1, 1)
        error = dot((y_real - y_hat).T, y_real - y_hat)
        return sqrt(error[0][0])

    @staticmethod
    def polya(x):
        """
        A function that can be used to approximate normal distribution accurately.
        :param x: Input value.
        :return: The CDF value.
        """
        return 1 / 2 + (sign(x) / 2) * sqrt(1 - exp(-2 * (x ** 2) / pi))

    def _compute_option_price(self, initial: float, price: float, X: float, PVF: float, disc: float, t: float,
                              cate="call", function="Norm"):
        val = initial * sqrt(t)
        self.d1 = log(PVF / X * disc) / val + val / 2
        self.d2 = log(PVF / X * disc) / val - val / 2
        # self.d2 = self.d1 - val
        if function == "Norm":
            func = norm.cdf
        elif function == "polya":
            func = self.polya
        if cate == "call":
            f = PVF * func(self.d1) - X * disc * func(self.d2) - price
        else:
            f = X * disc * func(-self.d2) - PVF * func(-self.d1) - price
        f_ = PVF * sqrt(t / 2 * pi) * exp(-self.d1 ** 2 / 2)
        return f, f_

    def implied_volatility(self, option_prices: dict, X: float, t: float, initial=0.25, threshold=0.000001,
                           function="Norm"):
        """
        Compute options' implied volatility by using Newton's method.
        :param option_prices: The dictionary of option prices, contains both call and put.
        :param X: The strike price.
        :param t: The time to maturity.
        :param initial: The initial volatility, default 0.25.
        :param threshold: The threshold for Newton's method iteration, default 0.000001.
        :param function: The function that is going to use in the computation of BS model CDF item.
        :return: Both call and put implied volatility, stored in a dictionary.
        """
        coefficient = self.result_dict["coefficient"]
        PVF, disc = coefficient.ravel()[0], coefficient.ravel()[1]
        c, p = option_prices["call"], option_prices["put"]
        pre_call, call = initial - 1, initial
        while abs(call - pre_call) > threshold:
            pre_call = call
            f_c, f_ = self._compute_option_price(call, c, X, PVF, disc, t, cate="call", function=function)
            call -= f_c / f_
        pre_put, put = initial - 1, initial
        while abs(put - pre_put) > threshold:
            pre_put = put
            f_p, f_ = self._compute_option_price(put, p, X, PVF, disc, t, cate="put", function=function)
            put -= f_p / f_
        options = {X: {"call": call, "put": put}}
        return options


class PortfolioOptimize(Cholesky):
    def __init__(self, mu, covariance):
        super().__init__()
        self.mu = mu
        self.covariance = covariance

    def min_variance_weights(self, emu: float, rf: float, mu=None, covariance=None, solver="cholesky"):
        """
        Compute the weightings of the minimum variance portfolio.
        :param emu: The expected return of a portfolio.
        :param rf: The risk-free rate.
        :param mu: The return array of the risky assets, in 1-D array format.
        :param covariance: The covariance matrix, in 2-D array format.
        :param solver: Which solver to use, cholesky or linalg solver.
        :return: The optimal weightings array.
        """
        if mu is None:
            mu = deepcopy(self.mu)
        if covariance is None:
            covariance = deepcopy(self.covariance)
        mu = mu.reshape(-1, 1) - rf
        if solver == "cholesky":
            x = array(self.compute_single_linear_system(b=mu, A=covariance))
            w = (emu - rf) / dot(mu.T, x) * x
        else:
            w = (emu - rf) / multi_dot([mu.T, inv(covariance), mu]) * dot(inv(covariance), mu)
        w = w.reshape(1, -1)[0]
        return w, 1 - dot(ones(len(w)), w)

    def min_variance_std(self, emu: float, rf: float, mu=None, covariance=None, solver="cholesky"):
        """
        Compute the standard deviation of the minimum variance portfolio.
        :param emu: The expected return of a portfolio.
        :param rf: The risk-free rate.
        :param mu: The return array of the risky assets, in 1-D array format.
        :param covariance: The covariance matrix, in 2-D array format.
        :param solver: Which solver to use, cholesky or linalg solver.
        :return: The standard deviation of the minimum variance portfolio.
        """
        if mu is None:
            mu = deepcopy(self.mu)
        if covariance is None:
            covariance = deepcopy(self.covariance)
        mu = mu.reshape(-1, 1) - rf
        if solver == "cholesky":
            x = array(self.compute_single_linear_system(b=mu, A=covariance))
            sigma = (emu - rf) / sqrt(dot(mu.T, x))
        else:
            sigma = (emu - rf) / sqrt(multi_dot([mu.T, inv(covariance), mu]))
        return sigma.reshape(1, -1)[0][0]

    def min_var_tangency(self, emu: float, rf: float, mu=None, covariance=None, solver="cholesky"):
        """
        Using a tangency portfolio to construct the minimum variance portfolio.
        :param emu: The expected return of a portfolio.
        :param rf: The risk-free rate.
        :param mu: The return array of the risky assets, in 1-D array format.
        :param covariance: The covariance matrix, in 2-D array format.
        :param solver: Which solver to use, cholesky or linalg solver.
        :return: The weightings for a tangency portfolio, the cash weightings, the asset weightings.
        """
        if mu is None:
            mu = deepcopy(self.mu)
        if covariance is None:
            covariance = deepcopy(self.covariance)
        mu = mu.reshape(-1, 1) - rf
        one = ones(mu.shape[0]).reshape(1, -1)

        if solver == "cholesky":
            x = array(self.compute_single_linear_system(b=mu, A=covariance))
            w_T = x / dot(one, x)
        else:
            w_T = dot(inv(covariance), mu) / multi_dot([one, inv(covariance), mu])

        w_cash = 1 - (emu - rf) / dot(mu.T, w_T)
        w = (1 - w_cash) * w_T
        return w_T.reshape(1, -1)[0], w_cash[0][0], w.reshape(1, -1)[0]

    def max_return_weights(self, e_sigma: float, rf: float, mu=None, covariance=None, solver="cholesky"):
        """
        Compute the weightings of the maximum return portfolio.
        :param e_sigma: The expected volatility of a portfolio.
        :param rf: The risk-free rate.
        :param mu: The return array of the risky assets, in 1-D array format.
        :param covariance: The covariance matrix, in 2-D array format.
        :param solver: Which solver to use, cholesky or linalg solver.
        :return: The optimal weightings of assets and cash position.
        """
        if mu is None:
            mu = deepcopy(self.mu)
        if covariance is None:
            covariance = deepcopy(self.covariance)
        mu = mu.reshape(-1, 1) - rf
        if solver == "cholesky":
            x = array(self.compute_single_linear_system(b=mu, A=covariance))
            w = (e_sigma / sqrt(dot(mu.T, x))) * x
        else:
            w = (e_sigma / sqrt(multi_dot([mu.T, inv(covariance), mu]))) * dot(inv(covariance), mu)
        w = w.reshape(1, -1)[0]
        return w, 1 - dot(ones(len(w)), w)

    def max_return_mu(self, e_sigma: float, rf: float, mu=None, covariance=None, solver="cholesky"):
        """
        Compute the return of the maximum return portfolio.
        :param e_sigma: The expected volatility of a portfolio.
        :param rf: The risk-free rate.
        :param mu: The return array of the risky assets, in 1-D array format.
        :param cov: The covariance matrix, in 2-D array format.
        :param solver: Which solver to use, cholesky or linalg solver.
        :return: The return of the maximum return portfolio.
        """
        if mu is None:
            mu = deepcopy(self.mu)
        if covariance is None:
            covariance = deepcopy(self.covariance)
        mu = mu.reshape(-1, 1) - rf
        if solver == "cholesky":
            x = array(self.compute_single_linear_system(b=mu, A=covariance))
            p_mu = rf + e_sigma * sqrt(dot(mu.T, x))
        else:
            p_mu = rf + e_sigma * sqrt(multi_dot([mu.T, inv(covariance), mu]))
        return p_mu.reshape(1, -1)[0][0]

    def max_return_tangency(self, e_sigma: float, rf: float, mu=None, covariance=None, solver="cholesky"):
        """
        Using a tangency portfolio to construct maximum return portfolio.
        :param e_sigma: The expected volatility of a portfolio.
        :param rf: The risk-free rate.
        :param mu: The return array of the risky assets, in 1-D array format.
        :param covariance: The covariance matrix, in 2-D array format.
        :param solver: Which solver to use, cholesky or linalg solver.
        :return: The weightings for a tangency portfolio, the cash weightings, the asset weightings.
        """
        if mu is None:
            mu = deepcopy(self.mu)
        if covariance is None:
            covariance = deepcopy(self.covariance)
        mu = mu.reshape(-1, 1) - rf
        one = ones(mu.shape[0]).reshape(1, -1)

        if solver == "cholesky":
            x = array(self.compute_single_linear_system(b=mu, A=covariance))
            w_T = x / dot(one, x)
            w_cash = 1 - e_sigma / sqrt(multi_dot([w_T.T, covariance, w_T])) * sign(dot(one, x))
        else:
            w_T = dot(inv(covariance), mu) / multi_dot([one, inv(covariance), mu])
            w_cash = 1 - e_sigma / sqrt(multi_dot([w_T.T, covariance, w_T])) * sign(
                multi_dot([one, inv(covariance), mu]))

        w = (1 - w_cash) * w_T
        return w_T.reshape(1, -1)[0], w_cash[0][0], w.reshape(1, -1)[0]

    def tangency_portfolip(self, rf: float, mu=None, covariance=None, solver="cholesky"):
        """
        Computing the tangency portfolio asset allocation.
        :param rf: The risk-free rate.
        :param mu: The return array of the risky assets, in 1-D array format.
        :param covariance: The covariance matrix, in 2-D array format.
        :param solver: Which solver to use, cholesky or linalg solver.
        :return: The weightings for a tangency portfolio
        """
        if mu is None:
            mu = deepcopy(self.mu)
        if covariance is None:
            covariance = deepcopy(self.covariance)
        mu = mu.reshape(-1, 1) - rf
        one = ones(mu.shape[0]).reshape(1, -1)

        if solver == "cholesky":
            x = array(self.compute_single_linear_system(b=mu, A=covariance))
            w_T = x / dot(one, x)
        else:
            w_T = dot(inv(covariance), mu) / multi_dot([one, inv(covariance), mu])
        return w_T

    def min_variance_weights_no_cash(self, covariance=None, solver="cholesky"):
        """
        Compute the weightings of the minimum variance portfolio without cash position.
        :param covariance: The covariance matrix, in 2-D array format.
        :param solver: Which solver to use, cholesky or linalg solver.
        :return: The optimal weightings array.
        """
        if covariance is None:
            covariance = deepcopy(self.covariance)
        one = ones(covariance.shape[0]).reshape(-1, 1)
        if solver == "cholesky":
            x = array(self.compute_single_linear_system(b=one, A=covariance))
            w = 1 / dot(one.T, x) * x
        else:
            w = 1 / multi_dot([one.T, inv(covariance), one]) * dot(inv(covariance), one)
        w = w.reshape(1, -1)[0]
        return w, 1 - dot(ones(len(w)), w)

    def min_variance_std_no_cash(self, covariance=None, solver="cholesky"):
        """
        Compute the standard deviation of the minimum variance portfolio without cash position.
        :param covariance: The covariance matrix, in 2-D array format.
        :param solver: Which solver to use, cholesky or linalg solver.
        :return: The standard deviation of the minimum variance portfolio.
        """
        if covariance is None:
            covariance = deepcopy(self.covariance)
        one = ones(covariance.shape[0]).reshape(-1, 1)
        if solver == "cholesky":
            x = array(self.compute_single_linear_system(b=one, A=covariance))
            sigma = 1 / sqrt(dot(one.T, x))
        else:
            sigma = 1 / sqrt(multi_dot([one.T, inv(covariance), one]))
        return sqrt(sigma.reshape(1, -1)[0][0])