from numpy.linalg import det, eig, multi_dot
from numpy import matrix, array, mean, dot, diag, sqrt
from math import sin, cos, pi
from collections import defaultdict
from Verifications import check_root
import pandas_market_calendars as mcal


def compute_tri_diagonal_symmetric(d: int, a: int, n: int, digit=6):
    """
    Solve the tri-diagonal symmetric matrix eigenvalues and corresponding eigenvectors.
    :param d: The diagonal entry value.
    :param a: The entry value next to diagonal entries.
    :param n: The size of the tri-diagonal matrix.
    :param digit: The digits to be kept for the result.
    :return: The eigenvalues and corresponding eigenvectors.
    """
    e_dict = defaultdict(list)
    for j in range(1, 1 + n):
        core = j * pi / (n + 1)
        eigenvalue = round(d - 2 * a * cos(core), digit)
        eigenvector = array([round(sin(i * core), digit) for i in range(1, n + 1)])
        e_dict[eigenvalue].append(eigenvector)
    return e_dict


def compute_root_quadratic_equation(a: float, b: float, c: float):
    """
    Compute the root of a quadratic equation.
    :param a: The coefficient of the quadratic term.
    :param b: The coefficient of the linear term.
    :param c: The constant of the equation.
    :return: The roots of the equation.
    """
    delta = b * b - 4 * a * c
    if not check_root(delta):
        raise ValueError("The quadratic equation does not has roots.")
    return (-b + sqrt(delta)) / (2 * a), (-b - sqrt(delta)) / (2 * a)


def mean_normalize(t_series: array):
    """
    Compute the mean normalized return matrix.
    :param t_series: The raw time series data in array format. The row s are time series for an asset.
    :return: The mean normalized return matrix
    """
    norm_series = []
    for series in t_series:
        mu = mean(series)
        norm_series.append(series - mu)
    return array(norm_series)


def mean_normalized_corr(norm_series: array):
    """
    Compute the sample correlation matrix by using normalized time series.
    :param norm_series:
    :return:
    """
    M = []
    for series in norm_series:
        paradigm = dot(series.reshape(1, -1), series.reshape(-1, 1))
        M.append(paradigm[0][0])
    M = diag(array(M))
    return multi_dot([M, norm_series, norm_series.T, M])


def trading_days_computation(ex: str, start: str, end: str, return_type="list"):
    """
    Compute the trading days between given start date and end date.
    :param ex: The exchange name, referring to the market.
    :param start: The start date, in format "%Y-%m-%d".
    :param end: The end date, in format "%Y-%m-%d".
    :param return_type: The return type of the date, either "date list" (list) or "date length" (int).
    :return: The trading date list between the given start date and end date.
    """
    symbol = mcal.get_calendar(ex)
    date_list = symbol.valid_days(start_date=start, end_date=end)
    return date_list if return_type == "list" else len(date_list)
