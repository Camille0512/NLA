from numpy.linalg import det, eig
from numpy import matrix, array
from math import sin, cos, pi
from collections import defaultdict


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