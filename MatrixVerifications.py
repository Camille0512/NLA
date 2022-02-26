from numpy import matrix, array, dot, sum
from numpy.linalg import det


def check_lower_mtx(M: matrix):
    """
    To check whether a matrix is a lower matrix.
    :param M: Input matrix to be checked.
    :return: Whether it is a lower matrix.
    """
    if M.shape[0] != M.shape[1]:
        return False
    for i in range(M.shape[0]):
        m = ~(M[i, i + 1:] == 0)
        if sum(m) > 0:
            return False
    return True


def check_upper_mtx(M: matrix):
    """
    To check whether a matrix is a upper matrix.
    :param M: Input matrix to be checked.
    :return: Whether it is a upper matrix.
    """
    for i in range(M.shape[0]):
        m = ~(M[i, :i] == 0)
        if sum(m) > 0:
            return False
    return True


def check_non_singular(M):
    """
    Check whether the matrix is singular.
    :param M: Input matrix.
    :return: True (non singular) if determinant is not zero.
    """
    return True if det(M) != 0 else False


