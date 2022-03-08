from numpy import matrix, array, dot, sum
from numpy.linalg import det, eigvals


def check_dimensions(A, B, cate="list"):
    """
    Check the input object size.
    :param A: Object 1.
    :param B: Object 2.
    :param cate: Category of the object, list (1 dimensional), list2 (2 dimensional), array.
    :return: Bool.
    """
    if cate == "list":
        return True if len(A) == len(B) else False
    elif cate == "list2":
        if len(A) == len(B):
            return True if len(A[0]) == len(B[0]) else False
        else:
            return True


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


def check_non_singular(M: matrix):
    """
    Check whether the matrix is singular.
    :param M: Input matrix to be checked.
    :return: True (non singular) if determinant is not zero.
    """
    return True if det(M) != 0 else False


def check_spd(M: matrix):
    """
    Check whether the matrix is symmetric positive definite.
    :param M: Input matrix to be checked.
    :return: Bool value shows whether the matrix is symmetric positive definite.
    """
    eigenvalue = eigvals(M)
    for i in eigenvalue:
        if i <= 0:
            return False
    return True
