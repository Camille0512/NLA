from copy import deepcopy
from numpy.linalg import det
from numpy import array, dot, diag, zeros, log, exp


def trans_discount_factor_zero_rate(discount_factor: list, periods: list, period_type: str):
    len_df, len_p = len(discount_factor), len(periods)
    if len_df != len_p: raise ValueError("Wrong input, please check the length of the input lists.")

    zero_rate = []
    period_mapping = {
        "day": 365,
        "std_day": 360,
        "week": 52,
        "month": 12,
        "semi": 2
    }
    for i in range(len_df):
        # Continuous compounding
        zr = log(1 / discount_factor[i]) * (period_mapping[period_type] / periods[i])
        zero_rate.append(zr)
    return zero_rate


def compute_discount_factor(t: float, poly_params: dict):
    """
    Compute the discount factors.
    :param t: The given time. For bonds, it should be the cash flow date.
    :param poly_params: The polynomial parameters.
    :return: The discount factor of given time.
    """
    a, b, c, d = 0, 0, 0, 0
    for (l, r), params in poly_params.items():
        if l <= t <= r: a, b, c, d = params
    r = a + b * t + c * t * t + d * t * t * t
    return exp(-t * r)


class LU:
    def __init__(self, A=None, L=None, U=None, P=None, b=None, interval=None, interval_value=None):
        self.A = A
        self.L = L
        self.U = U
        self.P = P
        self.b = b
        self.n = len(self.A) if self.A is not None else None
        self.discount_factor_comp = {
            "TM": self._dfc_tm,
            "RP": self._dfc_rp
        }

    def create_clean_matrix(self):
        """
        Clear the matrix.
        :return: A clean matrix.
        """
        if self.n is None: print("Error! Please input matrix size.")
        return [[0] * self.n for i in range(self.n)]

    def forward_substitution(self, L=None, b=None, d=6):
        """
        Solve the Lx=b equation by using forward substitution, given L and b.
        :param L: The lower triangular matrix of the linear problem, in list format.
        :param b: The vector for the result.
        :param d: The decimal that would like to keep, default is 6 decimals.
        :return: The solved x list.
        """
        if L is None: L = deepcopy(self.L)
        if b is None: b = deepcopy(self.b)
        n = len(L)
        x = [None] * n
        x[0] = b[0] / L[0][0]
        for i in range(1, n):
            numerator = b[i] - sum([j * x[e] for e, j in enumerate(L[i][:i])])
            x[i] = round(numerator / L[i][i], d)
        return x

    def backward_substitution(self, U=None, b=None, d=6):
        """
        Solve the Ux=b equation by using forward substitution, given U and b.
        :param U: The upper triangular matrix of the linear problem, in list format.
        :param b: The vector for the result.
        :param d: The decimal that would like to keep, default is 6 decimals.
        :return: The solved x list.
        """
        if U is None: U = deepcopy(self.U)
        if b is None: b = deepcopy(self.b)
        n = len(U) - 1
        x = [None] * (n + 1)
        x[n] = b[n] / U[n][n]
        for i in range(1, n + 1):
            numerator = b[n - i] - sum([j * x[n - e] for e, j in enumerate(U[n - i][(n - i + 1):][::-1])])
            x[n - i] = round(numerator / U[n - i][n - i], d)
        return x

    def _update_LU_A(self, A, ind: int):
        """
        The core computation of the LU decomposition.
        :param A: Matrix A used in computatioon.
        :param ind: The current updating index.
        :return: No return. Update matrix A, L, U.
        """
        for k in range(ind, self.n):
            self.U[ind][k] = A[ind][k]
            self.L[k][ind] = A[k][ind] / A[ind][ind]
        for j in range(ind + 1, self.n):
            for k in range(ind + 1, self.n):
                A[j][k] = A[j][k] - self.L[j][ind] * self.U[ind][k]
        return A

    def lu_no_pivoting(self, A=None):
        """
        LU decomposition without pivoting given non-singular matrix A.
        Only exists if all the leading principal minors of the matrix are nonzero.
        :param A: The given non-singular matrix A.
        :return: The LU decomposition, lower triangular matrix L and upper triangular matrix U.
        """
        if A is None:
            if self.A is None: raise ValueError("Please initiate A or input A for the method.")
            A = deepcopy(self.A)
        n = len(A)
        verify = 1
        for i in range(1, n + 1):
            arr = array([a[:i] for a in A[:i]]).reshape(i, i)
            verify *= det(arr)
        if verify == 0:
            print("The given matrix has no LU decomposition without pivoting.")
            return

        self.L, self.U = self.create_clean_matrix(), self.create_clean_matrix()
        for i in range(0, n - 1):
            A = self._update_LU_A(A, i)
        self.L[n - 1][n - 1], self.U[n - 1][n - 1] = 1, A[n - 1][n - 1]
        return self.L, self.U

    def lu_row_pivoting(self, A=None, givenLU=False):
        """
        Compute the LU decomposition with row pivoting.
        Any non-singular matrix has an LU decomposition with row pivoting.
        :param A: The given non-singular matrix A.
        :param givenLU: Whether use the previous given L & U matrix, default False (not use).
        :return: The LU decomposition (lower triangular matrix L and upper triangular matrix U), permutation matrix P.
        """
        if A is None:
            if self.A is None: raise ValueError("Please initiate A or input A for the method.")
            A = deepcopy(self.A)
        n = len(A)
        if not givenLU: self.L, self.U = self.create_clean_matrix(), self.create_clean_matrix()
        P = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

        for i in range(0, n - 1):
            # Find the largest absolute value and put it in the first one.
            col = [elm[i] for elm in A[i:]]
            ind = col.index(max(col)) + i
            A[i], A[ind] = A[ind].copy(), A[i].copy()
            P[ind], P[i] = P[i], P[ind]
            if i > 0:
                self.L[ind], self.L[i] = self.L[i].copy(), self.L[ind].copy()

            A = self._update_LU_A(A, i)
        self.P = P
        self.L[n - 1][n - 1], self.U[n - 1][n - 1] = 1, A[n - 1][n - 1]
        return self.L, self.U, self.P

    def _dfc_tm(self, cate: str, M=None, b=None):
        if b is None: b = deepcopy(self.b)
        if cate.lower() == "fs":
            return self.forward_substitution(M, b) if M is not None else self.forward_substitution(b)
        elif cate.lower() == "bs":
            return self.backward_substitution(M, b) if M is not None else self.forward_substitution(b)

    def _dfc_rp(self, P=None, b=None):
        if self.b is None and b is None: raise EOFError("Please give the vector b.")
        if b is None: b = deepcopy(self.b)
        if P is None:
            if self.P is None: self.lu_row_pivoting()
            P = deepcopy(self.P)
        pb = list(dot(array(P), array(b)))
        y = self.forward_substitution(b=pb)
        return self.backward_substitution(b=y)

    def compute_discount_factor(self, cate: str, M=None, b=None):
        cate_list = cate.split("_")
        func = self.discount_factor_comp[cate_list[0]]
        return func(cate_list[1], M, b) if len(cate_list) > 1 is not None else func(M, b)

    def compute_linear_system(self, B: list, A=None, givenLU=False):
        """
        Solve the linear system for the same given matrix.
        :param B: The given results for the linear system (the right side of the equations).
        :param A: The given matrix for the linear system.
        :param givenLU: Whether use the previous given L & U matrix, default False (not use).
        :return: The solution for the linear system.
        """
        if A is None:
            if self.A is None: raise ValueError("Please initiate A or input A for the method.")
            A = deepcopy(self.A)

        if not givenLU: self.L, self.U = self.create_clean_matrix(), self.create_clean_matrix()
        _, _, P = self.lu_row_pivoting(A=A)

        B = array(B)
        s = B.shape[1]
        LS = diag(zeros(s))
        for i in range(s):
            v = dot(P, B[:, i])
            y = self.forward_substitution(b=v)
            x = self.backward_substitution(b=y)
            LS[:, i] = x
        res = [list(LS[row, :]) for row in range(s)]
        return res


class EquationSimulation(LU):
    def __init__(self):
        super().__init__()
        self.M = None
        self.v = None

    def _interpolation_matrix(self, intervals: list):
        """
        Generate the linear system for the cubic spline interpolation.
        :param intervals: The intervals for the problem.
        :return: The matrix to solve the cubic spline interpolation problem.
        """
        n = len(intervals) - 1
        M = diag(zeros(n * n))
        self.n = n * n

        M[0][2], M[0][3] = 2, 6 * intervals[0]
        for sec in range(n - 1):
            # Equation 1
            M[1 + sec * 4][0 + sec * 4] = 1
            M[1 + sec * 4][1 + sec * 4] = intervals[sec]
            M[1 + sec * 4][2 + sec * 4] = intervals[sec] * intervals[sec]
            M[1 + sec * 4][3 + sec * 4] = intervals[sec] * intervals[sec] * intervals[sec]

            # Equation 2
            M[2 + sec * 4][0 + sec * 4] = 1
            M[2 + sec * 4][1 + sec * 4] = intervals[sec + 1]
            M[2 + sec * 4][2 + sec * 4] = intervals[sec + 1] * intervals[sec + 1]
            M[2 + sec * 4][3 + sec * 4] = intervals[sec + 1] * intervals[sec + 1] * intervals[sec + 1]

            # Equation 3
            M[3 + sec * 4][1 + sec * 4] = 1
            M[3 + sec * 4][2 + sec * 4] = 2 * intervals[sec + 1]
            M[3 + sec * 4][3 + sec * 4] = 3 * intervals[sec + 1] * intervals[sec + 1]
            M[3 + sec * 4][5 + sec * 4] = -1
            M[3 + sec * 4][6 + sec * 4] = -2 * intervals[sec + 1]
            M[3 + sec * 4][7 + sec * 4] = -3 * intervals[sec + 1] * intervals[sec + 1]

            # Equation 4
            M[4 + sec * 4][2 + sec * 4] = 2
            M[4 + sec * 4][3 + sec * 4] = 6 * intervals[sec + 1]
            M[4 + sec * 4][6 + sec * 4] = -2
            M[4 + sec * 4][7 + sec * 4] = -6 * intervals[sec + 1]
        M[-3][-4] = 1
        M[-3][-3] = intervals[n - 1]
        M[-3][-2] = intervals[n - 1] * intervals[n - 1]
        M[-3][-1] = intervals[n - 1] * intervals[n - 1] * intervals[n - 1]
        M[-2][-4] = 1
        M[-2][-3] = intervals[n]
        M[-2][-2] = intervals[n] * intervals[n]
        M[-2][-1] = intervals[n] * intervals[n] * intervals[n]
        M[-1][-2], M[-1][-1] = 2, 6 * intervals[n]
        self.M = M

    def _interpolation_vector(self, f_x: list):
        """
        Generate the linear system for the cubic spline interpolation.
        :param f_x: The value corresponding to each of the intervals.
        :return: The matrix to solve the cubic spline interpolation problem.
        """
        if self.n is None: self.n = len(f_x) * len(f_x)
        val = [0] * self.n
        n = int(self.n ** 0.5)
        val[0] = 0
        for sec in range(n - 1):
            val[sec * 4 + 1] = f_x[sec]
            val[sec * 4 + 2] = f_x[sec + 1]
            val[sec * 4 + 3], val[sec * 4 + 4] = 0, 0
        val[-3], val[-2], val[-1] = f_x[-2], f_x[-1], 0
        self.v = val

    def cubic_spline_interpolation(self, intervals: list, f_x: list):
        """
        To solve the cubic spline interpolation problem.
        :param intervals: The intervals for the problem.
        :param f_x: The value corresponding to each of the intervals.
        :return: The parameters for the equations in each interval and the corresponding intervals.
        """
        self._interpolation_matrix(intervals)
        self._interpolation_vector(f_x)
        _ = self.lu_row_pivoting(self.M)
        x = self._dfc_rp(b=self.v)
        equations = {}
        for i in range(4):
            equations[(intervals[i], (intervals[i + 1]))] = x[i * 4: (i + 1) * 4]
        return x, equations
