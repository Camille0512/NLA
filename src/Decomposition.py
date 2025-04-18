from copy import deepcopy
from numpy.linalg import det
from numpy import array, dot, diag, zeros, log, exp, ones, matrix, append, around, square, sqrt, matrix, around
import matplotlib.pyplot as plt
from pandas import DataFrame

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))  # Adds the current dir to Python path
import Verifications as v


def trans_discount_factor_zero_rate(discount_factor: list, periods: list, period_type: str):
    """
    Transform discount factor into spot rate.
    :param discount_factor: Given discount factor list.
    :param periods: Given period list.
    :param period_type: The period type. Could be day, std_day, week, month, semi.
    :return: Spot rates corresponding to given discount factors.
    """
    len_df, len_p = len(discount_factor), len(periods)
    if len_df != len_p:
        raise ValueError("Wrong input, please check the length of the input lists.")

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
        if l <= t <= r:
            a, b, c, d = params
    r = a + b * t + c * t * t + d * t * t * t
    return exp(-t * r)


class LU:
    def __init__(self, A=None, L=None, U=None, P=None, b=None):
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
        if self.n is None:
            raise ValueError("Error! Please input matrix size.")
        return [[0] * self.n for i in range(self.n)]

    def forward_substitution(self, L=None, b=None, d=6):
        """
        Solve the Lx=b equation by using forward substitution, given L and b.
        :param L: The lower triangular matrix of the linear problem, in list format.
        :param b: The vector for the result.
        :param d: The decimal that would like to keep, default is 6 decimals.
        :return: The solved x list.
        """
        if L is None:
            L = deepcopy(self.L)
        if b is None:
            b = deepcopy(self.b)
        n = len(L)
        x = [None] * n
        x[0] = b[0] / L[0][0]
        for i in range(1, n):
            numerator = b[i] - sum([j * x[e] for e, j in enumerate(L[i][:i])])
            x[i] = around(numerator / L[i][i], d)
        return x

    def backward_substitution(self, U=None, b=None, d=6):
        """
        Solve the Ux=b equation by using forward substitution, given U and b.
        :param U: The upper triangular matrix of the linear problem, in list format.
        :param b: The vector for the result.
        :param d: The decimal that would like to keep, default is 6 decimals.
        :return: The solved x list.
        """
        if U is None:
            U = deepcopy(self.U)
        if b is None:
            b = deepcopy(self.b)
        n = len(U) - 1
        x = [None] * (n + 1)
        x[n] = b[n] / U[n][n]
        for i in range(1, n + 1):
            numerator = b[n - i] - sum([j * x[n - e] for e, j in enumerate(U[n - i][(n - i + 1):][::-1])])
            x[n - i] = around(numerator / U[n - i][n - i], d)
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
            if self.A is None:
                raise ValueError("Please initiate A or input A for the method.")
            A = deepcopy(self.A)
        if self.n is None:
            self.n = len(A)
        n = self.n
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
            if self.A is None:
                raise ValueError("Please initiate A or input A for the method.")
            A = deepcopy(self.A)
        if self.n is None:
            self.n = A.shape[0]
        n = self.n
        if not givenLU:
            self.L, self.U = self.create_clean_matrix(), self.create_clean_matrix()
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
        if b is None:
            b = deepcopy(self.b)
        if cate.lower() == "fs":
            return self.forward_substitution(M, b) if M is not None else self.forward_substitution(b)
        elif cate.lower() == "bs":
            return self.backward_substitution(M, b) if M is not None else self.forward_substitution(b)

    def _dfc_rp(self, A=None, P=None, b=None):
        if self.b is None and b is None:
            raise EOFError("Please give the vector b.")
        if b is None:
            b = deepcopy(self.b)
        if A is not None:
            self.lu_row_pivoting(A=A)
        if P is None:
            if self.P is None:
                self.lu_row_pivoting()
            P = deepcopy(self.P)
        pb = list(dot(array(P), array(b)))
        y = self.forward_substitution(b=pb)
        return self.backward_substitution(b=y)

    def compute_discount_factor(self, cate: str, M=None, b=None):
        """
        Coumpute discount factors by solving a linear system.
        :param cate: The method to be used in solving the linear system. Could be "TM" or "RP".
        :param M: Given matrix corresponding to the left side of the linear system.
        :param b: Given vector corresponding to the right side of the linear system.
        :return: The solution (discount factor) for the linear system.
        """
        cate_list = cate.split("_")
        func = self.discount_factor_comp[cate_list[0]]
        return func(cate_list[1], M, b) if cate_list is not None and len(cate_list) > 1 else func(M, b)

    def compute_linear_system(self, B: list, A=None, givenLU=False):
        """
        Solve the multiple linear system for the same given matrix.
        :param B: The given results for the linear system (the right side of the equations), a list of list.
        :param A: The given matrix for the linear system.
        :param givenLU: Whether use the previous given L & U matrix, default False (not use).
        :return: The solution for the multiple linear system.
        """
        if A is None:
            if self.A is None:
                raise ValueError("Please initiate A or input A for the method.")
            A = deepcopy(self.A)
        if self.n is None:
            self.n = self.A.shape[0]

        if not givenLU:
            self.L, self.U = self.create_clean_matrix(), self.create_clean_matrix()
        _, _, P = self.lu_row_pivoting(A=A)

        B = array(B)
        s = B.shape[1]
        LS = diag(zeros(s))
        for i in range(s):
            b = dot(P, B[:, i])
            y = self.forward_substitution(b=b)
            x = self.backward_substitution(b=y)
            LS[:, i] = x
        res = [list(LS[row, :]) for row in range(s)]
        return res


class Cholesky(LU):
    def __init__(self, A=None, U=None, b=None):
        super().__init__(A, U, b)
        self.U_t = None

    def compute_U(self, inx=1, A=None, U=None):
        """
        Update of U for the Cholesky decomposition.
        :param inx: The index for the position of U, a pointer.
        :param A: The matrix of the linear system.
        :param U: The upper triangular matrix for matrix A.
        :return: Computed U.
        """
        p = inx - 1
        if U is None:
            U = self.U.copy()
        if A is None:
            A = self.A.copy()
        U[p][p] = sqrt(A[p][p])
        for k in range(p + 1, self.n):
            U[p][k] = A[p][k] / U[p][p]
        return U

    def _update_A(self, inx: int, A=None):
        """
        Update the entries of matrix A.
        :param inx: The index for the position of A and U, a pointer.
        :param A: The matrix A of the linear system.
        :return: The updated matrix A.
        """
        if A is None:
            raise ValueError("Please input matrix A.")
        for j in range(inx, self.n):
            for k in range(j, self.n):
                A[j][k] = float(A[j][k] - self.U[inx - 1][j] * self.U[inx - 1][k])
        return A

    def Cholesky_decomposition(self, A=None):
        """
        Implementation of Cholesky decomposition on matrix A.
        :param A: The input matrix A. Default None. When there is not initiation, A must be not None.
        :return: The decomposed triangular matrix U.
        """
        if A is None and self.A is None:
            raise ValueError("Please initialize matrix A or input matrix A.")
        if A is None:
            A = self.A.copy()

        self.n = len(A[0])
        if not v.check_spd(matrix(A)):
            raise ValueError("The matrix cannot be implemented Cholesky decomposition.")
        self.U = self.create_clean_matrix()

        A = A.astype(float)
        for i in range(1, self.n):
            self.U = self.compute_U(A=A, inx=i)
            A = self._update_A(A=A, inx=i)
        self.U[self.n - 1][self.n - 1] = sqrt(A[self.n - 1][self.n - 1])
        return self.U

    def compute_single_linear_system(self, b: list, A=None, U=None):
        """
        Solve a single linear system for the matrix A.
        :param b: The right side of the linear system.
        :param A: The matrix A.
        :param U: The upper triangular matrix U of the Cholesky decomposition of A.
        :return: The solution for the single linear system x.
        """
        if A is None:
            if self.A is None:
                raise ValueError("Please initiate A or input A for the method.")
            A = deepcopy(self.A)
        if self.n is None:
            self.n = A.shape[0]
        A = A.astype(float)
        if U is None:
            self.U = self.Cholesky_decomposition(deepcopy(A))
        else:
            self.U = deepcopy(U)

        self.U_t = [list(arr) for arr in array(self.U).T]
        y = self.forward_substitution(L=self.U_t, b=b)
        x = self.backward_substitution(U=self.U, b=y)
        return x

    def compute_linear_system(self, B: list, A=None, given_U=False):
        """
        Solve the multiple linear system for the same given matrix.
        :param B: The given vector for the linear system (the right side of the equations).
        :param A: The given matrix for the linear system.
        :param given_U: Whether use the previous given U matrix, default False (not use).
        :return: The solution for the multiple linear system x list.
        """
        if A is None:
            if self.A is None:
                raise ValueError("Please initiate A or input A for the method.")
            A = deepcopy(self.A)
        if self.n is None:
            self.n = self.A.shape[0]
        A = A.astype(float)

        if not given_U:
            self.U = self.Cholesky_decomposition(A)
        B = array(deepcopy(B)).reshape(-1, 1)
        B = B.astpye(float)
        s = B.shape[1]
        X = diag(zeros(s))
        for e, b in enumerate(B):
            x = self.compute_single_linear_system(b=b, A=A)
            X[:, e] = x
        res = [list(X[row, :]) for row in range(s)]
        return res


class EquationSimulation(LU):
    def __init__(self):
        super().__init__()
        self.M = None
        self.v = None

    def interpolation_matrix_lu(self, intervals: list):
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

    def interpolation_matrix_cd(self, intervals: list):
        """
        Generate the linear system for the cubic spline interpolation by using Cholesky decomposition.
        :param intervals: The intervals for the problem.
        :return: The matrix to solve the cubic spline interpolation problem.
        """
        if self.n is None:
            n = len(intervals) - 2
        else:
            n = self.n
        M = [[0] * n for i in range(n)]
        for i in range(n):
            M[i][i] = 2 * (intervals[i + 2] - intervals[i])
            if i < n - 1:
                M[i][i + 1] = intervals[i + 2] - intervals[i + 1]
            if i > 0:
                M[i][i - 1] = intervals[i + 1] - intervals[i]
        self.M = M

    def _interpolation_vector_lu(self, f_x: list):
        """
        Generate the linear system for the cubic spline interpolation.
        :param f_x: The value corresponding to each of the intervals.
        :return: The matrix to solve the cubic spline interpolation problem.
        """
        if self.n is None:
            self.n = len(f_x) * len(f_x)
        val = [0] * self.n
        n = int(self.n ** 0.5)
        val[0] = 0
        for sec in range(n - 1):
            val[sec * 4 + 1] = f_x[sec]
            val[sec * 4 + 2] = f_x[sec + 1]
            val[sec * 4 + 3], val[sec * 4 + 4] = 0, 0
        val[-3], val[-2], val[-1] = f_x[-2], f_x[-1], 0
        self.v = val

    def _interpolation_vector_cd(self, interval: list, f_x: list):
        """
        Generate the vector of the right side of the linear system for the cubic spline interpolation.
        :param interval: The intervals of the objects.
        :param f_x: The value corresponding to each of the intervals.
        :return: The matrix to solve the cubic spline interpolation problem.
        """
        z = []
        for i in range(self.n):
            res1 = (f_x[i + 2] - f_x[i + 1]) / (interval[i + 2] - interval[i + 1])
            res2 = (f_x[i + 1] - f_x[i]) / (interval[i + 1] - interval[i])
            z.append(6 * (res1 - res2))
        self.v = z

    def cubic_spline_interpolation_lu(self, intervals: list, f_x: list):
        """
        To solve the cubic spline interpolation problem by using Cholesky decomposition.
        Generate the pricing equation for bonds.
        :param intervals: The intervals for the problem.
        :param f_x: The value corresponding to each of the intervals.
        :return: The parameters for the equations in each interval and the corresponding intervals.
        """
        if not v.check_dimensions(intervals, f_x):
            raise ValueError("Please input the invervals and f_x with same dimensions.")
        self.interpolation_matrix_lu(intervals)
        self._interpolation_vector_lu(f_x)
        _ = self.lu_row_pivoting(self.M)
        x = self._dfc_rp(b=self.v)
        equations, m = {}, len(intervals) - 1
        for i in range(m):
            equations[(intervals[i], (intervals[i + 1]))] = x[i * m: (i + 1) * m]
        return x, equations

    def cubic_spline_interpolation_cd(self, intervals: list, f_x: list):
        """
        To solve the cubic spline interpolation problem by using Cholesky decomposition.
        Generate the pricing equation for bonds.
        :param intervals: The intervals for the problem.
        :param f_x: The value corresponding to each of the intervals.
        :return: The parameters for the equations in each interval and the corresponding intervals.
        """
        if not v.check_dimensions(intervals, f_x):
            raise ValueError("Please input the invervals and f_x with same dimensions.")
        self.n = len(intervals) - 2
        self._interpolation_vector_cd(intervals, f_x)
        self.interpolation_matrix_cd(intervals=intervals)
        w = self._dfc_rp(A=self.M, b=self.v)
        w = [0] + w + [0]
        equations, m = {}, len(intervals) - 1
        a, b, c, d, q, r = [0] * m, [0] * m, [0] * m, [0] * m, [0] * m, [0] * m
        for i in range(m):
            c[i] = (w[i] * intervals[i + 1] - w[i + 1] * intervals[i]) / (2 * (intervals[i + 1] - intervals[i]))
            d[i] = (w[i + 1] - w[i]) / (6 * (intervals[i + 1] - intervals[i]))
        for i in range(m):
            q[i] = f_x[i] - c[i] * (intervals[i] ** 2) - d[i] * (intervals[i] ** 3)
            r[i] = f_x[i + 1] - c[i] * (intervals[i + 1] ** 2) - d[i] * (intervals[i + 1] ** 3)
        for i in range(m):
            a[i] = (q[i] * intervals[i + 1] - r[i] * intervals[i]) / (intervals[i + 1] - intervals[i])
            b[i] = (r[i] - q[i]) / (intervals[i + 1] - intervals[i])
            equations[(intervals[i], (intervals[i + 1]))] = [a[i], b[i], c[i], d[i]]
        return w, equations


class OnePeriodMarketModel(LU):
    def __init__(self, init_price_vec=None, states=None, option_info=None):
        """
        Initiate the variables.
        :param init_price_vec (array, list, or pd.Series): The prices of securities at time 0.
        :param states (array, list, or pd.Series): The states of the following periods.
        :param option_info (dictionary): The option information. Key is the option type, value is the strike price.
        """
        super().__init__()
        self.init_price_vec = array(init_price_vec).reshape(-1, 1)
        self.states = array(states).reshape(-1, 1)
        self.option_info = option_info
        self.payoff = None
        self.Q = None
        self.options = None
        self.strikes = None

    @staticmethod
    def __option_info(option_info):
        """
        Extract the option information, including the put call type and their corresponding strikes.
        :param option_info: The option(s) information.
        :return: Option types (put/call), corresponding strikes.
        """
        options, strikes = [], []
        for key, val in option_info.items():
            options.append(key[0])
            strikes.append(val)
        options, strikes = array(options).reshape(-1, 1), array(strikes).reshape(-1, 1)
        # Calls will be 1, puts will be -1.
        options = (options == "P") * (-2) + 1
        return options, strikes

    def _option_payoff_matrix(self, option_info=None, update=True):
        """
        Compute the payoff(s) matrix of given information.
        :param option_info: The option(s) information. Default None.
        :param update: Whether update the class variable (payoff matrix).
        :return: The payoff(s) matrix of given information, options position, strikes, option_info.
        """
        if len(self.states.shape) == 1:
            self.states = array([self.states])
        states = self.states
        if option_info is None:
            option_info = deepcopy(self.option_info)
        options, strikes = self.__option_info(option_info)

        # Compute the payoff matrix
        strikes = dot(strikes, ones(states.shape[0]).reshape(1, -1))
        payoff = options * (states.reshape(1, -1) - strikes)
        payoff = zeros(strikes.shape) + (payoff > 0) * payoff
        if update:
            self.payoff, self.options, self.strikes = payoff, options, strikes
        return payoff, options, strikes

    @staticmethod
    def check_complete(M: matrix):
        """
        Check whether the matrix is square and nonsingular.
        Used in checking the complete market.
        :param M: The input matrix to be checked whether is complete.
        :return: Whether the market is complete.
        """
        row, col = M.shape
        if row != col:
            return False
        if not v.check_non_singular(M):
            return False
        return True

    def check_arbitrage_free(self, M: matrix, S0: array, options=None):
        """
        Check whether the given payoff matrix gives a solution all larger than 0.
        The is to check the matrix of a one period model is arbitrage-free.
        :param M: The payoff matrix.
        :param S0: The price vector at time 0.
        :param options (array): The option type information.
        :return: Whether it is arbitrage free.
        """
        row, col = M.shape
        if row != col:
            raise ValueError("Please input a square matrix")

        # If M is made up of a lower triangular matrix and an upper triangular matrix.
        if options is None: options = deepcopy(self.options)
        if self.options is None: raise ValueError("Please input options type information.")
        cut = sum(options == -1)[0]
        m1, m2 = M[:cut, :cut], M[cut:, cut:]
        M, S0 = [list(i) for i in M], list(S0.reshape(1, -1)[0])
        if v.check_lower_mtx(m1) + v.check_upper_mtx(m1) > 0 and v.check_lower_mtx(m2) + v.check_upper_mtx(m2) > 0:
            if v.check_lower_mtx(m1):
                m1, m2 = [list(i) for i in m1], [list(i) for i in m2]
                q1 = self.forward_substitution(L=m1, b=S0[:cut])
                q2 = self.backward_substitution(U=m2, b=S0[cut:])
            else:
                m1, m2 = [list(i) for i in m1], [list(i) for i in m2]
                q1 = self.backward_substitution(U=m1, b=S0[:cut])
                q2 = self.forward_substitution(L=m2, b=S0[cut:])
            Q = append(q1, q2)
            self.Q = Q
            return False if sum(array(Q) < 0) > 0 else True

        # If M is a normal square matrix.
        lpm = 1
        for i in range(row):
            lpm *= i
        # LU Decomposition
        if lpm == 0:
            _ = self.lu_no_pivoting(A=M)
            b = S0
        else:
            _ = self.lu_row_pivoting(A=M)
            b = list(dot(self.P, S0))
        # Solve linear system
        Q = self.compute_discount_factor("RP", M=M, b=b)
        self.Q = Q
        return False if sum(array(Q) < 0) > 0 else True

    def option_pricing_model(self, M=None, S0=None, options=None, update=True):
        """
        Get the option pricing model by using payoff matrix, St0, and selected options.
        :param M: The payoff matrix. Can be computed or given directly.
        :param S0: The price vector at time 0. Should be given or initiated.
        :param options: The option type information (array).
        :param update: Whether update the class variable (payoff matrix).
        :return: The pricing model vector Q, and whether it is arbitrage free.
            Q is the discounted value of the expected cash flow.
        """
        af = True
        if M is None:
            try:
                M, _, _ = self._option_payoff_matrix(update=update)
            except:
                raise ValueError("Please either initiate the variables or input payoff matrix.")
        if S0 is None:
            S0 = deepcopy(self.init_price_vec)
        if not self.check_complete(M):
            raise ValueError("The model is not complete.")
        if not self.check_arbitrage_free(M, S0, options):
            af = False
        return self.Q, af

    def option_pricing(self, option_info: dict, Q=None, d=0, update=False):
        """
        Option pricing by using computed Q and new incoming option type and strike.
        :param option_info: The option information. In a dictionary format, key is the option type, value is the strike.
        :param Q: The pricing model vector.
        :param d: The number of decimals to be kept.
        :param update: Whether update the class variable (payoff matrix).
        :return: The option price vector corresponding to each states.
        """
        if Q is None:
            Q = deepcopy(self.Q)
        payoff, options, strikes = self._option_payoff_matrix(option_info, update=update)
        price = dot(payoff, Q)
        option_price_info = {}
        for e, (opt, X) in enumerate(zip(options, strikes)):
            round_str = ":.{}f".format(d)
            X = ("{" + round_str + "}").format(X[0])
            key = ""
            if opt == 1:
                key += "C" + str(X)
            else:
                key += "P" + str(X)
            option_price_info[key] = price[e]
        return option_price_info

    def compute_abs_error(self, option_price: dict, Q=None, d=0):
        """
        Compute the Average Absolute Error of the model estimated price and the real price of the corresponding same
        options.
        :param option_price: The option information. In a dictionary format, key is the option type, value is the price.
        :param Q: The pricing model vector.
        :param d: The number of decimals to be kept.
        :return: The average absolute error of the model.
        """
        option_info = {k: int(k[1:]) if float(k[1:]) % 5 == 0 else float(k[1:]) for k in option_price.keys()}
        model_prices = self.option_pricing(option_info=option_info, Q=Q)
        abse = 0
        for k in model_prices.keys():
            abse += abs(model_prices.get(k, 0) - option_price.get(k, 0)) / option_price.get(k, 0)
        return abse / len(model_prices.keys())

    def compute_mse_error(self, option_price: dict, Q=None, d=0):
        """
        Compute the MSE of the model estimated price and the real price of the corresponding same options.
        :param option_price: The option information. In a dictionary format, key is the option type, value is the price.
        :param Q: The pricing model vector.
        :param d: The number of decimals to be kept.
        :return: The root-mean-squared error (MSE) of the model.
        """
        option_info = {k: int(k[1:]) if float(k[1:]) % 5 == 0 else float(k[1:]) for k in option_price.keys()}
        model_prices = self.option_pricing(option_info=option_info, Q=Q)
        mse = 0
        for k in model_prices.keys():
            mse += square((model_prices.get(k, 0) - option_price.get(k, 0)) / option_price.get(k, 0))
        return mse

    def compute_RMS(self, option_price: dict, Q=None, d=0):
        """
        Compute the RMSE of the model estimated price and the real price of the corresponding same options.
        :param option_price: The option information. In a dictionary format, key is the option type, value is the price.
        :param Q: The pricing model vector.
        :param d: The number of decimals to be kept.
        :return: The root-mean-squared error (RMSE) of the model.
        """
        if Q is None:
            Q = deepcopy(self.Q)
        mse = self.compute_mse_error(option_price, Q)
        rmse = sqrt(mse / len(option_price.keys()))
        return rmse

    def graph_error_distribution(self, err_df: DataFrame, inx: int, cols: dict, save_fig=False, fig_name="graph"):
        """
        Graph the error of each securities.
        :param err_df: The error dataframe.
        :param inx: The index to separate calls and puts.
        :param cols: A dictionary corresponding to security, error, and market price columns of the error dataframe.
        :param save_fig: Whether to save config.
        :param fig_name: The figure name to be saved.
        :return: Plot a graph or save the graph.
        """
        opt, err, mkt = cols["opt"], cols["err"], cols["mkt"]
        cx, cy = array([int(i[1:]) for i in err_df[opt] if i[0] == "C"]), array(err_df.loc[:inx, err])
        px, py = array([int(i[1:]) for i in err_df[opt] if i[0] == "P"]), array(err_df.loc[inx+1:, err])
        c_price, p_price = array(err_df.loc[:inx, mkt]), array(err_df.loc[inx+1:, mkt])
        plt.figure(figsize=(15, 6))
        ax = plt.axes()
        ax_ = ax.twinx()
        ax.plot(cx, cy, color="orange", linewidth=2, label="Call")
        ax.plot(px, py, color="seagreen", linewidth=2, label="Put")
        ax_.plot(cx, c_price, color="yellow", linewidth=2, label="Call")
        ax_.plot(px, p_price, color="cornflowerblue", linewidth=2, label="Put")
        for i in self.option_info.keys():
            t, x_point = i[0], int(i[1:])
            ax.scatter(x_point, 0, marker="*", s=300, color="red", edgecolors="black")
        ax.set_xlabel("Error", fontsize=12)
        ax.set_ylabel("Strike", fontsize=12)
        ax_.set_ylabel("Market Price", fontsize=12)
        ax.legend()
        ax.grid(True)
        ax.set_title("Error Trend of All the Securities", fontsize=16)
        if save_fig:
            plt.savefig(fig_name + ".png")
        else:
            plt.show()