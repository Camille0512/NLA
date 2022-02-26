import numpy as np
from Decomposition import LU, EquationSimulation, trans_discount_factor_zero_rate, compute_discount_factor

np.set_printoptions(suppress=True)


if __name__ == "__main__":
    # # Question 1
    # lu = LU()
    #
    # L = [
    #     [100, 0, 0, 0],
    #     [2, 102, 0, 0],
    #     [3, 3, 103, 0],
    #     [2, 2, 2, 102]
    # ]
    # b = [98.75, 102, 103.5, 105.5]
    # x = lu.forward_substitution(L, b)
    # print(x)
    #
    # L = [
    #     [100, 0, 0, 0],
    #     [6, 106, 0, 0],
    #     [8, 8, 108, 0],
    #     [5, 5, 5, 105]
    # ]
    # b = [98, 104, 111, 102]
    # x = lu.forward_substitution(L, b)
    # print("Forward decomposition")
    # print(x)
    # print("")
    #
    # U = [i[::-1] for i in L][::-1]
    # b = b[::-1]
    # x = lu.backward_substitution(U, b)
    # print("Backward decomposition")
    # print(x)
    # print("")
    #
    # A = [
    #     [2, -1, 3, 0],
    #     [-4, 5, -7, -2],
    #     [-2, 10, -4, -7],
    #     [4, -14, 8, 10]
    # ]
    # lu.n = len(A)
    # L, U = lu.lu_no_pivoting(A)
    # print("LU decomposition")
    # print(L)
    # print(U)
    # print("")
    #
    # A = [
    #     [1, 101, 0, 0],
    #     [2, 2, 102, 0],
    #     [5, 0, 105, 0],
    #     [2.5, 2.5, 2.5, 102.5]
    # ]
    # b = [100.6, 103.3, 107.3, 110.3]
    # lu.n = len(A)
    # L, U, P = lu.lu_row_pivoting(A)
    # print("LU decomposition with row pivoting")
    # print(L)
    # print(U)
    # print(P)
    # print("")
    #
    # print("===========")
    # print("Discount factors computation")
    #
    # x = lu.compute_discount_factor("RP", P, b)
    # print(x)
    # print("")
    #
    # lu.L = [
    #     [100, 0, 0, 0],
    #     [6, 106, 0, 0],
    #     [8, 8, 108, 0],
    #     [5, 5, 5, 105]
    # ]
    # lu.b = [98, 104, 111, 102]
    # x = lu.compute_discount_factor("TM_fs")
    # print(x)

    # # Question 2
    # A = [
    #     [2, -1, 0, 1],
    #     [-2, 0, 1, -1],
    #     [4, -1, 0, 1],
    #     [4, -3, 0, 2]
    # ]
    # L = [
    #     [1, 0, 0, 0],
    #     [1, 1, 0, 0],
    #     [-0.5, 0.25, 1, 0],
    #     [0.5, 0.25, 0, 1]
    # ]
    # U = [
    #     [4, -1, 0, 1],
    #     [0, -2, 0, 1],
    #     [0, 0, 1, -0.75],
    #     [0, 0, 0, 0.25]
    # ]
    # P = [
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1],
    #     [0, 1, 0, 0],
    #     [1, 0, 0, 0]
    # ]
    #
    # b = [3, -1, 0, 2]
    # lu = LU(A, L, U, P, b)
    #
    # x = lu.compute_discount_factor("RP")
    # print(x)
    #
    # lu2 = LU(A, b=b)
    # x = lu2.compute_discount_factor("RP")
    # print(x)

    # B = np.diag(np.ones(4))
    # res = lu.compute_linear_system(list(B))
    # print('res:\n', res)
    #
    # for i in range(4):
    #     lu = LU(A, L, U, P, B[i])
    #     x = lu.compute_discount_factor("RP")
    #     print(x)

    # # Question 3
    # lu = LU()
    # interval = [0, 2/12, 5/12, 11/12, 15/12]
    # fx = [0.999973, 0.998, 0.9935, 0.982, 0.9775]
    # x = lu.cubic_spline_interpolation(interval, fx)
    # print("x:", x)
    # for i in range(4):
    #     print(x[i * 4 : (i + 1) * 4])

    # # Question 4
    disc_factor = [0.998, 0.9935, 0.982, 0.9775]
    ps = [2, 5, 11, 15]
    zr = [round(i, 6) for i in trans_discount_factor_zero_rate(disc_factor, ps, "month")]

    lu = EquationSimulation()
    interval = [0, 2 / 12, 5 / 12, 11 / 12, 15 / 12]
    fx = [0.01] + zr
    x, eq = lu.cubic_spline_interpolation(interval, fx)

    res, new_interval = 0, [1 / 12, 4 / 12, 7 / 12, 10 / 12, 13 / 12]
    for i in range(5):
        t = new_interval[i]
        d = compute_discount_factor(t, eq)
        if i == 4: res += 100 * d
        res += 0.625 * d
    print(res)

    # # # Question 5
    # fx = [0.005, 0.0065, 0.0085, 0.0105, 0.012]
    #
    # lu = EquationSimulation()
    # interval = [0, 2 / 12, 6 / 12, 1, 20 / 12]
    # x, eq = lu.cubic_spline_interpolation(interval, fx)
    # print(eq)
    #
    # res, new_interval = 0, [1 / 12, 4 / 12, 7 / 12, 10 / 12, 13 / 12]
    # for i in range(5):
    #     t = new_interval[i]
    #     d = compute_discount_factor(t, eq)
    #     if i == 4: res += 100 * d
    #     res += 0.75 * d
    # print(res)