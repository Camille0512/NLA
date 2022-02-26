import numpy as np
from collections import OrderedDict
from Decomposition import LU, EquationSimulation, OnePeriodMarketModel, trans_discount_factor_zero_rate, \
    compute_discount_factor

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

    # # # Question 4
    # disc_factor = [0.998, 0.9935, 0.982, 0.9775]
    # ps = [2, 5, 11, 15]
    # zr = [round(i, 6) for i in trans_discount_factor_zero_rate(disc_factor, ps, "month")]
    #
    # lu = EquationSimulation()
    # interval = [0, 2 / 12, 5 / 12, 11 / 12, 15 / 12]
    # fx = [0.01] + zr
    # x, eq = lu.cubic_spline_interpolation(interval, fx)
    #
    # res, new_interval = 0, [1 / 12, 4 / 12, 7 / 12, 10 / 12, 13 / 12]
    # for i in range(5):
    #     t = new_interval[i]
    #     d = compute_discount_factor(t, eq)
    #     if i == 4: res += 100 * d
    #     res += 0.625 * d
    # print(res)

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

    # For Q5 & Q6
    all_securities = {
        "P1175": 46.6, "P1200": 51.55, "P1225": 57.15, "P1250": 63.3, "P1275": 70.15,
        "P1300": 77.7, "P1325": 86.2, "P1350": 95.3, "P1375": 105.3,
        "P1400": 116.55, "P1425": 129, "P1450": 143.2,
        "P1500": 173.95, "P1550": 210.8, "P1575": 230.9, "P1600": 252.4,
        "C1175": 225.4, "C1200": 205.55, "C1225": 186.2, "C1250": 167.5, "C1275": 149.15,
        "C1300": 131.7, "C1325": 115.25, "C1350": 99.55, "C1375": 84.9, "C1400": 71.1,
        "C1425": 58.7, "C1450": 47.25, "C1500": 29.25, "C1550": 15.8, "C1575": 11.1, "C1600": 7.9
    }
    # Question 5
    model_opt = ["P1175", "P1200", "P1250", "P1350", "C1350", "C1375", "C1450", "C1550", "C1600"]
    securities = {k: float(k[1:]) for k in all_securities.keys() if k in model_opt}
    other_securities = {k: v for k, v in all_securities.items() if k not in model_opt}

    St0 = np.array([val for k, val in all_securities.items() if k in model_opt])
    states = [1187.5, 1225, 1300, 1362.5, 1412.5, 1500, 1575]
    f_states, l_states = [800, 950, 1100], [1650, 1700, 1800]
    states_comb = []
    for f in f_states:
        for l in l_states:
            states_comb.append([f] + states + [l])
    print("State combinations: {}\n{}".format(len(states_comb), states_comb))

    for states in states_comb:
        print("{} - {}".format(states[0], states[-1]))
        opm = OnePeriodMarketModel(init_price_vec=St0, states=states, option_info=securities)
        Q = opm.option_pricing_model()
        print("Q:", Q)
        new_option = {"C1300": 1300}
        option_price = opm.option_pricing(new_option, Q=Q)
        print("Option price:", option_price)

    # # # Question 6
    # model_opt = ["P1200", "P1300", "P1400", "C1400", "C1450", "C1550", "C1600"]
    # securities = {k: float(k[1:]) for k in all_securities.keys() if k in model_opt}
    # other_securities = {k: v for k, v in all_securities.items() if k not in model_opt}
    #
    # St0 = np.array([51.55, 77.7, 116.55, 71.1, 47.25, 15.8, 7.9])
    # states = [1250, 1350, 1425, 1500, 1575]
    # f_state, l_state = 1000, 1700
    # states = [f_state] + states + [l_state]
    # opm = OnePeriodMarketModel(init_price_vec=St0, states=states, option_info=securities)
    # Q = opm.option_pricing_model()
    # print("Q:", Q)
    # new_option = {"C1300": 1300}
    # option_price = opm.option_pricing(new_option, Q=Q)
    # print("Option price:", option_price)
    #
    # rmse = opm.compute_RMS(other_securities)
    # print("The RMSE of the model is: {:.2f}%".format(rmse * 100))