import numpy as np
import pytest
import sys # noqa
import os # noqa
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import NLA.src.utils as utils
from NLA.src.Decomposition import compute_discount_factor
from NLA.src.Decomposition import trans_discount_factor_zero_rate
from NLA.src.Decomposition import Cholesky
from NLA.src.Decomposition import EquationSimulation
from NLA.src.Decomposition import LU
from NLA.src.Decomposition import OnePeriodMarketModel
from NLA.src.OtherFunctions import compute_root_quadratic_equation
from NLA.src.OtherFunctions import compute_tri_diagonal_symmetric
from NLA.src.OtherFunctions import trading_days_computation
from NLA.src.OLRRegression import OLR
from NLA.src.OLRRegression import PortfolioOptimize

np.set_printoptions(suppress=True)

def test_forward_substitution():
    lu = LU()

    L = [
        [100, 0, 0, 0],
        [2, 102, 0, 0],
        [3, 3, 103, 0],
        [2, 2, 2, 102]
    ]
    b = [98.75, 102, 103.5, 105.5]
    x = utils.matrix_formatter(lu.forward_substitution(L, b), 1)
    assert x == [0.9875, 0.9806, 0.9475, 0.9771]

    L = [
        [100, 0, 0, 0],
        [6, 106, 0, 0],
        [8, 8, 108, 0],
        [5, 5, 5, 105]
    ]
    b = [98, 104, 111, 102]
    x = utils.matrix_formatter(lu.forward_substitution(L, b), 1)
    assert x == [0.98, 0.9257, 0.8866, 0.8385]


if __name__ == "__main__":
    test_forward_substitution()