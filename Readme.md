# Readme

### Introduction

NLA is a project designed to implement the **numerical linear algebra** techniques taught by Baruch Pre-MFE program. It is organized according to the course sequence.

- LU decomposition
- Computation of discount factors by solving linear system
- Multiple period model by using cubic spline interpolationlation
- One price model implementation in option pricing
- Eigenvalue and Eigenvector
- Symmetric matrix

The techniques implemented is only a <font color="red">implementation practice of NLA</font> in financial field. Only used for study purpose.



### How to use

Unblock the `run_script.py` according to the `Question *` to test different functions.



### Files

- **Decomposition.py**

  Including the following contents:

  - `Class LU`

    Mainly related functions based on or using *LU decomposition*.

    Offers the following functions:

    - Forward substitution

    - Backward substitution

    - LU decomposition (without / with row pivoting)

    - Computation of discount factors by solving linear system

    - Solving a multiple linear system shared the same matrix

  - `Class EquationSimulation`

    Inherit from `class LU`, mainly implement the cubic spline interpolation model.

    Offers the following function:

    - Simulation of the bond pricing equation

  - `Class OnePeriodMarketModel`

    Inherit from `class LU`, mainly implement the *Arrow-Debreu one period market model* for option pricing.

    Offers the following functions:

    - Check complete market
    - Check arbitrage free
    - Generate one period pricing model (parameters)
    - Option pricing
    - Error computation (average absolute value / root mean squared error)
    - Graph of the errors for all the securities in the market

- **OLRRegression.py**

  - `Class OLR`

    Inherit from the `class Cholesky`, mainly responsible for the *Least squares* computation by using NLA techniques.

    Offers the following functions:

    - OLR equation estimates
    - Compute errors
    - A special function provided according to the paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2908494
    - Implied volatility computation according to the BSM

  - `Class PortfolioOptimize`

    Inherit from the `class Cholesky`, mainly responsible for portfolio optimization based on mean-variance theory.

    Offers the following functions:

    - Tangency portfolio weighting computation
    - Min variance weighting computation
    - Min variance portfolio standard variance computation
    - Max return weighting computation
    - Max return portfolio return computation
    - Min variance portfolio without cash position computation
    - Min variance portfolio without cash position standard variance computation

- **OtherFunctions.py**

  Functions that support for the lectures.

- **Verifications.py**

  Mainly contains functions for matrix checking during the previous functions.

- **run_script.py**

  Offer some samples for the implementation of the above functions.



### TODO

Write test script