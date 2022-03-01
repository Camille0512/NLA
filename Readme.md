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

    Offers the following function:

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

    Offers the following function:

    - Check complete market
    - Check arbitrage free
    - Generate one period pricing model (parameters)
    - Option pricing
    - Error computation (average absolute value / root mean squared error)
    - Graph of the errors for all the securities in the market

- **Eigens.py**

  Functions for the eigenvector and eigenvalue computation under special cases using theories.

- **MatrixVerifications.py**

  Mainly contains functions for matrix checking during the previous functions.

- **run_script.py**

  Offer some samples for the implementation of the above functions.



### TODO

Write test script