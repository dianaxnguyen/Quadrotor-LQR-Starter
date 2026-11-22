
"""
lqr_controller.py

This is where we work out the LQR controller. The goal of LQR is to find a
gain matrix K so that the control law u = -Kx brings the drone back to hover
while balancing two things: how far off it is (weighted by Q) and how hard the
motors have to work (weighted by R).

To get K we first need to solve the Continuous-Time Algebraic Riccati
Equation (CARE) for a matrix P. You can pick which solver to use: SciPy's
built-in one, or the matrix sign function solver we wrote ourselves.
"""

import numpy as np
from scipy.linalg import solve_continuous_are

from riccati_solver import solve_care_matrix_sign


def calculate_lqr(A, B, Q, R, method="scipy"):
    """
    Work out the LQR gain matrix K and the Riccati solution P.

    Inputs:
        A, B   - the drone's state-space matrices (from drone_model.py)
        Q      - how much we care about each state being off (bigger = fix it faster)
        R      - how much we care about using the motors (bigger = go easier on them)
        method - which CARE solver to use:
                 "scipy"  uses SciPy's solver, which we treat as the reference answer
                 "custom" uses our own matrix sign function solver

    Returns K (the gain matrix) and P (the Riccati solution).
    """

    # Option 1: let SciPy solve the Riccati equation
    if method == "scipy":
        P = solve_continuous_are(A, B, Q, R)

    # Option 2: use our own solver, which also gives back K directly
    elif method == "custom":
        P, K, residual, iterations = (
            solve_care_matrix_sign(A, B, Q, R)
        )

        # Print some info so we can see how the solver did
        print("\nCustom CARE solver:")
        print("Iterations:", iterations)
        print("CARE residual:", residual)

        return K, P

    # Anything else is a typo, so let the user know
    else:
        raise ValueError(
            "Method must be 'scipy' or 'custom'."
        )

    # For the SciPy route we still need K. The formula is K = R^-1 B^T P,
    # and we use solve() so we never have to invert R ourselves
    K = np.linalg.solve(R, B.T @ P)

    return K, P
