
"""
test_drone_care.py

A check on our own CARE solver using the actual drone model. We solve the same
problem with our matrix sign solver and with SciPy, then compare the two. If
our solver is right, the P and K matrices should match SciPy's almost exactly,
and the controlled drone should come out stable.
"""

import numpy as np
from scipy.linalg import solve_continuous_are

# The drone's A and B matrices
from drone_model import state_space

# Our own matrix sign CARE solver
from riccati_solver import solve_care_matrix_sign


def compare_care_solvers(A, B, Q, R):
    """
    Solve the drone's CARE with our solver and with SciPy, then print how close they are.

    Inputs: A, B (drone model) and Q, R (LQR weights).
    Returns the gain matrix K and Riccati solution P from our solver.
    """

    # Step 1: solve it with our own solver
    P_custom, K_custom, residual, iterations = (
        solve_care_matrix_sign(A, B, Q, R)
    )

    # Step 2: solve it again with SciPy so we have a reference answer
    P_scipy = solve_continuous_are(A, B, Q, R)

    # SciPy only gives P, so we work out K = R^-1 B^T P ourselves
    K_scipy = np.linalg.solve(
        R, B.T @ P_scipy
    )

    # Step 3: print both answers side by side
    print("\n--- Drone CARE Comparison ---")

    print("\nCustom gain matrix K:")
    print(np.round(K_custom, 6))

    print("\nSciPy gain matrix K:")
    print(np.round(K_scipy, 6))

    print("\nCARE residual:", residual)
    print("Matrix sign iterations:", iterations)

    # These differences should be very close to zero
    print(
        "\nP difference from SciPy:",
        np.linalg.norm(P_custom - P_scipy, ord="fro")
    )

    print(
        "K difference from SciPy:",
        np.linalg.norm(K_custom - K_scipy, ord="fro")
    )

    # Step 4: make sure the controller actually stabilizes the drone
    closed_loop = A - B @ K_custom  # how the drone behaves with the controller on

    eigenvalues = np.linalg.eigvals(closed_loop)

    print("\nClosed-loop eigenvalues:")
    print(eigenvalues)

    # Stable means every eigenvalue has a negative real part
    stable = np.all(np.real(eigenvalues) < 0)

    print("\nStable:", stable)

    return K_custom, P_custom


# Only runs when this file is run directly
if __name__ == "__main__":

    # Step 1: get the drone's A and B matrices
    A, B = state_space()

    # Step 2: pick the LQR weights. A bigger number means we care more
    # about keeping that state near zero.
    # State order: [x, z, theta, vx, vz, omega]
    Q = np.diag([
        30.0,  # horizontal position
        30.0,  # altitude
        50.0,  # pitch angle
        5.0,   # horizontal velocity
        5.0,   # vertical velocity
        1.0    # pitch rate
    ])

    # A bigger number here means we'd rather go easy on that motor input.
    # Control order: [extra thrust, pitch torque]
    R = np.diag([
        1.0,   # thrust penalty
        1.0    # torque penalty
    ])

    # Step 3: run the comparison
    compare_care_solvers(A, B, Q, R)
