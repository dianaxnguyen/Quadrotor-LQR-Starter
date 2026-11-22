
"""
riccati_solver.py

Our own solver for the Continuous-Time Algebraic Riccati Equation (CARE),
built on the matrix sign function. We wrote this ourselves instead of only
relying on SciPy so we could see how the numerical method works. SciPy is
still used in the tests, but only to double-check our answers.

What's in here:
    matrix_sign            - Newton's method for the matrix sign function
    test_matrix_sign       - quick check on a matrix where we already know the answer
    solve_care_matrix_sign - solves the CARE using the Hamiltonian matrix
    test_care_solver       - compares our CARE solver with SciPy on a small system
"""

import numpy as np


def matrix_sign(A, tolerance=1e-10, max_iterations=100):
    """
    Compute the matrix sign function with Newton's method.

    The idea is to keep averaging the matrix with its inverse:
        S_next = 0.5 * (S + inverse(S))
    Eigenvalues with a positive real part get pulled toward +1, and ones with
    a negative real part get pulled toward -1.

    Inputs:
        A              - a square matrix with no eigenvalues sitting on the imaginary axis
        tolerance      - we stop once S changes by less than this between steps
        max_iterations - give up if it still hasn't settled after this many steps

    Returns the sign matrix S, how many iterations it took, and the final change.
    """

    A = np.array(A, dtype=float)  # make sure we're working with floats

    # The sign function only makes sense for square matrices
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")

    S = A.copy()  # starting guess is just A itself
    identity = np.eye(A.shape[0])  # identity matrix, used to get the inverse

    # Keep repeating the Newton step until S stops changing
    for iteration in range(1, max_iterations + 1):

        # Get the inverse of S by solving S * X = I. This is more accurate
        # than calling an inverse function directly
        try:
            inverse_S = np.linalg.solve(S, identity)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Matrix became singular during iteration."
            )

        # The Newton step: average S with its inverse
        S_next = 0.5 * (S + inverse_S)

        # See how much S changed, relative to its size
        error = np.linalg.norm(
            S_next - S, ord="fro"
        ) / max(1.0, np.linalg.norm(S_next, ord="fro"))

        S = S_next

        # If the change is tiny, we've converged
        if error < tolerance:
            return S, iteration, error

    # If we get here, the loop ran out of tries without converging
    raise RuntimeError(
        "Matrix sign iteration did not converge."
    )


def test_matrix_sign():
    """Check matrix_sign on a diagonal matrix, where we already know what the answer should be."""

    # Diagonal matrix with two negative and two positive entries
    A = np.diag([-3.0, -1.0, 2.0, 4.0])

    S, iterations, error = matrix_sign(A)

    # Negatives should turn into -1 and positives into +1
    expected = np.diag([-1.0, -1.0, 1.0, 1.0])

    print("--- Matrix Sign Function Test ---")

    print("\nComputed matrix sign:")
    print(np.round(S, 6))

    print("\nExpected matrix sign:")
    print(expected)

    print("\nIterations:", iterations)
    print("Final relative change:", error)

    # How far our answer is from the known answer
    print(
        "Reference error:",
        np.linalg.norm(S - expected, ord="fro")
    )

    # A sign matrix squared should give the identity, so this should be close to zero
    print(
        "Sign identity error:",
        np.linalg.norm(
            S @ S - np.eye(A.shape[0]),
            ord="fro"
        )
    )


# SciPy's CARE solver, only used in the test below to check our answer
from scipy.linalg import solve_continuous_are


def solve_care_matrix_sign(A, B, Q, R):
    """
    Solve the CARE using the matrix sign function on the Hamiltonian matrix.

    The equation we're solving for P is:
        A.T @ P + P @ A - P @ B @ inv(R) @ B.T @ P + Q = 0

    Rough idea: we stack A, B, Q and R into one big "Hamiltonian" matrix.
    Its sign function lets us pick out the stable half of that matrix, and
    P can be read straight off the stable half.

    Inputs: A, B (drone model) and Q, R (LQR weights).
    Returns P, the gain matrix K, the CARE residual (how close the equation is
    to zero, so smaller is better), and the number of sign iterations used.
    """

    n = A.shape[0]  # number of states (6 for the drone)

    # G = B R^-1 B^T shows up in both the Hamiltonian and the residual check
    G = B @ np.linalg.solve(R, B.T)

    # Build the 2n x 2n Hamiltonian matrix out of four blocks
    H = np.block([
        [A, -G],
        [-Q, -A.T]
    ])

    # Run Newton's method to get sign(H)
    S, iterations, error = matrix_sign(
        H,
        tolerance=1e-10,
        max_iterations=200
    )

    # Split sign(H) into its four n x n blocks
    S11 = S[:n, :n]
    S12 = S[:n, n:]
    S21 = S[n:, :n]
    S22 = S[n:, n:]

    # (I - S) / 2 keeps only the stable part of H (eigenvalues with negative real part)
    projector = 0.5 * (np.eye(2 * n) - S)

    # Use the SVD to get a clean set of basis vectors for that stable part
    U, singular_values, _ = np.linalg.svd(projector)

    # The first n columns of U span the stable part
    U_stable = U[:, :n]

    # Split that basis into its top half and bottom half
    U1 = U_stable[:n, :]
    U2 = U_stable[n:, :]

    # The Riccati solution is P = U2 * U1^-1
    P = np.linalg.solve(U1.T, U2.T).T

    # P should be symmetric. Rounding errors can make it slightly off,
    # so we average it with its transpose to fix that
    P = 0.5 * (P + P.T)

    # Gain matrix for the controller: K = R^-1 B^T P
    K = np.linalg.solve(R, B.T @ P)

    # Plug P back into the CARE to see how close to zero we get
    residual_matrix = (
        A.T @ P + P @ A
        - P @ G @ P + Q
    )

    # Turn the leftover matrix into one number (Frobenius norm)
    residual = np.linalg.norm(
        residual_matrix, ord="fro"
    )

    return P, K, residual, iterations


def test_care_solver():
    """Compare our CARE solver against SciPy's on a small 2-state example."""

    # A simple system: position and velocity, where we control acceleration
    A = np.array([
        [0.0, 1.0],
        [0.0, 0.0]
    ])

    B = np.array([
        [0.0],
        [1.0]
    ])

    Q = np.diag([10.0, 1.0])  # care more about position than velocity
    R = np.array([[1.0]])  # cost of using the control

    # Solve it with our own method
    P_custom, K_custom, residual, iterations = (
        solve_care_matrix_sign(A, B, Q, R)
    )

    # Solve it with SciPy so we have something to compare against
    P_reference = solve_continuous_are(A, B, Q, R)

    K_reference = np.linalg.solve(
        R, B.T @ P_reference
    )

    print("\n--- CARE Solver Test ---")

    print("\nCustom K:")
    print(np.round(K_custom, 6))

    print("\nSciPy K:")
    print(np.round(K_reference, 6))

    print("\nCARE residual:", residual)

    print("Matrix sign iterations:", iterations)

    # This should be close to zero if both solvers agree
    print(
        "Difference from SciPy:",
        np.linalg.norm(
            P_custom - P_reference, ord="fro"
        )
    )

    # All of these should have negative real parts (stable)
    print(
        "Closed-loop eigenvalues:",
        np.linalg.eigvals(A - B @ K_custom)
    )

# Running this file directly runs both tests
if __name__ == "__main__":
    test_matrix_sign()
    test_care_solver()
