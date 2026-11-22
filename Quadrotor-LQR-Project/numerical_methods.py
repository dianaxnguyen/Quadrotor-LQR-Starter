
"""
numerical_methods.py

Our own versions of two classic numerical linear algebra tools:
    householder_qr  - splits a matrix into Q (orthogonal) times R (upper triangular)
    qr_eigenvalues  - finds eigenvalues by running QR decompositions over and over
Each one has a test function that checks our answer against a known result or NumPy.
"""

import numpy as np


def householder_qr(A):
    """
    Compute the QR decomposition A = Q @ R using Householder reflections.

    Q is orthogonal (its columns are perpendicular unit vectors) and R is
    upper triangular (everything below the diagonal is zero).

    Each step uses a "reflection" to wipe out the entries below the
    diagonal in one column, moving left to right until R is triangular.
    """

    A = np.array(A, dtype=float)  # work in floats so the division behaves
    m, n = A.shape  # m rows, n columns

    Q = np.eye(m)  # start Q as the identity and build it up as we go
    R = A.copy()  # R starts as A and gets turned into a triangle

    # Clean up one column at a time
    for k in range(min(m, n)):

        # Grab the part of column k on and below the diagonal
        x = R[k:, k].copy()

        # Length of that piece of the column
        norm_x = np.linalg.norm(x)

        # If it's already all zeros there's nothing to do
        if norm_x == 0:
            continue

        # Build the reflection vector v. We pick the sign that matches x[0]
        # to avoid subtracting two nearly equal numbers (which loses accuracy)
        sign = 1.0 if x[0] >= 0 else -1.0

        v = x.copy()
        v[0] += sign * norm_x

        v_norm = np.linalg.norm(v)

        # Safety check so we never divide by zero
        if v_norm == 0:
            continue

        v = v / v_norm  # make v a unit vector

        # Apply the reflection to R, which zeros out column k below the diagonal
        R[k:, k:] -= 2 * np.outer(
            v, v @ R[k:, k:]
        )

        # Apply the same reflection to Q so that Q @ R still equals A
        Q[:, k:] -= 2 * np.outer(
            Q[:, k:] @ v, v
        )

    return Q, R


def test_householder_qr():
    """Check householder_qr on a small 3x3 example (a common textbook matrix)."""

    A = np.array([
        [12, -51, 4],
        [6, 167, -68],
        [-4, 24, -41]
    ], dtype=float)

    Q, R = householder_qr(A)

    print("Original matrix A:")
    print(A)

    print("\nOrthogonal matrix Q:")
    print(np.round(Q, 4))

    print("\nUpper triangular matrix R:")
    print(np.round(R, 4))

    # Multiply Q and R back together. It should match A
    reconstruction_error = np.linalg.norm(
        A - Q @ R, ord="fro"
    )

    # Q^T Q should be the identity if Q is truly orthogonal
    orthogonality_error = np.linalg.norm(
        Q.T @ Q - np.eye(Q.shape[0]),
        ord="fro"
    )

    # Both of these should come out close to zero
    print("\nReconstruction error:", reconstruction_error)
    print("Orthogonality error:", orthogonality_error)

def qr_eigenvalues(A, max_iterations=2000, tolerance=1e-10):
    """
    Estimate eigenvalues with the shifted QR algorithm.

    We repeatedly factor the matrix into Q and R, then multiply them back
    in the opposite order (R @ Q). Each round makes the matrix a little more
    diagonal, and once it's diagonal the eigenvalues are just the diagonal
    entries. The "shift" makes this converge a lot faster.

    Note: this version only works for real symmetric matrices.

    Inputs:
        A              - a square, symmetric matrix
        max_iterations - most rounds we'll try before stopping
        tolerance      - we stop once everything below the diagonal is smaller than this

    Returns the eigenvalues, the number of rounds used, and the final off-diagonal size.
    """

    A = np.array(A, dtype=float)

    # Make sure the matrix is square
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")

    # Make sure it's symmetric, since that's all this version can handle
    if not np.allclose(A, A.T):
        raise ValueError("This implementation requires a symmetric matrix.")

    n = A.shape[0]  # size of the matrix
    Ak = A.copy()  # the matrix we keep updating each round

    # Keep doing QR steps until the matrix is basically diagonal
    for iteration in range(max_iterations):

        # Add up how big everything below the diagonal still is
        off_diagonal = np.linalg.norm(
            np.tril(Ak, -1), ord="fro"
        )

        # Close enough to diagonal, so read off the eigenvalues
        if off_diagonal < tolerance:
            return np.diag(Ak), iteration, off_diagonal

        # Use the bottom-right entry as the shift to speed things up
        shift = Ak[-1, -1]

        # Factor the shifted matrix with our own QR function
        Q, R = householder_qr(
            Ak - shift * np.eye(n)
        )

        # Multiply back in reverse order and undo the shift
        Ak = R @ Q + shift * np.eye(n)

        # Rounding can make Ak slightly unsymmetric, so we fix that here
        Ak = 0.5 * (Ak + Ak.T)

    # Ran out of rounds, so return the best estimate we have
    return np.diag(Ak), max_iterations, off_diagonal


def test_qr_eigenvalues():
    """Compare our QR eigenvalues with NumPy's on a small symmetric matrix."""

    A = np.array([
        [4, 1, 1],
        [1, 3, 0],
        [1, 0, 2]
    ], dtype=float)

    eigenvalues, iterations, error = qr_eigenvalues(A)

    # NumPy's answer, used as the reference
    reference = np.linalg.eigvalsh(A)

    print("\n--- QR Eigenvalue Test ---")

    print("Computed eigenvalues:")
    print(np.sort(eigenvalues))

    print("\nNumPy eigenvalues:")
    print(reference)

    print("\nIterations:", iterations)
    print("Final off-diagonal norm:", error)

    # Sort both lists so they line up, then find the biggest gap
    print(
        "Maximum eigenvalue error:",
        np.max(
            np.abs(np.sort(eigenvalues) - reference)
        )
    )


# Running this file directly runs both tests
if __name__ == "__main__":
    test_householder_qr()
    test_qr_eigenvalues()
