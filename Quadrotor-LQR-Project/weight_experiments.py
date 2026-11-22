
"""
weight_experiments.py

How do the Q and R weights change the controller's behavior? Here we try a few
different weight settings, hit each version with the same wind gust, and
compare how far the drone drifts, how long it takes to recover, and how much
motor effort it uses.

Saves results/weight_comparison.png and results/weight_metrics.csv.
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from drone_model import state_space
from lqr_controller import calculate_lqr
from wind_experiments import simulate_wind, calculate_metrics


def main():

    # Linear drone model
    A, B = state_space()

    # Our starting (baseline) weights, the same as in main.py
    Q_base = np.diag([30, 30, 40, 8, 8, 2])
    R_base = np.diag([1, 1])

    # Each setting changes what the controller cares about most:
    #   Higher state penalty   -> fix position errors more aggressively
    #   Higher control penalty -> save motor effort and accept more drift
    #   Lower control penalty  -> let the motors work harder
    configurations = {
        "Baseline": (Q_base, R_base),
        "Higher state penalty": (3 * Q_base, R_base),
        "Higher control penalty": (Q_base, 3 * R_base),
        "Lower control penalty": (Q_base, 0.3 * R_base)
    }

    wind_strength = 2.0  # every setting faces the same gust (m/s^2) so the comparison is fair

    results = []  # one row of numbers for each setting

    # Save output files in results/ (create it if it's missing)
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    print("\n--- LQR Weighting Experiments ---")

    # Design, check, and simulate a controller for each weight setting
    for name, (Q, R) in configurations.items():

        # Get the gain matrix using our own CARE solver
        K, P = calculate_lqr(
            A, B, Q, R, method="custom"
        )

        # Plug P back into the Riccati equation to make sure the solution is accurate
        G = B @ np.linalg.solve(R, B.T)

        residual = np.linalg.norm(
            A.T @ P + P @ A - P @ G @ P + Q,
            ord="fro"
        )

        # Make sure this controller is stable (all eigenvalues have negative real parts)
        eigenvalues = np.linalg.eigvals(A - B @ K)
        stable = bool(np.all(np.real(eigenvalues) < 0))

        # Simulate this controller against the wind gust
        t, states = simulate_wind(
            A, B, K, wind_strength
        )

        # Measure peak drift, recovery time, and motor effort
        peak, recovery, effort = calculate_metrics(
            t, states, K
        )

        # Save this setting's numbers
        results.append([
            name,
            float(peak),
            float(recovery),
            float(effort),
            float(residual),
            stable
        ])

        ax.plot(t, states[0], label=name)

    # Shade the time window when the gust is blowing
    ax.axvspan(
        1.0, 1.3,
        color="gray",
        alpha=0.15,
        label="Wind gust"
    )

    ax.set(
        xlabel="Time (s)",
        ylabel="Horizontal position error (m)",
        title="LQR Weighting Comparison Under 2 m/s² Wind Gust"
    )

    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    fig.savefig(
        output_dir / "weight_comparison.png",
        dpi=180
    )

    # Print a results table with headers
    print(
        f"\n{'Configuration':<25}"
        f"{'Peak (m)':>12}"
        f"{'Recovery (s)':>15}"
        f"{'Effort':>14}"
        f"{'Stable':>10}"
    )

    # One line per setting, plus its CARE residual underneath
    for name, peak, recovery, effort, residual, stable in results:
        print(
            f"{name:<25}"
            f"{peak:>12.6f}"
            f"{recovery:>15.6f}"
            f"{effort:>14.6f}"
            f"{str(stable):>10}"
        )

        print(f"    CARE residual: {residual:.3e}")

    # Save the same numbers to a CSV file so we can use them in the report
    with open(
        output_dir / "weight_metrics.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Configuration",
            "Peak Horizontal Error (m)",
            "Recovery Time (s)",
            "Integrated Squared Control Effort",
            "CARE Residual",
            "Stable"
        ])

        writer.writerows(results)

    plt.show()


# Only run main() when this file is run directly
if __name__ == "__main__":
    main()
