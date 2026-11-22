
"""
wind_experiments.py

How well does the controller handle wind gusts of different strengths? We
keep the controller the same and just make the gust stronger, then measure
how far the drone drifts, how quickly it recovers, and how much motor effort
it uses.

The two helper functions here (simulate_wind and calculate_metrics) are also
used by weight_experiments.py.

Saves results/wind_comparison.png and results/wind_metrics.csv.
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from drone_model import state_space, wind_acceleration
from lqr_controller import calculate_lqr


def simulate_wind(A, B, K, strength):
    """
    Simulate 8 seconds of the drone with the LQR controller on while a wind gust hits it.

    Inputs:
        A, B     - the drone model matrices
        K        - the LQR gain matrix
        strength - how hard the gust pushes sideways (m/s^2)

    Returns the time points t and the states at those times (6 rows, one per state).
    """

    def dynamics(t, state):
        """How the drone's state changes at time t, with the controller on and the wind blowing."""
        # Controller output based on how far off the drone is
        u = -K @ state

        # Linear drone model: dx/dt = Ax + Bu
        derivative = A @ state + B @ u

        # Add the wind's sideways push (only between 1.0 s and 1.3 s)
        derivative[3] += wind_acceleration(
            t,
            gust_start=1.0,
            gust_end=1.3,
            strength=strength
        )

        return derivative

    # The gust switches on and off suddenly, so we solve before, during, and
    # after it separately. Every chunk saves a point every 0.01 seconds
    time_parts = [
        np.linspace(0, 1, 101),
        np.linspace(1, 1.3, 31),
        np.linspace(1.3, 8, 671)
    ]

    initial = np.zeros(6)  # start right at hover
    segments = []  # results from each chunk

    # Solve one chunk at a time
    for i, t_eval in enumerate(time_parts):

        sol = solve_ivp(
            dynamics,
            (float(t_eval[0]), float(t_eval[-1])),
            initial,
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-11,
            max_step=0.01
        )

        # Stop if the solver ran into trouble
        if not sol.success:
            raise RuntimeError(sol.message)

        # Drop the last point of the first two chunks since the next chunk starts there too
        segments.append(
            sol.y[:, :-1] if i < 2 else sol.y
        )

        initial = sol.y[:, -1]  # the next chunk picks up from here

    # Stitch the time points together the same way, skipping repeats
    t = np.concatenate([
        p[:-1] if i < 2 else p
        for i, p in enumerate(time_parts)
    ])

    states = np.hstack(segments)  # combine all chunks into one array

    return t, states


def calculate_metrics(t, states, K):
    """
    Work out three numbers that summarize how well the drone recovered.

    Returns:
        peak_error    - the furthest the drone drifted sideways (m)
        recovery_time - seconds after the gust ends until the drone is back
                        within 1 cm and stays there (NaN if it never gets there)
        effort        - total motor effort, i.e. the area under (thrust^2 + torque^2)
    """

    horizontal_error = states[0]  # row 0 is the sideways position x

    # Furthest the drone got from where it started
    peak_error = np.max(np.abs(horizontal_error))

    # We call the drone "recovered" once it's within 1 cm of where it started
    tolerance = 0.01

    gust_end = 1.3  # time the gust stops (s)

    recovery_time = np.nan  # stays NaN if the drone never recovers

    # Look at each time step after the gust ends and find the first one
    # where the drone is within 1 cm and stays there for the rest of the run
    for i in range(len(t)):

        # Skip anything before the gust ends
        if t[i] < gust_end:
            continue

        if np.all(np.abs(horizontal_error[i:]) <= tolerance):
            recovery_time = t[i] - gust_end
            break

    # Work out the motor commands at every time step
    control = -K @ states

    # Total effort: square both inputs, add them up, then integrate over time
    effort = np.trapezoid(
        np.sum(control ** 2, axis=0),
        t
    )

    return peak_error, recovery_time, effort


def main():

    # Linear drone model
    A, B = state_space()

    # Same weights as main.py
    Q = np.diag([30, 30, 40, 8, 8, 2])
    R = np.diag([1, 1])

    # Get the gain matrix from our own CARE solver
    K, P = calculate_lqr(
        A, B, Q, R, method="custom"
    )

    # Gust strengths to test (m/s^2). 0 is a "no wind" baseline
    wind_strengths = [0, 1, 2, 4]

    results = []  # one row of numbers for each wind strength

    # Save output files in results/ (create it if it's missing)
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Simulate and measure each wind strength
    for strength in wind_strengths:

        t, states = simulate_wind(
            A, B, K, strength
        )

        peak, recovery, effort = calculate_metrics(
            t, states, K
        )

        results.append([
            strength,
            float(peak),
            float(recovery),
            float(effort)
        ])

        plt.plot(
            t,
            states[0],
            label=f"Wind = {strength} m/s²"
        )

    # Shade the time window when the gust is blowing
    plt.axvspan(
        1.0, 1.3,
        color="gray",
        alpha=0.15,
        label="Wind gust"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Horizontal position error (m)")
    plt.title("LQR Hover Recovery Under Different Wind Strengths")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_dir / "wind_comparison.png",
        dpi=180
    )

    print("\n--- Wind Disturbance Experiments ---")

    # Table header
    print(
        f"{'Wind':>8} {'Peak Error':>15} "
        f"{'Recovery (s)':>15} {'Effort':>15}"
    )

    # One row per wind strength
    for strength, peak, recovery, effort in results:
        print(
            f"{strength:8.1f} {peak:15.6f} "
            f"{recovery:15.6f} {effort:15.6f}"
        )

    # Save the results to a CSV file so we can use them in the report
    with open(
        output_dir / "wind_metrics.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Wind Acceleration (m/s^2)",
            "Peak Horizontal Error (m)",
            "Recovery Time (s)",
            "Integrated Squared Control Effort"
        ])

        writer.writerows(results)

    plt.show()


# Only run main() when this file is run directly
if __name__ == "__main__":
    main()
