
"""
convergence_experiments.py

How accurate is our numerical integration, and what does that accuracy cost
us? We simulate the nonlinear drone with the RK45 solver at a few different
tolerance settings and compare each run to a very high-accuracy "reference"
run. Tighter tolerances should give smaller errors, but they take more
function evaluations and more time.

Saves results/convergence_comparison.png and results/convergence_metrics.csv.
"""

from pathlib import Path
from time import perf_counter
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from drone_model import (
    state_space,
    wind_acceleration,
    MASS,
    INERTIA,
    GRAVITY
)
from lqr_controller import calculate_lqr


def main():

    # Build the controller the same way as in the other experiments
    A, B = state_space()

    Q = np.diag([30, 30, 40, 8, 8, 2])
    R = np.diag([1, 1])

    K, P = calculate_lqr(
        A, B, Q, R, method="custom"
    )

    # We use the nonlinear model here since it's the harder one for the solver.
    # The gust strength is fixed at 4 m/s^2 for every run
    wind_strength = 4.0

    def dynamics(t, state, wind_on):
        """Nonlinear drone physics. wind_on says whether the gust is blowing right now."""
        x, z, theta, vx, vz, omega = state  # unpack the state so the equations are easier to read

        # Controller output, plus the thrust needed to hold the drone up
        delta_thrust, torque = -K @ state
        thrust = MASS * GRAVITY + delta_thrust

        # Positions change by their velocities
        dx = vx
        dz = vz
        dtheta = omega

        # Thrust split into sideways and upward parts, and torque causes tilting
        dvx = (thrust / MASS) * np.sin(theta)
        dvz = (thrust / MASS) * np.cos(theta) - GRAVITY
        domega = torque / INERTIA

        # Add the wind's push only while the gust is blowing
        if wind_on:
            dvx += wind_strength

        return np.array([
            dx, dz, dtheta,
            dvx, dvz, domega
        ])

    # Three time chunks: before, during, and after the gust.
    # The third value says whether the wind is on in that chunk
    intervals = [
        (0.0, 1.0, False),
        (1.0, 1.3, True),
        (1.3, 8.0, False)
    ]

    # Every run saves its results at these same times so we can compare them point by point
    evaluation_times = np.linspace(0, 8, 801)

    def simulate(rtol, atol):
        """
        Run the full 8 second simulation at the given solver tolerances.

        rtol - relative tolerance (allowed error as a fraction of the value)
        atol - absolute tolerance (allowed error near zero)

        Returns the times, the states, the total number of function
        evaluations (how much work the solver did), and the run time in seconds.
        """

        state = np.zeros(6)  # start right at hover

        all_times = []  # time points from each chunk
        all_states = []  # states from each chunk

        total_evaluations = 0  # running count of how many times the solver called dynamics()

        start_clock = perf_counter()  # start the timer

        # Solve one chunk at a time
        for index, (start, end, wind_on) in enumerate(intervals):

            # Pick out the save times that fall inside this chunk.
            # The tiny 1e-12 margin is there so rounding doesn't drop the endpoints
            t_eval = evaluation_times[
                (evaluation_times >= start - 1e-12)
                & (evaluation_times <= end + 1e-12)
            ]

            solution = solve_ivp(
                lambda t, y: dynamics(t, y, wind_on),
                (start, end),
                state,
                method="RK45",
                t_eval=t_eval,
                rtol=rtol,
                atol=atol
            )

            # Stop if the solver ran into trouble
            if not solution.success:
                raise RuntimeError(solution.message)

            total_evaluations += solution.nfev

            # Skip the first point after chunk 1 since it repeats the last chunk's end
            if index > 0:
                all_times.append(solution.t[1:])
                all_states.append(solution.y[:, 1:])
            else:
                all_times.append(solution.t)
                all_states.append(solution.y)

            state = solution.y[:, -1]  # the next chunk starts from here

        elapsed = perf_counter() - start_clock  # how long the whole run took

        times = np.concatenate(all_times)
        states = np.hstack(all_states)

        return times, states, total_evaluations, elapsed

    # Run once with extremely tight tolerances. We treat this as the "true" answer
    t_ref, states_ref, _, _ = simulate(
        rtol=1e-12,
        atol=1e-14
    )

    # Tolerance pairs to test, going from loose to tight: (rtol, atol)
    tolerances = [
        (1e-4, 1e-6),
        (1e-6, 1e-8),
        (1e-8, 1e-10),
        (1e-10, 1e-12)
    ]

    results = []  # one row of numbers for each tolerance setting

    # Save output files in results/ (create it if it's missing)
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    print("\n--- Numerical Convergence Study ---")

    # Run the simulation at each tolerance and compare it to the reference
    for rtol, atol in tolerances:

        t, states, nfev, elapsed = simulate(
            rtol, atol
        )

        # Biggest gap from the reference, across every state and every time
        max_error = np.max(
            np.abs(states - states_ref)
        )

        # Same thing, but only for the sideways position
        horizontal_error = np.max(
            np.abs(states[0] - states_ref[0])
        )

        results.append([
            rtol,
            atol,
            float(max_error),
            float(horizontal_error),
            nfev,
            elapsed
        ])

        # Plot how far this run drifts from the reference over time
        ax.plot(
            t,
            states[0] - states_ref[0],
            label=f"rtol = {rtol:.0e}"
        )

    ax.set(
        xlabel="Time (s)",
        ylabel="Horizontal position difference (m)",
        title="RK45 Numerical Error Relative to Reference"
    )

    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    fig.savefig(
        output_dir / "convergence_comparison.png",
        dpi=180
    )

    # Table header
    print(
        f"\n{'RTOL':>12} {'Max Error':>14} "
        f"{'Position Error':>16} "
        f"{'Evaluations':>14} {'Time (s)':>12}"
    )

    # One row per tolerance setting
    for rtol, atol, error, position_error, nfev, elapsed in results:
        print(
            f"{rtol:12.0e} {error:14.3e} "
            f"{position_error:16.3e} "
            f"{nfev:14d} {elapsed:12.6f}"
        )

    # Save the results to a CSV file so we can use them in the report
    with open(
        output_dir / "convergence_metrics.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Relative Tolerance",
            "Absolute Tolerance",
            "Maximum State Difference",
            "Maximum Horizontal Position Difference (m)",
            "Function Evaluations",
            "Runtime (s)"
        ])

        writer.writerows(results)

    plt.show()


# Only run main() when this file is run directly
if __name__ == "__main__":
    main()
