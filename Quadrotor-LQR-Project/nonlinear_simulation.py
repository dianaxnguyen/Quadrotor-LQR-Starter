
"""
nonlinear_simulation.py

Our LQR controller was designed with the linear (simplified) drone model, but
a real drone is nonlinear. For example, the sideways push from tilting really
goes like sin(theta), not just theta. This script runs the same controller on
both models under three wind gust strengths to see how far apart they get.

It saves results/linear_vs_nonlinear.png, which has one subplot per wind strength.
"""

from pathlib import Path

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
    # Linear model of the drone near hover. We only use it to design the controller
    A, B = state_space()

    # Same LQR weights we use everywhere else in the project
    Q = np.diag([30, 30, 40, 8, 8, 2])
    R = np.diag([1, 1])

    # Get the gain matrix K from our own matrix sign CARE solver
    K, P = calculate_lqr(
        A, B, Q, R, method="custom"
    )

    # The drone starts right at hover, so every error is zero.
    # State order: [x, z, theta, vx, vz, omega]
    initial_state = np.zeros(6)

    # Gust strengths to try, going from mild to strong (m/s^2)
    wind_strengths = [2.0, 4.0, 8.0]

    # Save the plot in a results/ folder (create it if it's missing)
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)

    # One subplot for each wind strength, stacked on top of each other
    fig, axes = plt.subplots(
        len(wind_strengths), 1,
        figsize=(10, 9),
        sharex=True
    )

    # Run both models for each wind strength and plot them together
    for ax, strength in zip(axes, wind_strengths):

        def linear_dynamics(t, state):
            """The simplified linear model: dx/dt = Ax + Bu + wind."""
            u = -K @ state  # LQR feedback

            derivative = A @ state + B @ u
            # Add the wind's sideways push to the horizontal acceleration
            derivative[3] += wind_acceleration(
                t, strength=strength
            )

            return derivative

        def nonlinear_dynamics(t, state):
            """The full physics model, using sin and cos of the pitch angle."""
            x, z, theta, vx, vz, omega = state  # unpack the state so the equations are easier to read

            # The controller is the same one designed on the linear model
            delta_thrust, torque = -K @ state

            # The controller only gives the extra thrust, so we add back
            # the thrust needed to hold the drone up (m * g)
            thrust = MASS * GRAVITY + delta_thrust

            # Positions change by their velocities
            dx = vx
            dz = vz
            dtheta = omega

            # Tilting sends part of the thrust sideways, and the wind pushes too
            dvx = (
                (thrust / MASS) * np.sin(theta)
                + wind_acceleration(t, strength=strength)
            )

            # The upward part of the thrust fights against gravity
            dvz = (
                (thrust / MASS) * np.cos(theta)
                - GRAVITY
            )

            # Torque changes how fast the drone tilts
            domega = torque / INERTIA

            return np.array([
                dx, dz, dtheta,
                dvx, dvz, domega
            ])

        # The gust turns on and off suddenly, so we solve before, during,
        # and after the gust separately to keep the solver accurate
        intervals = [
            (0.0, 1.0),
            (1.0, 1.3),
            (1.3, 8.0)
        ]

        def run(dynamics):
            """Simulate 8 seconds using the given model and return times and states."""
            state = initial_state.copy()
            times = []  # time points from each chunk
            states = []  # states from each chunk

            # Solve one time chunk at a time
            for i, (start, end) in enumerate(intervals):
                # Save a data point every 0.01 seconds
                t_eval = np.linspace(
                    start, end,
                    int(round((end - start) / 0.01)) + 1
                )

                solution = solve_ivp(
                    dynamics,
                    (start, end),
                    state,
                    t_eval=t_eval,
                    rtol=1e-9,
                    atol=1e-11,
                    max_step=0.01
                )

                # Stop if the solver ran into trouble
                if not solution.success:
                    raise RuntimeError(solution.message)

                # Each chunk starts where the last one ended, so skip
                # the first point after chunk 1 to avoid a duplicate
                if i > 0:
                    times.append(solution.t[1:])
                    states.append(solution.y[:, 1:])
                else:
                    times.append(solution.t)
                    states.append(solution.y)

                state = solution.y[:, -1]  # the next chunk starts from here

            return np.concatenate(times), np.hstack(states)

        # Simulate both models
        t, linear = run(linear_dynamics)
        _, nonlinear = run(nonlinear_dynamics)

        # Biggest sideways drift each model sees
        linear_peak = np.max(np.abs(linear[0]))
        nonlinear_peak = np.max(np.abs(nonlinear[0]))

        # Print a quick summary for this wind strength
        print(f"\nWind strength: {strength} m/s²")
        print(f"Linear peak error: {linear_peak:.6f} m")
        print(f"Nonlinear peak error: {nonlinear_peak:.6f} m")
        print(
            "Peak error difference:",
            f"{abs(nonlinear_peak - linear_peak):.6f} m"
        )
        # If the drone tilts a lot, the linear model starts to be less accurate
        print(
            "Maximum absolute pitch (nonlinear):",
            f"{np.max(np.abs(np.rad2deg(nonlinear[2]))):.4f} degrees"
        )

        # Plot both models on the same subplot
        ax.plot(t, linear[0], label="Linear model")
        ax.plot(
            t, nonlinear[0],
            linestyle="--",
            label="Nonlinear model"
        )

        # Shade the time window when the gust is blowing
        ax.axvspan(1.0, 1.3, color="gray", alpha=0.15)

        ax.set_ylabel("Position error (m)")
        ax.set_title(f"Wind acceleration: {strength} m/s²")
        ax.grid(True)
        ax.legend()

    # Only the bottom subplot needs a time label
    axes[-1].set_xlabel("Time (s)")

    fig.suptitle(
        "Linear vs Nonlinear Quadrotor Hover Recovery"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "linear_vs_nonlinear.png",
        dpi=180
    )

    plt.show()


# Only run main() when this file is run directly
if __name__ == "__main__":
    main()
