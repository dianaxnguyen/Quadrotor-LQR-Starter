"""
main.py

This is the main demo for the project. It builds the drone model, computes the
LQR controller, hits the drone with a short wind gust, and then compares what
happens with the controller turned on versus off. At the end it saves two plots
into the results/ folder:
    position_error.png    - how far the drone drifts sideways, with and without LQR
    pitch_and_control.png - the drone's tilt and how hard the motors work over time
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from drone_model import state_space, wind_acceleration
from lqr_controller import calculate_lqr


def main():
    # Get the linearized drone model (A and B matrices)
    A, B = state_space()
    # Q says how much we care about each state being off target:
    # [x, z, pitch, vx, vz, pitch rate]. Pitch gets the most weight since tilting matters most
    Q = np.diag([30.0, 30.0, 40.0, 8.0, 8.0, 2.0])
    # R says how much we want to save motor effort: [thrust, torque]
    R = np.diag([1.0, 1.0])
    # Solve for the gain matrix K using our own CARE solver
    K, P = calculate_lqr(A, B, Q, R, method="custom")
    # Eigenvalues of A - BK tell us whether the controlled drone is stable
    eig = np.linalg.eigvals(A - B @ K)
    # Plug P back into the Riccati equation. If P is right this should be about zero
    residual = A.T @ P + P @ A - P @ B @ np.linalg.solve(R, B.T @ P) + Q
    # Print out the main results so we can check them
    print('State order: [x, z, pitch, vx, vz, pitch_rate]')
    print('Gain matrix K:\n', np.round(K, 4))
    print('Closed-loop eigenvalues:\n', np.round(eig, 4))
    print('CARE residual (Frobenius norm):', np.linalg.norm(residual))
    # The system is stable if every eigenvalue has a negative real part
    print('Stable:', bool(np.all(np.real(eig) < 0)))

    def dynamics(t, state, controlled):
        """How the drone's state changes at time t. If controlled is False, the motors do nothing extra."""
        # Use LQR feedback if the controller is on, otherwise no correction at all
        u = -K @ state if controlled else np.zeros(2)
        # Linear model: dx/dt = Ax + Bu
        derivative = A @ state + B @ u
        # The wind pushes sideways, so it gets added to the horizontal acceleration
        derivative[3] += wind_acceleration(t)
        return derivative

    # The wind switches on suddenly at t = 1 s and off at t = 1.3 s. The ODE solver
    # can stumble over sudden jumps like that, so we split time into three chunks
    # (before, during, and after the gust) and solve each chunk separately.
    # That way both runs see the exact same gust.
    time_parts = [np.linspace(0, 1, 101), np.linspace(1, 1.3, 31), np.linspace(1.3, 8, 671)]
    def run(controlled):
        """Simulate 8 seconds of flight, with or without the controller."""
        initial = np.zeros(6)  # start perfectly at hover, so all errors are zero
        segments = []  # holds the results from each time chunk
        # Go through the three time chunks one at a time
        for i, t_eval in enumerate(time_parts):
            sol = solve_ivp(lambda t, s: dynamics(t, s, controlled),
                            (float(t_eval[0]), float(t_eval[-1])), initial,
                            t_eval=t_eval, rtol=1e-9, atol=1e-11, max_step=0.01)
            # Stop if the solver ran into trouble
            if not sol.success:
                raise RuntimeError(sol.message)
            # Each chunk ends where the next one starts, so drop the last point
            # of the first two chunks to avoid counting that moment twice
            segments.append(sol.y[:, :-1] if i < 2 else sol.y)
            initial = sol.y[:, -1]  # the next chunk picks up where this one ended
        # Stitch the time points together the same way (skipping the repeats)
        t = np.concatenate([p[:-1] if i < 2 else p for i, p in enumerate(time_parts)])
        return t, np.hstack(segments)

    # Run the simulation twice: once with LQR on, once with no control
    t, controlled = run(True)
    _, uncontrolled = run(False)
    # Work out the motor commands the controller used at every time step
    control_input = -(K @ controlled)
    # Total control effort = area under (thrust^2 + torque^2) over time.
    # Newer NumPy calls this trapezoid and older versions call it trapz, so we check which one exists
    effort = np.trapezoid(np.sum(control_input**2, axis=0), t) if hasattr(np, 'trapezoid') else np.trapz(np.sum(control_input**2, axis=0), t)
    print('Peak horizontal error (controlled):', round(float(np.max(np.abs(controlled[0]))), 5), 'm')
    print('Integrated squared control effort:', round(float(effort), 5))

    # Save the plots in a results/ folder next to this file (create it if it's missing)
    output_dir = Path(__file__).resolve().parent / 'results'
    output_dir.mkdir(exist_ok=True)
    # Plot 1: sideways drift over time, with and without the controller
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, controlled[0], label='With LQR')
    ax.plot(t, uncontrolled[0], label='Without control', linestyle='--')
    ax.axvspan(1, 1.3, alpha=0.15, color='gray', label='Wind gust')  # shade the gust window
    ax.set(xlabel='Time (s)', ylabel='Horizontal position error (m)', title='Quadrotor hover recovery after wind gust')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / 'position_error.png', dpi=180)

    # Plot 2: pitch angle on top, motor commands on the bottom
    fig2, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, np.rad2deg(controlled[2]))  # convert radians to degrees so it's easier to read
    axes[0].set_ylabel('Pitch angle (degrees)')
    axes[1].plot(t, control_input[0], label='Thrust correction (N)')
    axes[1].plot(t, control_input[1], label='Pitch torque (N m)')
    axes[1].set(xlabel='Time (s)', ylabel='Control input')
    axes[1].legend()
    # Add grid lines to both subplots
    for ax in axes:
        ax.grid(True)
    fig2.tight_layout()
    fig2.savefig(output_dir / 'pitch_and_control.png', dpi=180)
    plt.show()


# Only run main() when this file is run directly (not when it's imported)
if __name__ == '__main__':
    main()
