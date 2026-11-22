"""
drone_model.py

This file describes our drone in math form. We look at a simplified "planar"
quadrotor, meaning it can only move left/right (x), up/down (z), and tilt
forward/back (pitch, theta). Everything is in SI units (meters, seconds,
kilograms, radians).

Physical constants used here:
    MASS     - how heavy the drone is, in kilograms
    INERTIA  - how hard it is to spin the drone about its pitch axis (kg m^2)
    GRAVITY  - gravitational acceleration, in m/s^2
"""
import numpy as np

MASS = 1.0  # drone mass in kg (we picked a 1 kg drone to keep the numbers simple)
INERTIA = 0.02  # pitch moment of inertia in kg m^2 (resistance to tilting)
GRAVITY = 9.81  # gravity pulling the drone down, in m/s^2


def state_space(mass=MASS, inertia=INERTIA, gravity=GRAVITY):
    """Build the A and B matrices for the linearized hover model dx/dt = Ax + Bu.

    The state vector x has 6 entries, in this order:
        [x, z, theta, vx, vz, omega]
        x     = horizontal position (m)
        z     = altitude (m)
        theta = pitch angle (rad)
        vx    = horizontal velocity (m/s)
        vz    = vertical velocity (m/s)
        omega = pitch rate, how fast it's tilting (rad/s)

    The control vector u has 2 entries. Both are measured as the change away
    from what the drone needs to just hover in place:
        [extra total thrust (N), pitch torque (N m)]

    Sign convention: tilting to a positive pitch pushes the drone in the
    positive x direction.

    Returns A (6x6) and B (6x2).
    """
    # Physical values can't be zero or negative, so stop early if they are
    if mass <= 0 or inertia <= 0 or gravity <= 0:
        raise ValueError('mass, inertia, gravity must be positive')
    A = np.zeros((6, 6))  # start with all zeros, then fill in the parts that matter
    # Positions change according to their velocities (x' = vx, z' = vz, theta' = omega)
    A[0, 3] = A[1, 4] = A[2, 5] = 1.0
    # Tilting the drone points some of its thrust sideways, so pitch causes horizontal acceleration
    A[3, 2] = gravity
    B = np.zeros((6, 2))  # how the two control inputs feed into the state
    B[4, 0] = 1.0 / mass  # extra thrust speeds the drone up vertically (F = ma)
    B[5, 1] = 1.0 / inertia  # pitch torque makes the drone spin faster
    return A, B


def wind_acceleration(t, gust_start=1.0, gust_end=1.3, strength=2.0):
    """Return the sideways push (m/s^2) from a short wind gust at time t.

    The gust is a simple on/off pulse: it pushes with `strength` between
    gust_start and gust_end (in seconds), and does nothing the rest of the time.
    """
    return strength if gust_start <= t < gust_end else 0.0
