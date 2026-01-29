import numpy as np

# use fp64 thoughout
np.set_printoptions(precision=10)

def truncate(outvec, timevec, ts, te):
    """Crops the vector to the specified time window using a boolean mask."""
    mask = (timevec > ts) & (timevec < te)
    return outvec[mask]


def simple_firstorder_iir_filter(vec, tau):
    """Manual low-pass filter (for comparison purposes)."""
    outvec = np.zeros_like(vec, dtype=np.float64)
    outvec[0] = vec[0]
    for x in range(1, len(vec)):
        outvec[x] = (1 - tau) * outvec[x - 1] + tau * vec[x]
    return outvec


def estimate_attitude_mahony_trapezoidal(time_t, ax, ay, az, gr, gp, gy, twokp=1.0):
    """
    Attitude estimation using trapezoidal integration of quaternion kinematics.
    Mahony-style complementary filter using quaternion kinematics and trapezoidal integration.

    Mahony? is computationally efficient and handles the lack of a magnetometer (6 DOF) better than a
    standard Kalman filter by using the gravity vector to "level" the gyro's pitch and roll.

    Using Trapezoidal integration reduces the "truncation error" compared to simple rectangular integration.
    This keeps the orientation (quat) much more stable during the high-angular-rate portions of the flight.

    Trapezoidal is better tha Runge-Kutta in this case, because:
    - Stable with noisy flight_sensor_input_data
    - Best with irregular timestamps
    - Cancels some high-frequency noise
    - Aligns with post-flight analysis best practices (NASA, ESA)

    quat is the time history of the body’s orientation, expressed as a unit quaternion mapping the body frame to
    the inertial frame.

    Uses accelerometer for gravity direction correction and gyro for propagation.

    Returns:
        roll, pitch, yaw : Euler angles (rad)
        quat                : quaternion (body → inertial)
    """

    num_pts = len(time_t)
    q = np.zeros((num_pts, 4), dtype=np.float64)
    q[0] = [1.0, 0.0, 0.0, 0.0]

    for i in range(num_pts - 1):
        axi, ayi, azi = ax[i], ay[i], az[i]
        gri, gpi, gyi = gr[i], gp[i], gy[i]

        # Normalize accelerometer (gravity direction only)
        norm = np.sqrt(axi ** 2 + ayi ** 2 + azi ** 2)
        if norm > 0.0:
            axi /= norm
            ayi /= norm
            azi /= norm

        # Gravity alignment error
        ex, ey, ez = compute_gravity_error(q[i], axi, ayi, azi)

        # Proportional feedback (Mahony correction)
        gri += twokp * ex
        gpi += twokp * ey
        gyi += twokp * ez

        # Quaternion kinematics
        dt = time_t[i + 1] - time_t[i]
        dq = 0.5 * np.array([
            -q[i, 1] * gri - q[i, 2] * gpi - q[i, 3] * gyi,
            q[i, 0] * gri + q[i, 2] * gyi - q[i, 3] * gpi,
            q[i, 0] * gpi - q[i, 1] * gyi + q[i, 3] * gri,
            q[i, 0] * gyi + q[i, 1] * gpi - q[i, 2] * gri
        ], dtype=np.float64)

        # Trapezoidal / Euler integration step
        q[i + 1] = q[i] + dq * dt
        q[i + 1] /= np.linalg.norm(q[i + 1])

    # Convert quaternion history to Euler angles
    roll = np.arctan2(
        2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3]),
        1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    )

    pitch = np.arcsin(
        np.clip(2 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1]), -1.0, 1.0)
    )

    yaw = np.arctan2(
        2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
        1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)
    )

    return roll, pitch, yaw, q


def compute_gravity_error(qi, axi, ayi, azi):
    """
    Compute Mahony gravity-direction error vector.

    Parameters:
        qi  : quaternion at time i, shape (4,)
        axi : normalized accelerometer x
        ayi : normalized accelerometer y
        azi : normalized accelerometer z

    Returns:
        ex, ey, ez : gravity alignment error vector
    """

    # Estimated gravity direction from quaternion
    vx = 2.0 * (qi[1] * qi[3] - qi[0] * qi[2])
    vy = 2.0 * (qi[0] * qi[1] + qi[2] * qi[3])
    vz = qi[0] ** 2 - 0.5 + qi[3] ** 2

    # Gravity error (cross product: measured vs estimated)
    ex = ayi * vz - azi * vy
    ey = azi * vx - axi * vz
    ez = axi * vy - ayi * vx

    return ex, ey, ez

def correct_for_cog(ax, ay, az, gr, gp, gy, time_t, sensor_offset_from_cg=0.102):
    """
    Translate body-frame accelerations to center of gravity.
    :param ax: x-acceleration (m/s^2)
    :param ay: y-acceleration (m/s^2)
    :param az: z-acceleration (m/s^2)
    :param gr: roll rate (rad/s)
    :param gp: pitch rate (rad/s)
    :param gy: yaw rate (rad/s)
    :param dt:
    :param sensor_offset_from_cg:

    :return: ax_cg
    : ay_cg
    : az_cg
    """
    # Compute angular acceleration (rad/s²)
    gr_dot = np.gradient(gr, time_t)
    gp_dot = np.gradient(gp, time_t)
    gy_dot = np.gradient(gy, time_t)

    # Estimate rotational acceleration error along each axis
    ax_err = sensor_offset_from_cg * (gr_dot + gr ** 2)
    ay_err = sensor_offset_from_cg * (gp_dot + gp ** 2)
    az_err = sensor_offset_from_cg * (gy_dot + gy ** 2)

    # Correct accelerations by subtracting estimated rotational contribution
    ax_cg = ax - ax_err
    ay_cg = ay - ay_err
    az_cg = az - az_err

    # Error due to sensor & CG offset, at each point in time
    # for i in range(len(ax_cg)):
    #     err = np.sqrt(ax_err[i] ** 2 + ay_err[i] ** 2 + az_err[i] ** 2)
    #     print(f"{i=}, error: {err=}, {(err/9.81)*100:.0f}%")

    # Max error magnitude for reference
    a_error_mag = np.sqrt(ax_err ** 2 + ay_err ** 2 + az_err ** 2)

    return ax_cg, ay_cg, az_cg, a_error_mag

def remove_static_bias(ax, ay, az, gr, gp, gy, time_t, tlaunch):
    """
    Remove accelerometer and gyro biases using pre-launch window.
    """
    pre_launch_mask = (time_t < tlaunch)
    if np.any(pre_launch_mask):
        ax_bias, ay_bias = np.mean(ax[pre_launch_mask]), np.mean(ay[pre_launch_mask])
        gr_bias, gp_bias, gy_bias = np.mean(gr[pre_launch_mask]), np.mean(gp[pre_launch_mask]), np.mean(
            gy[pre_launch_mask])

        ax_final = ax - ax_bias
        ay_final = ay - ay_bias
        az_final = az

        gr_final = gr - gr_bias
        gp_final = gp - gp_bias
        gy_final = gy - gy_bias
    else:
        ax_final, ay_final, az_final = ax, ay, az
        gr_final, gp_final, gy_final = gr, gp, gy

    return ax_final, ay_final, az_final, gr_final, gp_final, gy_final