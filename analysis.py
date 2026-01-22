# Rocket trajectory post-flight analysis.
"""
Rocket trajectory post-flight analysis.

Data flow: raw → truncated → filtered → bias-corrected → inertial.

Cast input raw_input_data to FP64, Numpy defaults to 64-bit.

NAMING CONVENTION:
Preliminary raw_input_data procesing
    ax_t, ay_t, az_t   : truncated accel (flight window)
    gr_t, gp_t, gy_t   : truncated gyro (flight window)
    ax_f, ay_f, az_f   : accel filtered values (flight window)
    gr_f, gp_f, gy_f   : gyro filtered values (flight window)

Body frame (IMU frame):
    ax_b, ay_b, az_b   : body-frame accelerometer m/s^2: x, y, z
    gr_b, gp_b, gy_b   : body-frame angular rates (rad/s): roll, pitch, yaw

Inertial frame (earth-fixed):
    ax_I, ay_I, az_I   : inertial-frame accelerations

Plot axes:
    ax3d   : 3D trajectory axis
    ax_xy  : horizontal distance axis
    ax_alt : altitude / time-history axis

Axis convention (right-handed):

        +Z_I (up)
          |
          |
          o---- +X_I (downrange)
         /
       +Y_I (crossrange)

Body axes rotate with the body; inertial axes are fixed to Earth.

Outline of Analysis Pipeline
----------------------------
Data Truncation & Filtering:
    truncate(outvec, timevec, tlaunch, tland)
        → Crop a vector to a time window. tland really apogee
    butterworth filter - BETTER THAN IIR
    simple_firstorder_iir_filter(vec, tau)
        → Apply a simple IIR low-pass filter.
    *** Note we don't use either given rocket's small number samples and low noise

Spectral Analysis - verifies low noise
    get_psd(signal_vector, fs, nperseg=1024)
        → Compute power spectral density of a signal see if noise present
    rms_from_psd(df_psd) → Compute cumulative RMS from PSD

Quaternion / Attitude
    compute_gravity_error(qi, axi, ayi, azi)
        → Compute gravity alignment error for Mahony filter
    estimate_attitude_mahony_trapezoidal(time_t, ax, ay, az, gr, gp, gy, twokp=1.0)
        → Est body orientation as quaternions & Euler angles with trapezoidal Mahony filter
    body_to_inertial_acceleration(time_t, ax_b, ay_b, az_b, q)
        → Transform body-frame accelerations to inertial frame with gravity removed

Integration:
    integrate_acceleration(time_t, ax_inertial, ay_inertial, az_inertial, tland=None)
        → Double-integrate inertial accel to get vel & position with drift compensation

Correction for Center gravity - rocket CG vs sensor position:
    correct_for_cog(ax, ay, az, gr, gp, gy, rx=0.0, ry=0.0, rz=0.0, dt=0.01)
        → Translate body-frame accelerations to center-of-gravity frame.

Sensor Correction:
    remove_static_bias(ax, ay, az, gr, gp, gy, time_t, tlaunch)
        → Remove accelerometer and gyro bias using pre-launch window

BNO086 Fused Sensor Simplification:
=============================
    The BNO086 already provides fused quaternions (body → inertial) and
    linear acceleration (gravity subtracted).

    OUPUT NEEDED: 22 bytes per sample
     - timestamp - 2 bytes
     - quaternion - 8 bytes
     - linear_acceleration - 6 bytes
     - gyroscope - 6 bytes
     estimate total of 80K at 10 KiB/s
     - 76,800 bytes of data in 8 sec = transfer of about 10k/sec
     - 400 Hz * 24 bytes * 8 second (6 to 6.5 sec apogee + > 1 to 1.5 sec post burst)

    No need for:
    - Mahony or complementary filter integration
    - Manual gravity removal
    - Bias removal is mostly done with BNO086 & calibration
    Optional: may use a mild low-pass filter (10–50 Hz cutoff), test with PSD graph

    Still need COG correction (depends on rotation rate & distance)
     Will use COG correction on 8" shell: 4" away from CG which is aligned with center-of-rotation
     - with rotation ω = 30 deg/s ≈ 0.52 rad/s
     - r = 0.102 m (4" on an 8" shell)
     - Centripetal acceleration a = ω^2 r = 0.028 m/s² (0.3% of 9.8 m/s²)

    Double Integration:
    - Need to double integrate: linear acceleration → velocity → position:
        x, y, z = bno.linear_acceleration   # linear accel with no gravity!
    - Can skip linear drift compensation

    Simplified pipeline:
    - Read sensor → quaternion + linear_acceleration (no Gravity) + gyroscope
    - Optional filtering
        → mild low-pass on linear acceleration?
    - Inertial velocities
        → integrate linear acceleration
    - Inertial positions
        → integrate velocities (optional drift correction)
    - Visualization
        → 3D path, velocity, orientation

    To Calibrate before launch
    - Linear acceleration with gravity (raw accelerometer minus biases)
    - Gives true specific force + gravity
    Useful for:
        x, y, z = bno.acceleration # acceleration with gravity direction included
    _ Detecting orientation to Earth before lauch
    - Checking sensor calibration (should read ~9.81 m/s² when stationary)
    - Redundant safety check

SOURCE Credit - thanks to Carlos Montalvo!:
    https://github.com/cmontalvo251/aerospace/blob/main/rockets/PLAR/post_launch_analysis.py
    https://www.youtube.com/watch?v=mb1RNYKtWQE
"""

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets.widgets.trait_types import time_from_json

from mylib.add_2d_plot_note import add_2d_plot_note
from mylib.animate_projectile import animate_projectile
from mylib.quaternion_functions import quaternion_rotate
from mylib.read_prepare_6_dof import read_prepare_6_dof

# from mylib.read_prepare_6dof import read_prepare_9_dof

# use fp64 prints thoughout
np.set_printoptions(precision=10)

# --- CONSTANTS & UTILITIES ---
G_EARTH = 9.81


def body_to_inertial_acceleration(time_t, ax_b, ay_b, az_b, q):
    """
    Transform body-frame accelerometer measurements into inertial-frame
    accelerations using quaternions, with gravity removed.

    :params time_t  : timestamp
    :params ax_b    : body accelerometer x, with gravity
    :params ay_b    : body accelerometer y, with gravity
    :params az_b    : body accelerometer x, with gravity
    :params q       : quaternion BODY → INERTIAL
    :return ax_I    : inertial-frame x acceleration, gravity removed (m/s^2)
    :return ay_I    : inertial-frame y acceleration, gravity removed (m/s^2)
    :return az_I    : inertial-frame z acceleration, gravity removed (m/s^2)
    """

    num_pts = len(time_t)
    ax_inertial = np.zeros(num_pts)
    ay_inertial = np.zeros(num_pts)
    az_inertial = np.zeros(num_pts)

    for i in range(num_pts):
        a_body = np.array([ax_b[i], ay_b[i], az_b[i]])

        a_inertial_quaternion = quaternion_rotate(q[i], a_body)
        ax_inertial[i], ay_inertial[i], az_inertial[i] = a_inertial_quaternion

    return ax_inertial, ay_inertial, az_inertial


def integrate_acceleration(time_t, ax_inertial, ay_inertial, az_inertial, tland=None):
    """
    Integrate inertial accelerations to get velocity and position with linear drift compensation.

    :param time_t: timestamps (numpy array)
    :param ax_inertial: inertial x-acceleration (m/s^2)
    :param ay_inertial: inertial y-acceleration (m/s^2)
    :param az_inertial: inertial z-acceleration (m/s^2)
    :param tland: optional: flight end time (seconds) to clip positions after landing
    :return:
        vx_c, vy_c, vz_c : drift-compensated velocities
        px_f, py_f, pz_f : positions (meters)
    """
    num_pts = len(time_t)

    # Integrate acceleration to velocity
    vx = np.zeros(num_pts, dtype=np.float64)
    vy = np.zeros(num_pts, dtype=np.float64)
    vz = np.zeros(num_pts, dtype=np.float64)

    for i in range(1, num_pts):
        dt = time_t[i] - time_t[i - 1]
        vx[i] = vx[i - 1] + 0.5 * (ax_inertial[i] + ax_inertial[i - 1]) * dt
        vy[i] = vy[i - 1] + 0.5 * (ay_inertial[i] + ay_inertial[i - 1]) * dt
        vz[i] = vz[i - 1] + 0.5 * (az_inertial[i] + az_inertial[i - 1]) * dt

    # Drift compensation (linear)
    t_rel = time_t - time_t[0]
    total_t = time_t[-1] - time_t[0]
    vx_c = vx - (vx[-1] * t_rel / total_t)
    vy_c = vy - (vy[-1] * t_rel / total_t)
    vz_c = vz - (vz[-1] * t_rel / total_t)

    # Integrate velocity to position
    px_f = np.zeros(num_pts, dtype=np.float64)
    py_f = np.zeros(num_pts, dtype=np.float64)
    pz_f = np.zeros(num_pts, dtype=np.float64)

    for i in range(1, num_pts):
        dt = time_t[i] - time_t[i - 1]
        if tland is None or time_t[i] < tland:
            px_f[i] = px_f[i - 1] + 0.5 * (vx_c[i] + vx_c[i - 1]) * dt
            py_f[i] = py_f[i - 1] + 0.5 * (vy_c[i] + vy_c[i - 1]) * dt
            pz_f[i] = pz_f[i - 1] + 0.5 * (vz_c[i] + vz_c[i - 1]) * dt
        else:
            px_f[i], py_f[i], pz_f[i] = px_f[i - 1], py_f[i - 1], pz_f[i - 1]

    return vx_c, vy_c, vz_c, px_f, py_f, pz_f


##############################################################
# ANALYSIS OF IMU FLIGHT DATA - 9 DoF and 6 DoF processing
plot_directory = "plots"

# Read and prepare 6 DOF sensor data to create Fused results
filename = "raw_input_data/launch_data.txt"
time_f, ax_final, ay_final, az_final, ax_vert, quat, t_launch, t_land = read_prepare_6_dof(filename, plot_directory)

# Read and prepare 9 DOF sensor data Fused results
filename = "raw_input_data/shell_data_xxxxxx_v1.txt"
# CoG correction assumes spherical shell
# TODO check Quat & Gyro mapping
# time_t, ax_final, ay_final, az_final, ax_vert, quat, tlaunch, tland = read_prepare_9_dof_shell(filename, plot_directory)

# --- 1. INERTIAL TRANSFORM with Gravity removed
ax_I, ay_I, az_I = body_to_inertial_acceleration(time_f, ax_final, ay_final, az_final, quat)

plt.figure(figsize=(10, 4))
plt.plot(time_f, ax_I, label="Ax_Inertial")
plt.plot(time_f, ay_I, label="Ay_Inertial")
plt.plot(time_f, az_I, label="Az_Inertial", linewidth=2)
plt.title("Step 6: Inertial Acceleration (Gravity Subtracted)")
plt.ylabel("m/s^2")
plt.legend()
plt.grid(True)
add_2d_plot_note("Ay_I neg and Ax_I, positive during thrust, likely cross-axis artifact of miscalibration", x=0.02)
plt.savefig(f"{plot_directory}/inertial-acceleration-plot.pdf")
plt.show()

# --- 2. INTEGRATION of acceleration to create velocity and position
# Double Integrate:
#   1st: create velocities from acceleration
#   2nd: create positions from velocities
vx_c, vy_c, vz_c, px_f, py_f, pz_f = integrate_acceleration(time_f, ax_I, ay_I, az_I, tland=t_land)
print("\nCalculated Flight Data:")
print(f"\tMax altitude: {pz_f.max():.1f} m")
print(f"\tMax velocity: {vz_c.max():.1f} m/s")
print(f"\tMax acceleration: {az_I.max():.1f} m/s^2, {az_I.max() / G_EARTH:.1f} g")

# --- 3. FINAL Plots
print("\nCreate final plots")
# Stacked plot of altitude and ground drift horizontal
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(time_f, pz_f, color="blue", lw=2.5, label="Vertical Altitude (Z)")
plt.fill_between(time_f, pz_f, color="blue", alpha=0.1)
plt.axvline(t_launch, color="g", linestyle="--", label="Detected Launch")
plt.axvline(t_land, color="r", linestyle="--", label="Detected Chute")
plt.title("Step 8: Final Drift-Corrected Altitude")
plt.ylabel("Meters (m)")
add_2d_plot_note("likely altitude too low due to acceleration clipping at launch", x=0.4)
plt.legend()
# stacked
plt.subplot(2, 1, 2)
plt.plot(time_f, np.sqrt(px_f ** 2 + py_f ** 2), label="Ground Distance (Horizontal)", color="purple")
plt.title("Step 8: Ground Distance (Horizontal)")
plt.xlabel("Time (s)")
plt.ylabel("Meters (m)")
plt.legend()
plt.tight_layout()
add_2d_plot_note("seems reasonable drift scale", x=0.4)
plt.savefig(f"{plot_directory}/drift-plot.pdf")
plt.show()

# 3D Plot flight path
# set x-axis and y-axis to same range
x_min, x_max = px_f.min(), px_f.max()
y_min, y_max = py_f.min(), py_f.max()
x_center = 0.5 * (x_max + x_min)
y_center = 0.5 * (y_max + y_min)
span = max(x_max - x_min, y_max - y_min)
pad = 0.10 * span
half = 0.5 * span + pad

fig = plt.figure(figsize=(10, 8))
ax3d = fig.add_subplot(111, projection="3d")
ax3d.plot(px_f, py_f, pz_f, label="Flight Path", color="darkorange", lw=2)
ax3d.set_title("Step 8: 3D Flight Path - seems reasonable")
ax3d.set_xlabel("X (m)")
ax3d.set_ylabel("Y (m)")
ax3d.set_zlabel("Z (m)")
ax3d.set_xlim(x_center - half, x_center + half)
ax3d.set_ylim(y_center - half, y_center + half)
plt.savefig(f"{plot_directory}/flight-path.pdf")
plt.show()

# --- 3. FINAL ANIMATION in VPython
print("\nCreate Animation of flight")
# Animation of projectile in VPython using position, quaternion
# Output Altitude, Velocity, and Angular rotational velocity
animate_projectile(time_f, px_f, py_f, pz_f, vz_c, quat, ax_vert)
