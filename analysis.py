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
    truncate(outvec, timevec, tlaunch, tland) → Crop a vector to a time window. tland really apogee
    butterworth filter
    simple_firstorder_iir_filter(vec, tau) → Apply a simple IIR low-pass filter.
    *** Note we don't use either given rocket's small number samples and low noise

Spectral Analysis - verifies low noise
    get_psd(signal_vector, fs, nperseg=1024) → Compute power spectral density of a signal.
    rms_from_psd(df_psd) → Compute cumulative RMS from PSD.

Quaternion / Attitude
    compute_gravity_error(qi, axi, ayi, azi) → Compute gravity alignment error for Mahony filter.
    estimate_attitude_mahony_trapezoidal(time_t, ax, ay, az, gr, gp, gy, twokp=1.0) → Estimate body orientation as quaternions and Euler angles using trapezoidal Mahony filter.
    body_to_inertial_acceleration(time_t, ax_b, ay_b, az_b, q) → Transform body-frame accelerations to inertial frame with gravity removed.

Integration:
    integrate_acceleration(time_t, ax_inertial, ay_inertial, az_inertial, tland=None) → Double-integrate inertial accelerations to get velocity and position with drift compensation.

Correction for Center gravity - rocket CG vs sensor position:
    correct_for_cog(ax, ay, az, gr, gp, gy, rx=0.0, ry=0.0, rz=0.0, dt=0.01) → Translate body-frame accelerations to center-of-gravity frame.

Sensor Correction:
    remove_static_bias(ax, ay, az, gr, gp, gy, time_t, tlaunch) → Remove accelerometer and gyro bias using pre-launch window.


BNO086 Sensor Simplification:
    The BNO086 already provides fused quaternions (body → inertial) and linear acceleration (gravity subtracted).

    No need for:
    - Mahony or complementary filter integration
    - Manual gravity removal
     - Bias removal is mostly done with BNO086.
    - High sample rate + good sensor: you may skip manual low-pass filtering unless you want to remove residual vibration noise.
    Optional: a mild low-pass filter (10–50 Hz cutoff) if your vehicle vibrates. Test with PSD graph.

    May still need COG correction (if the IMU is rigidly attached near the CG)

    Integration:
    - Need to integrate linear acceleration → velocity → position:
        x, y, z = bno.linear_acceleration   # linear accel 3-tuple of x,y,z float returned with no gravity!
    - Can skip linear drift compensation

    Simplified pipeline:
    - Read sensor → quaternion + linear_acceleration + gyroscope
    - Optional filtering → mild low-pass on linear acceleration
    - Inertial velocities → integrate linear acceleration
    - Inertial positions → integrate velocities (optional drift correction)
    - Visualization → 3D path, velocity, orientation

    To Calibrate before launch
    - Linear acceleration with gravity (raw accelerometer minus biases)
    - Gives true specific force + gravity
    Useful for:
    x, y, z = bno.acceleration # acceleration 3-tuple of x,y,z float returned (gravity direction included)
    _ Detecting orientation to Earth before lauch
    - Checking sensor calibration (should read ~9.81 m/s² when stationary)
    - Redundant safety check

SOURCE Ideas:
    https://github.com/cmontalvo251/aerospace/blob/main/rockets/PLAR/post_launch_analysis.py
    https://www.youtube.com/watch?v=mb1RNYKtWQE
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from mylib.animate_projectile import animate_projectile
from mylib.quaternion_functions import quaternion_rotate

np.set_printoptions(precision=10)

# --- CONSTANTS & UTILITIES ---
G_EARTH = 9.81
filename = "raw_input_data/launch_data.txt"
plot_directory = "plots"


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


def add_2d_plot_note(note_text, ax=None, x=0.65, y=0.10, fontsize=9, color="green"):
    """
    Adds a semi-transparent text box note to a plot. with smaller 9pt Italic font

    Parameters:
        note_text (str): The text to display.
        ax (matplotlib.axes.Axes, optional): Axis to place the note on. current axes is default.
        x (float): x-axis position
        y (float): y-axis position
        fontsize (int): Font size for note.
        color (str): The color of note.
    """
    if ax is None:
        ax = plt.gca()
    ax.text(
        x, y,
        note_text,
        transform=ax.transAxes,  # Use axis-relative coordinates
        fontsize=fontsize,
        fontstyle="italic",
        color=color,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="green")
    )


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


def estimate_attitude_mahony_trapezoidal(time_t, ax, ay, az, gr, gp, gy, twokp=1.0):
    """
    Attitude estimation using trapezoidal integration of quaternion kinematics.
    Mahony-style complementary filter using quaternion kinematics and trapezoidal integration

    Trapezoidal better tha Runge-Kutta in this case, because
    - Stable with noisy raw_input_data
    - Works with irregular timestamps
    - Cancels some high-frequency noise
    - Matches post-flight analysis best practices (NASA, ESA)

    q is the time history of the body’s orientation, expressed as a unit quaternion mapping the body frame to the inertial frame

    Uses accelerometer for gravity direction correction and gyro for propagation.
    Trapezoidal/Euler integration is preferred over Runge-Kutta for noisy IMU raw_input_data
    with irregular timestamps.

    Returns:
        roll, pitch, yaw : Euler angles (rad)
        q                : quaternion history (body → inertial)
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


def body_to_inertial_acceleration(time_t, ax_b, ay_b, az_b, q):
    """
    Transform body-frame accelerometer measurements into inertial-frame
    accelerations using quaternions, with gravity removed.

    :params time_t  : timestamp
    :params ax_b    : body accelerometer position(m/s^2), bias-corrected
    :params ay_b    : body accelerometer orientation(m/s^2), bias-corrected
    :params az_b    : body accelerometer orientation(m/s^2), bias-corrected
    :params q       : quaternion BODY → INERTIAL
    :return ax_b    : body accelerometer x, with gravity removed (m/s^2)
    :return ay_b    : body accelerometer y, with gravity removed (m/s^2)
    :return az_b     : bodyaccelerometer z, with gravity removed (m/s^2)
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

    # --- Step 1: Integrate acceleration to velocity ---
    vx = np.zeros(num_pts, dtype=np.float64)
    vy = np.zeros(num_pts, dtype=np.float64)
    vz = np.zeros(num_pts, dtype=np.float64)

    for i in range(1, num_pts):
        dt = time_t[i] - time_t[i - 1]
        vx[i] = vx[i - 1] + 0.5 * (ax_inertial[i] + ax_inertial[i - 1]) * dt
        vy[i] = vy[i - 1] + 0.5 * (ay_inertial[i] + ay_inertial[i - 1]) * dt
        vz[i] = vz[i - 1] + 0.5 * (az_inertial[i] + az_inertial[i - 1]) * dt

    # --- Step 2: Drift compensation (linear) ---
    t_rel = time_t - time_t[0]
    total_t = time_t[-1] - time_t[0]
    vx_c = vx - (vx[-1] * t_rel / total_t)
    vy_c = vy - (vy[-1] * t_rel / total_t)
    vz_c = vz - (vz[-1] * t_rel / total_t)

    # --- Step 3: Integrate velocity to position ---
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


def correct_for_cog(ax, ay, az, gr, gp, gy, rx=0.0, ry=0.0, rz=0.0, dt=0.01):
    """
    Translate body-frame accelerations to center of gravity.
    :param ax: x-acceleration (m/s^2)
    :param ay: y-acceleration (m/s^2)
    :param az: z-acceleration (m/s^2)
    :param gr: roll gravity (m/s^2)
    :param gp: pitch gravity (m/s^2)
    :param gy: yaw y-acceleration (m/s^2)
    :param rx: roll x-acceleration (m/s^2)
    :param ry: yaw x-acceleration (m/s^2)
    :param rz: roll z-acceleration (m/s^2)
    :param dt:

    :return: ax_cg
    : ay_cg
    : az_cg
    """
    gr_dot, gp_dot, gy_dot = np.gradient(gr, dt), np.gradient(gp, dt), np.gradient(gy, dt)

    ax_cg = ax - (-(gp ** 2 + gy ** 2) * rx + (gr * gp - gy_dot) * ry + (gr * gy + gp_dot) * rz)
    ay_cg = ay - ((gr * gp + gy_dot) * rx - (gr ** 2 + gy ** 2) * ry + (gp * gy - gr_dot) * rz)
    az_cg = az - ((gr * gy - gp_dot) * rx + (gp * gy + gr_dot) * ry - (gr ** 2 + gp ** 2) * rz)

    return ax_cg, ay_cg, az_cg


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


# POWER SPECTRAL DENSITY
def get_psd(signal_vector, fs, nperseg=1024):
    """
    Power spectral density (PSD) calculation

    Usage:
        ax_freq, ax_psd = get_psd(ax_b, fs=sample_frequency, nperseg=1024)
        plot_psd(ax_freq, ax_psd, title="PSD of Ax_b (whole input raw_input_data)")
        print(f"{len(ax_b)=}, {len(ax_psd)=}")

    :param signal_vector: input
    :param fs: sampling frequency
    :param nperseg:  number of points for PSD
    :return:
    """
    nperseg = min(nperseg, len(signal_vector))

    f, psd = signal.welch(
        signal_vector,
        fs=fs,
        nperseg=nperseg,
        window="hann",
        detrend="constant",
        scaling="density"
    )

    return f, psd


def plot_psd(f, psd, title="Power Spectral Density"):
    plt.figure(figsize=(8, 4))
    plt.loglog(f[1:], psd[1:])  # skip DC bin
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (m²/s⁴/Hz)")
    plt.title(title)
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()


def rms_from_psd(df_psd):
    """
    TODO: Convert from pandas df to numpy array
    df_rms = rms_from_psd(psd)
    fig = fig_from_df(df_rms)
    fig.update_xaxes(type="log", title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Acceleration gRMS")
    fig.write_html('cum-rms.html', full_html=False, include_plotlyjs='cdn')
    df_rms.to_csv('cum-rms.csv')
    fig.show()
    """
    d_f = df_psd.index[1] - df_psd.index[0]
    df_rms = df_psd.copy()
    df_rms = df_rms * d_f
    df_rms = df_rms.cumsum()
    return df_rms ** 0.5


##############################################################

# --- 1. DATA LOADING & ANALYSIS ---
data = np.loadtxt(filename).astype(np.float64)
time = data[:, 0]
ax_b, ay_b, az_b = data[:, 1], data[:, 2], data[:, 3]
gr_b, gp_b, gy_b = data[:, 4], data[:, 5], data[:, 6]

# Calculate Magnitude and Sampling Stats
A_mag = np.sqrt(ax_b ** 2 + ay_b ** 2 + az_b ** 2)
dt_avg = np.mean(np.diff(time))
sample_frequency = 1.0 / dt_avg

print(f"File: {filename}")
print(f"Sample Data Freq: {sample_frequency:.2f} Hz")
print(f"Average time step: {dt_avg:.4f} seconds ({dt_avg * 1000:.1f} msec)")

# --- 2. Launch/Land DETECTION ---
launch_indices = np.where(A_mag > 35.0)[0]
if len(launch_indices) == 0:
    tlaunch = time[0]
else:
    tlaunch = time[launch_indices[0]]

post_launch_mask = (time > tlaunch + 2.0)
if np.any(post_launch_mask):
    impact_idx = np.argmax(A_mag[post_launch_mask])
    tland = time[post_launch_mask][impact_idx]
else:
    tland = time[-1]

# # TODO REMOVE THIS analysis for entire duration
# tland = time[-1]

plt.figure(figsize=(10, 4))
plt.plot(time, A_mag, label="Total Acceleration (A_mag)", color="gray", alpha=1.0)
plt.axvline(tlaunch, color="g", linestyle="--", label="Detected Launch")
plt.axvline(tland, color="r", linestyle="--", label="Detected Impact")
plt.title("Step 2: Launch and Impact Detection Check")
plt.ylabel("m/s^2")
plt.legend()
plt.xlim(tlaunch - 0.5, tland + 0.5)
plt.grid(True)
add_2d_plot_note("Actually launch & chute deploy")
plt.savefig(f"{plot_directory}/launch-chute-plot.pdf")
plt.show()

# Hand-adjusted launch/land times
tlaunch = 874.6
tland = 889.5
# # TODO REMOVE THIS analysis for entire duration
# tland = 919

print(f"Detected Launch: {tlaunch:.2f}s")
print(f"Detected Chute Deploy: {tland:.2f}s")
print(f"Flight Duration: {tland - tlaunch:.2f}s")

# Truncate raw_input_data for flight window + buffer
buffer = 1.0
time_t = truncate(time, time, tlaunch - buffer, tland + buffer)
ax_t, ay_t, az_t = [truncate(v, time, tlaunch - buffer, tland + buffer) for v in [ax_b, ay_b, az_b]]
gr_t, gp_t, gy_t = [truncate(v, time, tlaunch - buffer, tland + buffer) for v in [gr_b, gp_b, gy_b]]
A_mag_t = truncate(A_mag, time, tlaunch - buffer, tland + buffer)

plt.figure(figsize=(10, 4))
plt.plot(time, A_mag, label="Total Acceleration (A_mag)", color="gray", alpha=1.0)
plt.axvline(tlaunch, color="g", linestyle="--", label="Detected Launch")
plt.axvline(tland, color="r", linestyle="--", label="Detected Impact")
plt.title("Step 2: Hand-corrected Launch and Impact Detection Check")
plt.ylabel("m/s^2")
plt.legend()
plt.xlim(tlaunch - 0.3, tland + 0.3)
plt.grid(True)
add_2d_plot_note("hand adjusted flight duration")
plt.savefig(f"{plot_directory}/corrected-lauch-chute-plot.pdf")
plt.show()

# --- 3. FILTERING Data ---
# only perform simple IIR filter to see the error in the acceleration x-axis
ax_iir = simple_firstorder_iir_filter(ax_t, 0.5)

# More accurate Butterworth filtering
cutoff = sample_frequency / 4.0
sos = signal.butter(2, cutoff / (0.5 * sample_frequency), btype="lowpass", output="sos")
ax_f, ay_f, az_f = [signal.sosfiltfilt(sos, v) for v in [ax_t, ay_t, az_t]]
gr_f, gp_f, gy_f = [signal.sosfiltfilt(sos, v) for v in [gr_t, gp_t, gy_t]]

plt.figure(figsize=(12, 6))
plt.scatter(time_t, ax_t, color="black", s=8, alpha=1.0, label="Raw Data")
plt.plot(time_t, ax_iir, color="red", label="IIR (Lagged)")
plt.plot(time_t, ax_f, color="blue", linewidth=2, label="Butterworth (Zero Phase)")
plt.title("Step 3: Filter Phase-Lag Comparison")
plt.ylabel("m/s^2")
plt.legend()
add_2d_plot_note("IIR lags, only forward looking, use Butterworh filter", x=0.40)
plt.savefig(f"{plot_directory}/iir-butterworth-plot.pdf")
plt.show()

# --- 3b. Analyze PSD ---

ax_freq, ax_psd = get_psd(ax_b, fs=sample_frequency, nperseg=1024)
plot_psd(ax_freq, ax_psd, title="PSD of Ax_b (whole input raw_input_data)")
print(f"{len(ax_b)=}, {len(ax_psd)=}")

ax_f_freq, ax_f_psd = get_psd(ax_f, fs=sample_frequency, nperseg=1024)
plot_psd(ax_f_freq, ax_f_psd, title="PSD of Ax_f (truncated dataset)")
print(f"{len(ax_f)=}, {len(ax_f_psd )=}")

# REMOVE FILTERING
ax_f, ay_f, az_f = ax_t, ay_t, az_t
gr_f, gp_f, gy_f = gr_t, gp_t, gy_t

# --- 4. Sensor vs Center of Gravity TRANSLATION & BIAS REMOVAL ---
rx, ry, rz = 0.0, 0.0, 0.0
dt = dt_avg

ax_cg, ay_cg, az_cg = correct_for_cog(ax_f, ay_f, az_f, gr_f, gp_f, gy_f, rx, ry, rz, dt)
ax_final, ay_final, az_final, gr_final, gp_final, gy_final = remove_static_bias(
    ax_cg, ay_cg, az_cg, gr_f, gp_f, gy_f, time_t, tlaunch
)

plt.figure(figsize=(10, 4))
plt.plot(time_t, ax_f, label="Raw Filtered Ax", alpha=0.5)
plt.plot(time_t, ax_final, label="Bias Corrected Ax", color="blue")
plt.xlim(tlaunch - 0.8, tlaunch + 0.3)
plt.ylim(-0.5, 2.0)
plt.title("Step 4: post-flight Attempt at Static Bias Correction- Center Gravity & Bias Correction")
plt.legend()
plt.grid(True)
add_2d_plot_note("orig IMU raw_input_data should zero bias, launch at 874.64", x=0.35)
plt.savefig(f"{plot_directory}/imu-bias-plot.pdf")
plt.show()

# --- 5. ATTITUDE ESTIMATION - Trapezoidal Integration---
# Trapezoidal better tha Runge-Kutta in this use case

roll, pitch, yaw, q = estimate_attitude_mahony_trapezoidal(
    time_t, ax_final, ay_final, az_final, gr_final, gp_final, gy_final, twokp=1.0)

plt.figure(figsize=(10, 4))
plt.plot(time_t, np.degrees(pitch), label="Pitch (Degrees)", color="orange")
plt.plot(time_t, np.degrees(roll), label="Roll (Degrees)", alpha=0.5)
plt.title("Step 5: Attitude Estimation (Euler Angles)")
plt.ylabel("Degrees")
plt.legend()
plt.grid(True)
add_2d_plot_note("pitch & roll more as go higher, gimbal flip is discontinuity", x=0.085)
plt.savefig(f"{plot_directory}/roll-pitch-plot.pdf")
plt.show()

# --- 6. INERTIAL TRANSFORM ---
# Inertial transform with Gravity removed
ax_I, ay_I, az_I = body_to_inertial_acceleration(time_t, ax_final, ay_final, az_final, q)

plt.figure(figsize=(10, 4))
plt.plot(time_t, ax_I, label="Ax_Inertial")
plt.plot(time_t, ay_I, label="Ay_Inertial")
plt.plot(time_t, az_I, label="Az_Inertial", linewidth=2)
plt.title("Step 6: Inertial Acceleration (Gravity Subtracted)")
plt.ylabel("m/s^2")
plt.legend()
plt.grid(True)
add_2d_plot_note("Ay_I neg and Ax_I, positive during thrust, likely cross-axis artifact of miscalibration", x=0.02)
plt.savefig(f"{plot_directory}/inertial-acceleration-plot.pdf")
plt.show()

# --- 7. INTEGRATION & DRIFT COMPENSATION ---

# Double Integrate: 1st: create velocities from acceleration,  2nd: create positions from velocities
# vx_c = corrected
vx_c, vy_c, vz_c, px_f, py_f, pz_f = integrate_acceleration(time_t, ax_I, ay_I, az_I, tland=tland)

# --- 8. FINAL VISUALIZATIONS ---
# Clipping Check
plt.figure(figsize=(10, 4))
plt.plot(time_t, ax_t, alpha=0.3, label="Raw Ax")
plt.plot(time_t, ax_f, color="blue", label="Filtered Ax")
plt.xlim(tlaunch - 0.5, tlaunch + 6.0)
plt.ylim(-.2, 3.0)
plt.title("Step 8: Sensor Health Check (Thrust Phase Clipping)")
plt.ylabel("m/s^2")
plt.legend()
plt.grid(True)
add_2d_plot_note("We know massive clip at launch, looks minimized here", x=0.03)
plt.savefig(f"{plot_directory}/acceleration-clipping-plot.pdf")
plt.show()

# Stacked plot of altitude and ground drift horizontal
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(time_t, pz_f, color="blue", lw=2.5, label="Vertical Altitude (Z)")
plt.fill_between(time_t, pz_f, color="blue", alpha=0.1)
plt.axvline(tlaunch, color="g", linestyle="--", label="Detected Launch")
plt.axvline(tland, color="r", linestyle="--", label="Detected Chute")
plt.title("Step 8: Final Drift-Corrected Altitude")
plt.ylabel("Meters (m)")
add_2d_plot_note("likely altitude too low due to acceleration clipping at launch", x=0.4)
plt.legend()
# stacked
plt.subplot(2, 1, 2)
plt.plot(time_t, np.sqrt(px_f ** 2 + py_f ** 2), label="Ground Distance (Horizontal)", color="purple")
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

# animation of projectile in vpython using this raw_input_data
animate_projectile(time_t, px_f, py_f, pz_f, vz_c, q, ax_f)
