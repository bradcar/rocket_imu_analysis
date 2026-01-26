import matplotlib.pyplot as plt
import numpy as np

import mylib.prepare_functions as prep
import mylib.psd_functions as psd
from mylib.add_2d_plot_note import add_2d_plot_note

# use fp64 prints thoughout
np.set_printoptions(precision=10)


def read_prepare_6_dof(raw_data_file, plot_directory):
    """
    Reads raw 6DOF IMU data, processes attitude via Mahony filter, performs CoG
    translation, and transforms accelerations into the inertial frame.

    Parameters:
    -----------
    filename : str
        Path to the 'launch_data.txt' containing [time, ax, ay, az, gr, gp, gy].


    Returns:
    --------
    results : dict
        A dictionary containing processed flight data:
        - 'time': Truncated time vector
        - 'q': Quaternions (Body to Inertial)
        - 'euler': Tuple of (roll, pitch, yaw) in radians
        - 'accel_I': Tuple of (ax, ay, az) in the Inertial frame (gravity removed)
        - 'vel_I': Tuple of (vx, vy, vz) drift-compensated velocities
        - 'pos_I': Tuple of (px, py, pz) integrated positions
    """
    # --- A. Load Linear Accelerometer & Gyro
    data = np.loadtxt(raw_data_file).astype(np.float64)
    time_raw = data[:, 0]
    ax_b, ay_b, az_b = data[:, 1], data[:, 2], data[:, 3]
    gr_b, gp_b, gy_b = data[:, 4], data[:, 5], data[:, 6]

    # --- B. Calculate Acceleration Magnitude and Time Sampling Average interval (dt & freq)
    acceleration_mag = np.sqrt(ax_b ** 2 + ay_b ** 2 + az_b ** 2)
    deltas = np.diff(time_raw)
    dt_avg = np.mean(deltas)
    dt_min = np.min(deltas)
    dt_max = np.max(deltas)
    dt_std = np.std(deltas)
    sample_frequency = 1.0 / dt_avg

    print(f"\n--- Six DOF Processing START")
    print("Files:")
    print(f"\tData File: ./{raw_data_file}")
    print(f"\tPlot Directory: ./{plot_directory}")
    print("Flight Data:")

    # Data Freq should be greater than 100 Hz
    print(f"\tAverate Data Freq: {sample_frequency:.2f} Hz")
    print(f"\tAverage time step: {dt_avg:.4f} seconds ({dt_avg * 1000:.1f} msec)")
    print(f"\tMin/Max interval: {dt_min * 1000:.1f} ms / {dt_max * 1000:.1f} ms")

    # jitter should be less than 2%
    print(f"\tStandard Dev: {dt_std * 1000:.1f} ms, jitter: {(dt_std / dt_avg) * 100:.0f}%")

    # --- C. Launch/Land time Detection based on high acceleration
    launch_indices = np.where(acceleration_mag > 35.0)[0]
    if len(launch_indices) == 0:
        t_launch = time_raw[0]
    else:
        t_launch = time_raw[launch_indices[0]]

    post_launch_mask = (time_raw > t_launch + 2.0)
    if np.any(post_launch_mask):
        impact_idx = np.argmax(acceleration_mag[post_launch_mask])
        t_land = time_raw[post_launch_mask][impact_idx]
    else:
        t_land = time_raw[-1]

    plt.figure(figsize=(10, 4))
    plt.plot(time_raw, acceleration_mag, label="Total Acceleration (acceleration_mag)", color="gray", alpha=1.0)
    plt.axvline(t_launch, color="g", linestyle="--", label="Detected Launch")
    plt.axvline(t_land, color="r", linestyle="--", label="Detected Impact")
    plt.title("Step 2: Launch and Impact Detection Check")
    plt.ylabel("m/s^2")
    plt.legend()
    plt.xlim(t_launch - 0.5, t_land + 0.5)
    plt.grid(True)
    add_2d_plot_note("Actually launch & chute deploy")
    plt.savefig(f"{plot_directory}/launch-chute-plot.pdf")
    plt.show()

    # --- D.  Hand-adjusted launch/land times
    t_launch = 874.6
    t_land = 895.0

    print(f"\tDetected Launch: {t_launch:.2f}s")
    print(f"\tDetected Chute Deploy: {t_land:.2f}s")
    print(f"\tFlight Duration: {t_land - t_launch:.2f}s")

    # --- E.  Truncate raw_input_data for flight window add buffer
    buffer = 1.0
    time_t = prep.truncate(time_raw, time_raw, t_launch - buffer, t_land + buffer)
    ax_t, ay_t, az_t = [prep.truncate(v, time_raw, t_launch - buffer, t_land + buffer) for v in [ax_b, ay_b, az_b]]
    gr_t, gp_t, gy_t = [prep.truncate(v, time_raw, t_launch - buffer, t_land + buffer) for v in [gr_b, gp_b, gy_b]]
    acceleration_mag_t = prep.truncate(acceleration_mag, time_raw, t_launch - buffer, t_land + buffer)

    plt.figure(figsize=(10, 4))
    plt.plot(time_raw, acceleration_mag, label="Total Acceleration (acceleration_mag)", color="gray", alpha=1.0)
    plt.axvline(t_launch, color="g", linestyle="--", label="Detected Launch")
    plt.axvline(t_land, color="r", linestyle="--", label="Detected Impact")
    plt.title("Step 2: Hand-corrected Launch and Impact Detection Check")
    plt.ylabel("m/s^2")
    plt.legend()
    plt.xlim(t_launch - 0.3, t_land + 0.3)
    plt.grid(True)
    add_2d_plot_note("hand adjusted flight duration")
    plt.savefig(f"{plot_directory}/corrected-launch-chute-plot.pdf")
    plt.show()

    # --- F. Analyze PSD Power Spectral Density - Is filtering needed?
    ax_freq, ax_psd = psd.get_psd(ax_b, fs=sample_frequency, nperseg=1024)
    # plot_psd(ax_freq, ax_psd, title="PSD of Ax_b (whole input raw_input_data)")

    plt.figure(figsize=(8, 4))
    plt.loglog(ax_freq[1:], ax_psd[1:])  # skip DC bin
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (m²/s⁴/Hz)")
    plt.title("PSD of Ax_b (whole input raw_input_data)")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(f"{plot_directory}/full-psd-plot-ax.pdf")
    plt.show()

    ay_freq, ay_psd = psd.get_psd(ay_b, fs=sample_frequency, nperseg=1024)
    # plot_psd(ax_freq, ax_psd, title="PSD of Ax_b (whole input raw_input_data)")

    plt.figure(figsize=(8, 4))
    plt.loglog(ay_freq[1:], ay_psd[1:])  # skip DC bin
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (m²/s⁴/Hz)")
    plt.title("PSD of Ay_b (whole input raw_input_data)")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(f"{plot_directory}/full-psd-plot-ay.pdf")
    plt.show()

    az_freq, az_psd = psd.get_psd(az_b, fs=sample_frequency, nperseg=1024)
    # plot_psd(ax_freq, ax_psd, title="PSD of Ax_b (whole input raw_input_data)")

    plt.figure(figsize=(8, 4))
    plt.loglog(az_freq[1:], az_psd[1:])  # skip DC bin
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (m²/s⁴/Hz)")
    plt.title("PSD of Az_b (whole input raw_input_data)")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(f"{plot_directory}/full-psd-plot-az.pdf")
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.loglog(ax_freq[1:], ax_psd[1:])  # skip DC bin
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (m²/s⁴/Hz)")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.title("Zoomed PSD of Ax_b (whole input raw_input_data)")
    plt.xlim(0.2, 2.0)
    plt.ylim(0.1, 100.0)
    plt.savefig(f"{plot_directory}/zoomed-psd-plot-ax-zoom.pdf")
    plt.show()

    # print(f"PSD: {len(ax_b)=}, {len(ax_psd)=}")

    # --- G. FILTERING Data - Butterworth (or simple IIR)
    """ No Filtering indicated in PSD 
    # only perform simple IIR filter to see the error in the acceleration x-axis
    ax_iir = prep.simple_firstorder_iir_filter(ax_t, 0.5)

    # use higher accuracy Butterworth filtering
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

    # PSD of filtered Data
    ax_f_freq, ax_f_psd = psd.get_psd(ax_f, fs=sample_frequency, nperseg=1024)
    psd.plot_psd(ax_f_freq, ax_f_psd, title="PSD of Ax_f (truncated dataset)")
    print(f"{len(ax_f)=}, {len(ax_f_psd)=}")
    
    # Clipping Check
    plt.figure(figsize=(10, 4))
    plt.plot(time_t, ax_t, alpha=0.3, label="Raw Ax")
    plt.plot(time_t, ax_f, color="blue", label="Filtered Ax")
    plt.xlim(t_launch - 0.5, t_launch + 6.0)
    plt.ylim(-.2, 3.0)
    plt.title("Step 8: Sensor Health Check (Thrust Phase Clipping)")
    plt.ylabel("m/s^2")
    plt.legend()
    plt.grid(True)
    add_2d_plot_note("We know massive clip at launch, looks minimized here", x=0.03)
    plt.savefig(f"{plot_directory}/acceleration-clipping-plot.pdf")
    plt.show()

    """

    # *** REMOVE FILTERING
    ax_f, ay_f, az_f = ax_t, ay_t, az_t
    gr_f, gp_f, gy_f = gr_t, gp_t, gy_t

    # --- H. Center of Gravity TRANSLATION & BIAS REMOVAL ---
    rx, ry, rz = 0.0, 0.0, 0.0
    dt = dt_avg

    # Convert 4 inches offset to meters
    # TODO 4" offset is not for this rocket, but for 8" shell
    inch_offset = 4
    sensor_offset = inch_offset * 0.0254
    ax_cg, ay_cg, az_cg, a_error_mag = prep.correct_for_cog(ax_t, ay_t, az_t, gr_t, gp_t, gy_t, time_t, sensor_offset)
    # Print summary
    a_error_max = a_error_mag.max()
    print(
        f"\tEst max acceleration error due to {inch_offset}\" offset: {a_error_max:.2f} m/s², {(a_error_max / 9.81) * 100:.0f}%")

    ax_final, ay_final, az_final, gr_final, gp_final, gy_final = prep.remove_static_bias(
        ax_cg, ay_cg, az_cg, gr_f, gp_f, gy_f, time_t, t_launch
    )

    plt.figure(figsize=(10, 4))
    plt.plot(time_t, ax_f, label="Raw Filtered Ax", alpha=0.5)
    plt.plot(time_t, ax_final, label="Bias Corrected Ax", color="blue")
    plt.xlim(t_launch - 0.8, t_launch + 0.3)
    plt.ylim(-0.5, 2.0)
    plt.title("Step 4: post-flight Attempt at Static Bias Correction- Center Gravity & Bias Correction")
    plt.legend()
    plt.grid(True)
    add_2d_plot_note("orig IMU raw_input_data should zero bias, launch at 874.64", x=0.35)
    plt.savefig(f"{plot_directory}/imu-bias-plot.pdf")
    plt.show()

    # --- I. ATTITUDE ESTIMATION with Trapezoidal Integration-
    # Parameters: Linear Acceleration and Gyrometer
    # Returns: Quaternions (also Euler Angles)
    # Trapezoidal better tha Runge-Kutta in this use case
    roll, pitch, yaw, quat = prep.estimate_attitude_mahony_trapezoidal(
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

    print(f"--- Six DOF Processing END")
    return time_t, ax_final, ay_final, az_final, ax_f, quat, t_launch, t_land
