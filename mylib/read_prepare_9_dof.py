# read_prepare_9_dof.py
"""
BNO086 Post-Flight Analysis Mapping

1. BNO086 Sensor Frame (Body Frame) [ log_linacc_quat_gyro_flash_spi.py ]
-------------------------------------------------------------------------
The physical sensor orientation on the projectile.
	+X (Roll): Sensor Right (i component)
	+Y (Pitch): Sensor Forward (j component)
	+Z (Yaw): Sensor Up (k component)
	time_stamps in milliseconds (0.1 msec resolution)

Output CSV FILE:
	a_body: linear_acceleration (HW-fused, gravity removed).
	quat_final:   quaternion (Body → World)
	g_final:   quaternion (Body → World)
	time_stamps in SECONDS (0.0001 sec resolution)

2. Rigidbody / CoG Correction (Body Frame) [read_prepare_9_dof.py]
------------------------------------------------------------------
Before moving to world coordinates, we correct for the sensor offset.
	Input ω: Must be [X,Y,Z] order → [gr, gp, gy]
	Vector r: [sensor_offset, 0, 0] (Offset along the Right/Roll axis)

CoG Correction:
	a_cg = a_sensor −(α×r)−(ω×(ω×r))

Output:
	a_f = a_cg is still in the Body Frame

3. Inertial / World (Analysis Frame) [analysis.py]
--------------------------------------------------
Applying q_bw to a_cg to find true motion relative to the ground.
	+X I (Down-range): North/East or Launch direction.
	+Y I (Cross-range): Left/Right drift.
	+Z I (Altitude): Up (Vertical).

Double Integration:
	v_world = ∫ a_world dt
	p_world = ∫ v_world dt

4. VPython Display (Visual Frame) [animate_projectile.py]
---------------------------------
Mapping the Analysis Frame to VPython’s Screen Coordinates.
	VPython +X: Right (Screen) ← Analysis X (Down-range)
	VPython +Y: Up (Screen) ← Analysis Z (Altitude)
	VPython +Z: Toward User ← Analysis Y (Cross-range)

Code for Position:
	pos = vector(px_f, pz_f, py_f)
	a_z = az_I = acceleration up

Code for Orientation:
	VPython's pointer or axis uses the same remap:
	obj.axis = rotate(vector(1,0,0), quat)
		where the vector is remapped to match the visual world.

AXES Summary Table for Code Consistency
Quantity        Body	Analysis(Inertial)	VPython
Forward/Up       +Y           +Z              +Y
Right/Downrange  +X           +X              +X
Vertical/Cross   +Z           +Y              +Z
"""
import matplotlib.pyplot as plt
import numpy as np

import mylib.psd_functions as psd
from mylib.add_2d_plot_note import add_2d_plot_note

# use fp64 prints thoughout
np.set_printoptions(precision=10)


def cog_correction_shell(sensor_offset, time_f, ax_b, ay_b, az_b, gr_f, gy_f, gp_f):
    """
    CoG Correction for shell, accurate gyro data is critical
    This 9 DOF uses a proper Vector Cross Product method. Which is much better for a "tumbling shell."

    a_cg =a_sensor − (alpha × r) − (ω × (ω × r))
    alpha × r: Tangential acceleration (due to change in spin rate).
    ω × (ω × r): Centripetal acceleration (due to constant spin).
    """
    r_cg_to_sensor = np.array([sensor_offset, 0.00, 0.00])  # meters

    # Angular velocity (rad/s) in BODY frame: gr=ωx, gp=ωy, gy=ωz
    omega = np.column_stack((gr_f, gy_f, gp_f))

    # Angular acceleration (rad/s^2)
    alpha = np.column_stack((
        np.gradient(gr_f, time_f),
        np.gradient(gy_f, time_f),
        np.gradient(gp_f, time_f)
    ))

    # Sensor linear acceleration (gravity already removed)
    a_sensor = np.column_stack((ax_b, ay_b, az_b))

    # Rigid-body correction terms
    alpha_cross_r = np.cross(alpha, r_cg_to_sensor)
    omega_cross_r = np.cross(omega, r_cg_to_sensor)
    centripetal = np.cross(omega, omega_cross_r)

    # Center-of-gravity acceleration
    a_cg = a_sensor - alpha_cross_r - centripetal

    # Unpack
    ax_cg, ay_cg, az_cg = a_cg.T
    return ax_cg, ay_cg, az_cg


def read_prepare_9_dof_shell(raw_data_file, plot_directory, sensor_cm_offset):
    """
    Reads raw 9 DOF IMU data for spherical shell,
    performs CoG translation where CoG and center-of-rotation align,
    and transforms accelerations into the inertial frame.
    :param raw_data_file: filename of data file to analyze/simulate
    :param plot_directory: output plot directory
    :param sensor_cm_offset: cm's offset between sensor and center-of-gravity
    """
    # --- A. Load Quaternion, Linear Accelerometer (No Gravity), & Gyro

    # Using BNO086 22-byte packed format
    # data = np.loadtxt(raw_data_file).astype(np.float64)

    # read CSV file skip header
    data = np.genfromtxt(raw_data_file, delimiter=",", dtype=np.float64, skip_header=1)
    print("data read finished")
    time_raw = data[:, 0]

    # Linear Accel (Gravity already removed by BNO086 hardware)
    ax_lin, ay_lin, az_lin = data[:, 1], data[:, 2], data[:, 3]

    # TODO test re-map of Quaternion
    # BNO Quaternions: Hamiltonian (r, i, j, k)
    q_raw = np.column_stack((
        data[:, 4],  # qr Scalar is same
        data[:, 5],  # qi is Mapped to X (roll)
        data[:, 6],  # qj is Mapped to Z (yaw)
        data[:, 7]   # qk is Mapped to Y (pitch)
    ))

    # Gyros (used in CoG correction), we use Yaw-Pitch-Roll in sensor
    # Match Body Frame [X, Y, Z]
    gr = -data[:, 10]  # Roll mapped to X (negated)
    gy = data[:, 8]  # Yaw mapped to Y
    gp = data[:, 9]  # Pitch mapped to Z

    # --- B. Calculate Time Sampling Average interval (dt & freq)

    print("Time range:", time_raw.min(), time_raw.max())
    print("Accel raw sample:", ax_lin[:5], ay_lin[:5], az_lin[:5])
    print("Quaternion sample:", q_raw[:5])

    deltas = np.diff(time_raw)
    dt_avg = np.mean(deltas)
    dt_min = np.min(deltas)
    dt_max = np.max(deltas)
    dt_std = np.std(deltas)
    sample_frequency = 1.0 / dt_avg

    print(f"\n--- NINE DOF Processing START")
    print("Files:")
    print(f"\tData File: ./{raw_data_file}")
    print(f"\tPlot Directory: ./{plot_directory}")
    print("Flight Data:")

    # Data freq should be greater than 100 Hz
    print(f"\tAverate Data Freq: {sample_frequency:.2f} Hz")
    print(f"\tAverage time step: {dt_avg:.4f} seconds ({dt_avg * 1000:.1f} msec)")
    print(f"\tMin/Max interval: {dt_min * 1000:.1f} ms / {dt_max * 1000:.1f} ms")

    # Jitter should be less than 2%
    print(f"\tStandard Dev: {dt_std * 1000:.1f} ms, jitter: {(dt_std / dt_avg) * 100:.0f}%")

    # --- C. Launch/Land time Detection based on high acceleration and truncate data if needed
    t_launch_manual = None
    t_launch_manual = None
    t_land_manual = None

    acceleration_mag = np.sqrt(ax_lin ** 2 + ay_lin ** 2 + az_lin ** 2)
    acceleration_threshold = 100.0
    idx = np.where(acceleration_mag > acceleration_threshold)[0]

    if t_launch_manual is not None:
        t_launch = t_launch_manual
    elif idx.size == 0:
        print(f"No launch spike detected (>{acceleration_threshold:.1f} m/s^2). Proceeding without truncation.")
        t_launch = time_raw[0]
    else:
        t_launch = time_raw[idx[0]]

    t_land = t_land_manual if t_land_manual is not None else time_raw[-1]

    print(f"\tDetected Launch: {t_launch:.2f}s")
    print(f"\tDetected Land: {t_land:.2f}s")
    print(f"\tFlight Duration: {t_land - t_launch:.2f}s")

    # --- D.  Hand-adjusted launch/land times
    # TODO adjust as needed
    # t_launch = 874.6
    # t_land = 889.5
    # print(f"\tAdjust Launch: {t_launch:.2f}s")
    # print(f"\tAdjusted Land: {t_land:.2f}s")
    # print(f"\tAdjusted Flight Duration: {t_land - t_launch:.2f}s")

    # Truncate all data to the launch/land times
    mask = (time_raw > t_launch - 0.5) & (time_raw < t_land + 0.5)
    time_f = time_raw[mask]
    quat_f = q_raw[mask]
    ax_b, ay_b, az_b = ax_lin[mask], ay_lin[mask], az_lin[mask]
    gy_f, gp_f, gr_f = gy[mask], gp[mask], gr[mask]

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
    plt.savefig(f"{plot_directory}/corrected-lauch-chute-plot.pdf")
    plt.show()

    # --- E. Analyze PSD Power Spectral Density - Is filtering needed?
    ax_freq, ax_psd = psd.get_psd(ax_b, fs=sample_frequency, nperseg=1024)
    # plot_psd(ax_freq, ax_psd, title="PSD of Ax_b (whole input raw_input_data)")

    plt.figure(figsize=(8, 4))
    plt.loglog(ax_freq[1:], ax_psd[1:])  # skip DC bin
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (m²/s⁴/Hz)")
    plt.title("PSD of ax_b (whole input raw_input_data)")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(f"{plot_directory}/full-psd-plot.pdf")
    plt.show()

    ay_freq, ay_psd = psd.get_psd(ay_b, fs=sample_frequency, nperseg=1024)
    # plot_psd(ax_freq, ax_psd, title="PSD of Ax_b (whole input raw_input_data)")

    plt.figure(figsize=(8, 4))
    plt.loglog(ay_freq[1:], ay_psd[1:])  # skip DC bin
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (m²/s⁴/Hz)")
    plt.title("PSD of ay_b (whole input raw_input_data)")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(f"{plot_directory}/full-psd-plot-ax.pdf")
    plt.show()

    az_freq, az_psd = psd.get_psd(az_b, fs=sample_frequency, nperseg=1024)
    # plot_psd(ax_freq, ax_psd, title="PSD of Ax_b (whole input raw_input_data)")

    plt.figure(figsize=(8, 4))
    plt.loglog(az_freq[1:], az_psd[1:])  # skip DC bin
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (m²/s⁴/Hz)")
    plt.title("PSD of az_b (whole input raw_input_data)")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(f"{plot_directory}/full-psd-plot-ay.pdf")
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.loglog(ax_freq[1:], ax_psd[1:])  # skip DC bin
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (m²/s⁴/Hz)")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.title("Zoomed PSD of ax_b (whole input raw_input_data)")
    plt.xlim(0.2, 2.0)
    plt.ylim(0.1, 100.0)
    plt.savefig(f"{plot_directory}/zoomed-psd-plot-az.pdf")
    plt.show()

    # debug print
    # print(f"PSD: {len(ax_b)=}, {len(ax_psd)=}")

    # --- F. FILTERING Data - Butterworth (or simple IIR)
    # No Filtering indicated in PSD

    # # use higher accuracy Butterworth filtering
    # cutoff = sample_frequency / 4.0
    # sos = signal.butter(2, cutoff / (0.5 * sample_frequency), btype="lowpass", output="sos")
    # ax_f, ay_f, az_f = [signal.sosfiltfilt(sos, v) for v in [ax_b, ay_b, az_b]]
    # gy_f, gp_f, gr_f = [signal.sosfiltfilt(sos, v) for v in [gy_f, gp_f, gr_f]]
    #
    # plt.figure(figsize=(12, 6))
    # plt.scatter(time_t, ax_t, color="black", s=8, alpha=1.0, label="Raw Data")
    # plt.plot(time_t, ax_g, color="red", label="ax_b unfiltered")
    # plt.plot(time_t, ax_f, color="blue", linewidth=2, label="Butterworth (Zero Phase)")
    # plt.title("Step 3: Filter & Unfilterd Comparison")
    # plt.ylabel("m/s^2")
    # plt.legend()
    # add_2d_plot_note("add comment about Butterworh filter", x=0.40)
    # plt.savefig(f"{plot_directory}/butterworth-orig-plot.pdf")
    # plt.show()
    #
    # # PSD of filtered Data
    # ax_f_freq, ax_f_psd = psd.get_psd(ax_f, fs=sample_frequency, nperseg=1024)
    # psd.plot_psd(ax_f_freq, ax_f_psd, title="PSD of Ax_f (truncated dataset)")
    # print(f"{len(ax_f)=}, {len(ax_f_psd)=}")
    #
    # # Clipping Check raw vs. Filtered Zoomed in
    # plt.figure(figsize=(10, 4))
    # plt.plot(time_f, ax_b, alpha=0.3, label="Raw Ax")
    # plt.plot(time_f, ax_f, color="blue", label="Filtered Ax")
    # plt.xlim(t_launch - 0.5, t_launch + 6.0)
    # plt.ylim(-.2, 3.0)
    # plt.title("IMU Health Check (Lift Phase Clipping)")
    # plt.ylabel("m/s^2")
    # plt.legend()
    # plt.grid(True)
    # add_2d_plot_note("likely massive clip at launch", x=0.03)
    # plt.savefig(f"{plot_directory}/acceleration-clipping-plot.pdf")
    # plt.show()

    # 4. COG CORRECTION where CoG and Center-of-rotation coincide
    # For a tumbling ball, use the full 3D rigid body correction
    # Sensor position relative to CoG assuming CoG and center-rotation aligned, expressed in BODY frame
    sensor_offset = sensor_cm_offset / 1000.0
    print(f"Sensor offset: {sensor_offset:.3f} m, (should be 0.102 m for 4in on an 8in shell)")
    ax_cg, ay_cg, az_cg = cog_correction_shell(sensor_offset, time_f, ax_b, ay_b, az_b, gr_f, gy_f, gp_f)

    ax_final, ay_final, az_final = ax_cg, ay_cg, az_cg

    # # 5. CONVERT TO WORLD FRAME INERTIAL TRANSFORM
    # ax_I = np.zeros_like(time_f)
    # ay_I = np.zeros_like(time_f)
    # az_I = np.zeros_like(time_f)
    #
    # for i in range(len(time_f)):
    #     a_body = [ax_cg[i], ay_cg[i], az_cg[i]]
    #     a_world = quaternion_rotate(quaternion_conjugate(quat_f[i]), a_body)
    #     ax_I[i], ay_I[i], az_I[i] = a_world
    #
    #
    # print("Vertical accel mean:", np.mean(az_I))
    # print("Vertical accel std:", np.std(az_I))
    #
    # # Body-frame acceleration at CoG (gravity already removed)
    # ax_final = ax_cg
    # ay_final = ay_cg
    # az_final = az_cg
    #
    # # Vertical acceleration (sensor Z mapped to inertial Z)
    # # TODO CHECK Inertial Z is vertical and Quaternion is Earth-aligned
    # a_vertical = az_I

    print(f"--- NINE DOF Processing END")
    # return time_f, ax_final, ay_final, az_final, a_vertical, quat_f, t_launch, t_land
    return time_f, ax_final, ay_final, az_final, quat_f, t_launch, t_land
