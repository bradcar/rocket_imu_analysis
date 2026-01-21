import matplotlib.pyplot as plt
import numpy as np

import mylib.psd_functions as psd
import mylib.prepare_functions as prep
from mylib.add_2d_plot_note import add_2d_plot_note
from mylib.quaternion_functions import quaternion_rotate

# use fp64 thoughout
np.set_printoptions(precision=10)


def read_prepare_9_dof(raw_data_file, plot_directory):
    """
    Reads raw 6DOF IMU data, processes attitude via Mahony filter, performs CoG
    translation, and transforms accelerations into the inertial frame.

    """
    # --- A. Load Quaternion, Linear Accelerometer (No Gravity), & Gyro
    # Using BNO086 22-byte or similar packed format)
    data = np.loadtxt(raw_data_file).astype(np.float64)
    time_raw = data[:, 0]
    # BNO Quaternions: Hamiltonian (r, i, j, k)
    q_raw = data[:, 1:5]
    # Linear Accel (Gravity already removed by BNO086 hardware)
    ax_lin, ay_lin, az_lin = data[:, 5], data[:, 6], data[:, 7]
    # Gyros (used in CoG correction)
    gy, gp, gr = data[:, 8], data[:, 9], data[:, 10]

    # --- B. Calculate Time Sampling Averate interval (dt & freq)
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

    # Data Freq should be greater than 100 Hz
    print(f"\tAverate Data Freq: {sample_frequency:.2f} Hz")
    print(f"\tAverage time step: {dt_avg:.4f} seconds ({dt_avg * 1000:.1f} msec)")
    print(f"\tMin/Max interval: {dt_min * 1000:.1f} ms / {dt_max * 1000:.1f} ms")

    # jitter should be less than 2%
    print(f"\tStandard Dev: {dt_std * 1000:.1f} ms, jitter: {(dt_std / dt_avg) * 100:.0f}%")

    # --- C. Launch/Land time Detection based on high acceleration
    t_launch_manual = None
    t_land_manual = None
    a_mag = np.sqrt(ax_lin ** 2 + ay_lin ** 2 + az_lin ** 2)
    t_launch = t_launch_manual if t_launch_manual else time_raw[np.where(a_mag > 30.0)[0][0]]
    t_chute = t_land_manual if t_land_manual else time_raw[-1]  # Simplification

    # 3. TRUNCATION
    mask = (time_raw > t_launch - 0.5) & (time_raw < t_chute + 0.5)
    t_f = time_raw[mask]
    q_f = q_raw[mask]
    ax_b, ay_b, az_b = ax_lin[mask], ay_lin[mask], az_lin[mask]
    gr_f, gp_f, gy_f = gr[mask], gp[mask], gy[mask]

    # --- D.  Hand-adjusted launch/land times
    tlaunch = 874.6
    tland = 889.5

    print(f"\tDetected Launch: {tlaunch:.2f}s")
    print(f"\tDetected Land: {tland:.2f}s")
    print(f"\tFlight Duration: {tland - tlaunch:.2f}s")

    plt.figure(figsize=(10, 4))
    plt.plot(time_raw, acceleration_mag, label="Total Acceleration (acceleration_mag)", color="gray", alpha=1.0)
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

    plt.figure(figsize=(10, 4))
    plt.plot(time_raw, acceleration_mag, label="Total Acceleration (acceleration_mag)", color="gray", alpha=1.0)
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

    # --- E. Analyze PSD Power Spectral Density - Is filtering needed?
    ax_freq, ax_psd = psd.get_psd(ax_b, fs=sample_frequency, nperseg=1024)
    # plot_psd(ax_freq, ax_psd, title="PSD of Ax_b (whole input raw_input_data)")

    plt.figure(figsize=(8, 4))
    plt.loglog(ax_freq[1:], ax_psd[1:])  # skip DC bin
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (m²/s⁴/Hz)")
    plt.title("PSD of Ax_b (whole input raw_input_data)")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(f"{plot_directory}/full-psd-plot.pdf")
    plt.show()

    ax_freq, ax_psd = psd.get_psd(ay_b, fs=sample_frequency, nperseg=1024)
    # plot_psd(ax_freq, ax_psd, title="PSD of Ax_b (whole input raw_input_data)")

    plt.figure(figsize=(8, 4))
    plt.loglog(ax_freq[1:], ax_psd[1:])  # skip DC bin
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (m²/s⁴/Hz)")
    plt.title("PSD of Ay_b (whole input raw_input_data)")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(f"{plot_directory}/full-psd-plot.pdf")
    plt.show()

    ax_freq, ax_psd = psd.get_psd(az_b, fs=sample_frequency, nperseg=1024)
    # plot_psd(ax_freq, ax_psd, title="PSD of Ax_b (whole input raw_input_data)")

    plt.figure(figsize=(8, 4))
    plt.loglog(ax_freq[1:], ax_psd[1:])  # skip DC bin
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (m²/s⁴/Hz)")
    plt.title("PSD of Az_b (whole input raw_input_data)")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(f"{plot_directory}/full-psd-plot.pdf")
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
    plt.savefig(f"{plot_directory}/zoomed-psd-plot.pdf")
    plt.show()

    # print(f"PSD: {len(ax_b)=}, {len(ax_psd)=}")

    # --- F. FILTERING Data - Butterworth (or simple IIR)
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
    plt.xlim(tlaunch - 0.5, tlaunch + 6.0)
    plt.ylim(-.2, 3.0)
    plt.title("Step 8: Sensor Health Check (Thrust Phase Clipping)")
    plt.ylabel("m/s^2")
    plt.legend()
    plt.grid(True)
    add_2d_plot_note("We know massive clip at launch, looks minimized here", x=0.03)
    plt.savefig(f"{plot_directory}/acceleration-clipping-plot.pdf")
    plt.show()

    """

    # TODO CORRECT IPLEMENTATION ABOVE
    # TODO IMPLEMENT rest of processing !!!

    # 4. COG CORRECTION
    # 4 inches offset to meters
    # TODO 4" offset is guess for 8" shell
    inch_offset = 4
    sensor_offset = inch_offset * 0.0254
    gr_dot, gp_dot, gy_dot = [np.gradient(v, dt_avg) for v in [gr_f, gp_f, gy_f]]
    ax_cg = ax_b - (sensor_offset * (gr_dot + gr_f ** 2))
    ay_cg = ay_b - (sensor_offset * (gp_dot + gp_f ** 2))
    az_cg = az_b - (sensor_offset * (gy_dot + gy_f ** 2))

    # 5. INERTIAL TRANSFORM
    # We use the sensor's OWN quaternions to rotate the lin_accel to Inertial Frame
    ax_I = np.zeros_like(t_f)
    ay_I = np.zeros_like(t_f)
    az_I = np.zeros_like(t_f)

    for i in range(len(t_f)):
        # BNO086 provides Body -> Inertial orientation
        a_inertial = quaternion_rotate(q_f[i], [ax_cg[i], ay_cg[i], az_cg[i]])
        ax_I[i], ay_I[i], az_I[i] = a_inertial

    print(f"--- NINE DOF Processing END")
    return time_t, ax_final, ay_final, az_final, ax_f, quat, tlaunch, tland
