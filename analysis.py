import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Constants
GEARTH = 9.81
filename = 'data/launch_data.txt'


def truncate(outvec, timevec, ts, te):
    """Crops the vector to the specified time window using a boolean mask."""
    mask = (timevec > ts) & (timevec < te)
    return outvec[mask]


def simple_firstorder_IIR_filter(vec, tau):
    """The original manual low-pass filter (introduces phase lag)."""
    outvec = np.zeros_like(vec)
    outvec[0] = vec[0]
    for x in range(1, len(vec)):
        outvec[x] = (1 - tau) * outvec[x - 1] + tau * vec[x]
    return outvec


###############################################
# File load
data = np.loadtxt(filename)

time = data[:, 0]
Ax = data[:, 1]
Ay = data[:, 2]
Az = data[:, 3]
gx = data[:, 4]
gy = data[:, 5]
gz = data[:, 6]

# Calculate Magnitute for launch detection
A_mag = np.sqrt(Ax ** 2 + Ay ** 2 + Az ** 2)

# Calculate time statistics
dt_vector = np.diff(time)
dt_avg = np.mean(dt_vector)
sampling_freq = 1.0 / dt_avg

print("Data Analysis:")
print("==============")
print(f"File Loaded: {filename}")
print(f"Average time step: {dt_avg:.4f} seconds ({dt_avg * 1000.:.1f} msec)")
print(f"Average Sampling Frequency: {sampling_freq:.2f} Hz")

## --- REWRITTEN EVENT DETECTION ---

# 1. Detect Launch (First time we exceed a thrust threshold)
# 35 m/s^2 is roughly 3.5Gs - good for motor ignition
launch_threshold = 35.0
launch_indices = np.where(A_mag > launch_threshold)[0]

if len(launch_indices) == 0:
    print("Error: No launch detected. Check thresholds.")
    tlaunch = time[0]
    tland = time[-1]
else:
    tlaunch = time[launch_indices[0]]

    # 2. Detect Landing (The "Impact Spike")
    # We look for the maximum acceleration that occurs AFTER launch + some buffer
    # (Buffer of 2 seconds to ignore motor jitters/burnout)
    post_launch_mask = time > (tlaunch + 2.0)

    if np.any(post_launch_mask):
        # Find the index of the highest acceleration spike after the motor burn
        # This is almost always the ground impact or parachute deployment
        impact_idx = np.argmax(A_mag[post_launch_mask])
        tland = time[post_launch_mask][impact_idx]

        # Validation: If the impact "spike" is too small, it might be noise.
        # You can tune this 15.0 value based on your data.
        if A_mag[post_launch_mask][impact_idx] < 15.0:
            print("Warning: No clear impact spike found. Defaulting to end of file.")
            tland = time[-1]
    else:
        tland = time[-1]

print(f"Detected Launch: {tlaunch:.2f}s")
print(f"Detected Impact: {tland:.2f}s")
print(f"Flight Duration: {tland - tlaunch:.2f}s")

# Apply truncation with 1 second buffer
buffer = 1.0
time_t = truncate(time, time, tlaunch - buffer, tland + buffer)
Ax_t = truncate(Ax, time, tlaunch - buffer, tland + buffer)
Ay_t = truncate(Ay, time, tlaunch - buffer, tland + buffer)
Az_t = truncate(Az, time, tlaunch - buffer, tland + buffer)
gx_t = truncate(gx, time, tlaunch - buffer, tland + buffer)
gy_t = truncate(gy, time, tlaunch - buffer, tland + buffer)
gz_t = truncate(gz, time, tlaunch - buffer, tland + buffer)

plt.figure(figsize=(10, 4))
plt.plot(time, A_mag, label='Total Acceleration (A_mag)')
plt.axvline(tlaunch, color='g', label='Detected Launch', linestyle='--')
plt.axvline(tland, color='r', label='Detected Impact', linestyle='--')
plt.title('Launch and Impact Detection Check')
plt.ylabel('m/s^2')
plt.legend()
plt.xlim(870, 890)
plt.show()

# Filter the data - remove noise
# A) Original Method: First-Order IIR
tau = 0.5
Ax_iir = simple_firstorder_IIR_filter(Ax_t, tau)

# B) Modern Method: Zero-Phase Butterworth
# Using suggested fs/4 rule
cutoff_freq = sampling_freq / 4.0
nyquist = 0.5 * sampling_freq
Wn = cutoff_freq / nyquist

b, a = signal.butter(2, Wn, btype='low')
Ax_butter = signal.filtfilt(b, a, Ax_t)

plt.figure(figsize=(12, 7))
plt.grid(True, which='both', linestyle='--', alpha=0.6)

# Plot Raw Data points to see the source "noise"
plt.scatter(time_t, Ax_t, color='black', s=8, alpha=0.3, label='Raw Data Points')

# Plot the IIR Filter (Showcasing the lag)
plt.plot(time_t, Ax_iir, color='red', linewidth=1.5,
         label=f'Original IIR (tau={tau}) - Phase Lagged')

# Plot the Butterworth Filter (Showcasing zero-phase alignment)
plt.plot(time_t, Ax_butter, color='blue', linewidth=2.5,
         label=f'Butterworth filtfilt ({cutoff_freq:.2f}Hz) - Zero Phase')

plt.title(f'Filter Comparison at {sampling_freq:.2f} Hz Sampling Rate',
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Acceleration X (m/s^2)', fontsize=12)
plt.legend(loc='best', frameon=True, shadow=True)

# Annotate the lag difference
plt.annotate('IIR Peak is late (lag)', xy=(tlaunch, np.max(Ax_iir)),
             xytext=(tlaunch + 1, np.max(Ax_iir) + 5),
             arrowprops=dict(facecolor='red', shrink=0.05), color='red')

plt.show()

# since the data has been plotted let's use Butterworth on all data
# Ax_butter = signal.filtfilt(b, a, Ax_t) is recalculated as
# Ax_f = signal.filtfilt(b, a, Ax_t)
Ax_f = signal.filtfilt(b, a, Ax_t)
Ay_f = signal.filtfilt(b, a, Ay_t)
Az_f = signal.filtfilt(b, a, Az_t)
gx_f = signal.filtfilt(b, a, gx_t)
gy_f = signal.filtfilt(b, a, gy_t)
gz_f = signal.filtfilt(b, a, gz_t)

plt.plot(time_t, Ax_t, label='Ax_t')
plt.plot(time_t, Ay_t, label='Ay_t')
plt.plot(time_t, Az_t, label='Az_t')
plt.plot(time_t, Ax_f, label='Ax_f')
plt.plot(time_t, Ay_f, label='Ay_f')
plt.plot(time_t, Az_f, label='Az_tf')
plt.title('Acceleration data & filter comparison')
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Acceleration X (m/s^2)', fontsize=12)
plt.legend()
plt.xlim(time_t[0], 895)  # first data point to arbitrary number 895, TODO FIX better number
plt.show()

# 3.) Translate Accel to Center of Gravity (CG)

# Define the offset from the IMU to the CG (in meters)
# Example: If the IMU is 15cm forward of the CG, rx = 0.15
rx = 0.0  # Forward(+) / Aft(-)
ry = 0.0  # Right(+) / Left(-)
rz = 0.0  # Up(+) / Down(-)

# Calculate angular acceleration (alpha) by taking the derivative of gyro data
# np.diff returns an array of length N-1, so we pad it to maintain alignment
dt = dt_avg
gx_dot = np.gradient(gx_f, dt)
gy_dot = np.gradient(gy_f, dt)
gz_dot = np.gradient(gz_f, dt)

# Correct the Accelerometer data for the offset (Simplified Rigid Body Translation)
# This removes the "swing" effect if the IMU is not at the pivot point
Ax_cg = Ax_f - (-(gy_f ** 2 + gz_f ** 2) * rx + (gx_f * gy_f - gz_dot) * ry + (gx_f * gz_f + gy_dot) * rz)
Ay_cg = Ay_f - ((gx_f * gy_f + gz_dot) * rx - (gx_f ** 2 + gz_f ** 2) * ry + (gy_f * gz_f - gx_dot) * rz)
Az_cg = Az_f - ((gx_f * gz_f - gy_dot) * rx + (gy_f * gz_f + gx_dot) * ry - (gx_f ** 2 + gy_f ** 2) * rz)

# 3.5) Bias Removal (Static Calibration)
# Define the "quiet period" mask (data before the detected launch)
# We assume the projectile is stationary on the pad during this time.
pre_launch_mask = (time_t < tlaunch)

if np.any(pre_launch_mask):
    # 1. Accelerometer Bias (X and Y only)
    # Az is ignored here because it should measure exactly 1.0g (9.81 m/s^2)
    Ax_bias = np.mean(Ax_cg[pre_launch_mask])
    Ay_bias = np.mean(Ay_cg[pre_launch_mask])

    Ax_final = Ax_cg - Ax_bias
    Ay_final = Ay_cg - Ay_bias
    Az_final = Az_cg  # Az remains raw; gravity is subtracted in Step 5

    # 2. Gyroscope Bias (X, Y, and Z)
    # A stationary gyro should ideally read 0.0 rad/s on all axes.
    gx_bias = np.mean(gx_f[pre_launch_mask])
    gy_bias = np.mean(gy_f[pre_launch_mask])
    gz_bias = np.mean(gz_f[pre_launch_mask])

    gx_final = gx_f - gx_bias
    gy_final = gy_f - gy_bias
    gz_final = gz_f - gz_bias

    print(f"--- Bias Correction Applied (using {np.sum(pre_launch_mask)} samples) ---")
    print(f"Accel Bias (m/s^2): X:{Ax_bias:.4f}, Y:{Ay_bias:.4f}")
    print(f"Gyro Bias (rad/s):  X:{gx_bias:.5f}, Y:{gy_bias:.5f}, Z:{gz_bias:.5f}")
else:
    print("Warning: No pre-launch data found. Skipping bias correction.")
    Ax_final, Ay_final, Az_final = Ax_cg, Ay_cg, Az_cg
    gx_final, gy_final, gz_final = gx_f, gy_f, gz_f

plt.figure(figsize=(12, 5))
plt.plot(time_t, Ay_f, label='Measured Ay (at IMU)', alpha=0.5, linestyle='--')
plt.plot(time_t, Ay_final, label='Corrected Ay (at CG)', color='blue')
plt.title('IMU vs. Center of Gravity Acceleration (Y-Axis)')
plt.ylabel('m/s^2')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))
# Zoom into the window just before tlaunch
plt.xlim(tlaunch - 0.8, tlaunch + 0.2)
plt.ylim(-0.5, 0.5)  # Tight Y-axis to see small biases

plt.plot(time_t, Ax_final, label='Bias-Corrected Ax')
plt.plot(time_t, Ay_final, label='Bias-Corrected Ay')
plt.axhline(0, color='black', lw=1)  # The "Zero" target
plt.title('Static Bias Check (Pre-Launch Window)')
plt.ylabel('m/s^2')
plt.legend()
plt.grid(True)
plt.show()

print("--- CG Translation and Bias Correction Complete ---")

plt.figure(figsize=(12, 6))
plt.grid(True, alpha=0.3)
plt.plot(time_t, Ax_f, label='Raw Filtered Ax', color='gray', alpha=0.5, linestyle='--')
plt.plot(time_t, Ax_final, label='Bias-Corrected Ax', color='blue')
plt.axhline(0, color='black', linewidth=1)

# Zoom into the pre-launch area to see the bias correction
plt.xlim(tlaunch - 0.8, tlaunch + 0.5)
plt.ylim(-2, 2)

plt.title('Accelerometer Bias Correction (Zoomed at Launch)', fontsize=12)
plt.xlabel('Time (sec)')
plt.ylabel('Acceleration (m/s^2)')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.grid(True, alpha=0.3)
plt.plot(time_t, gy_f, label='Raw Filtered Gy', color='gray', alpha=0.5, linestyle='--')
plt.plot(time_t, gy_final, label='Bias-Corrected Gy', color='red')
plt.axhline(0, color='black', linewidth=1)

plt.title('Gyroscope Y-Axis Bias Correction', fontsize=12)
plt.xlabel('Time (sec)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(time_t, Ay_f, label='Measured Ay (at IMU)', alpha=0.6)
plt.plot(time_t, Ay_final, label='Corrected Ay (at CG)', color='green', linewidth=2)
plt.title('IMU Offset Correction: Measured vs. CG Acceleration')
plt.ylabel('m/s^2')
plt.legend()
plt.show()

## --- 4. ATTITUDE ESTIMATION (QUATERNION INTEGRATION) ---

# Initialize quaternion arrays (Length same as time_t)
num_pts = len(time_t)
q0 = np.ones(num_pts)
q1 = np.zeros(num_pts)
q2 = np.zeros(num_pts)
q3 = np.zeros(num_pts)

# Feedback gain (Kp) - controls how much we trust Accel vs Gyro
# With only 4Hz data, we need a steady hand.
twoKp = 1.0

for i in range(num_pts - 1):
    # 1. Get current data
    ax, ay, az = Ax_final[i], Ay_final[i], Az_final[i]
    gx, gy, gz = gx_final[i], gy_final[i], gz_final[i]

    # 2. Normalize Accelerometer measurement
    # This identifies the "Down" vector
    norm = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    if norm > 0:
        ax, ay, az = ax / norm, ay / norm, az / norm

    # 3. Estimated direction of gravity (from current quaternion)
    # This is the "Expected" gravity vector based on our last orientation
    vx = 2.0 * (q1[i] * q3[i] - q0[i] * q2[i])
    vy = 2.0 * (q0[i] * q1[i] + q2[i] * q3[i])
    vz = q0[i] ** 2 - 0.5 + q3[i] ** 2

    # 4. Error is cross product between estimated and measured gravity
    ex = (ay * vz - az * vy)
    ey = (az * vx - ax * vz)
    ez = (ax * vy - ay * vx)

    # 5. Apply feedback to Gyro (Mahony logic)
    gx += twoKp * ex
    gy += twoKp * ey
    gz += twoKp * ez

    # 6. Integrate Quaternion rate of change
    dt = time_t[i + 1] - time_t[i]

    # Pre-multiply common factors
    gxi, gyi, gzi = gx * 0.5 * dt, gy * 0.5 * dt, gz * 0.5 * dt

    # Update Quaternions
    q0[i + 1] = q0[i] + (-q1[i] * gxi - q2[i] * gyi - q3[i] * gzi)
    q1[i + 1] = q1[i] + (q0[i] * gxi + q2[i] * gzi - q3[i] * gyi)
    q2[i + 1] = q2[i] + (q0[i] * gyi - q1[i] * gzi + q3[i] * gxi)
    q3[i + 1] = q3[i] + (q0[i] * gzi + q1[i] * gyi - q2[i] * gxi)

    # 7. Normalize Quaternion (keeps it mathematically valid)
    q_norm = np.sqrt(q0[i + 1] ** 2 + q1[i + 1] ** 2 + q2[i + 1] ** 2 + q3[i + 1] ** 2)
    q0[i + 1] /= q_norm
    q1[i + 1] /= q_norm
    q2[i + 1] /= q_norm
    q3[i + 1] /= q_norm

# --- Convert Quaternions to Euler Angles (Radians) ---
roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2))
pitch = np.arcsin(np.clip(2 * (q0 * q2 - q3 * q1), -1, 1))  # Added clip for safety
yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))

plt.figure(figsize=(12, 7))
plt.grid(True, alpha=0.3)

# Convert to degrees for easier intuition (optional but recommended)
plt.plot(time_t, np.degrees(roll), label='Roll (phi)')
plt.plot(time_t, np.degrees(pitch), label='Pitch (theta)')
plt.plot(time_t, np.degrees(yaw), label='Yaw (psi)')

plt.axhline(0, color='black', lw=1, alpha=0.5)
plt.title('Projectile Attitude (Euler Angles)', fontsize=14, fontweight='bold')
plt.xlabel('Time (sec)')
plt.ylabel('Degrees (deg)')
plt.legend()
plt.xlim(time_t[0], 900)  # Using your 900s limit
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(time_t, np.degrees(pitch), label='Pitch (Stability Check)', color='orange')
plt.xlim(tlaunch - 2.0, tlaunch + 0.5)  # Look at the 2 seconds before launch
plt.title('Attitude Drift Check (Pre-Launch)')
plt.ylabel('Degrees')
plt.grid(True)
plt.legend()
plt.show()

# 1. Create the figure and the first axis (Pitch)
A_mag_t = truncate(A_mag, time, tlaunch - buffer, tland + buffer)
fig, ax1 = plt.subplots(figsize=(12, 6))

color_pitch = 'tab:blue'
ax1.set_xlabel('Time (sec)')
ax1.set_ylabel('Pitch (degrees)', color=color_pitch, fontsize=12, fontweight='bold')
ax1.plot(time_t, np.degrees(pitch), color=color_pitch, label='Pitch Angle')
ax1.tick_params(axis='y', labelcolor=color_pitch)
ax1.grid(True, alpha=0.3)

# 2. Create a second axis that shares the same x-axis (Acceleration)
ax2 = ax1.twinx()

color_accel = 'tab:red'
ax2.set_ylabel('Total Accel (m/s^2)', color=color_accel, fontsize=12, fontweight='bold')
ax2.plot(time_t, A_mag_t, color=color_accel, alpha=0.6, label='Total Acceleration')  # Ensure A_mag is truncated
ax2.tick_params(axis='y', labelcolor=color_accel)

# 3. Formatting
plt.title('Physics Sanity Check: Pitch vs. Launch Acceleration', pad=15)
plt.xlim(time_t[0], 900)  # Your requested limit
fig.tight_layout()
plt.show()

# notice it doesn't pitch until reaches the apogee

print("--- Attitude Estimation Complete ---")

plt.figure()
plt.grid()
plt.xlabel('Time (sec)')
plt.ylabel('Euler Angles (rad)')
plt.plot(time_t, roll, label='Roll')
plt.plot(time_t, pitch, label='Pitch')
plt.plot(time_t, yaw, label='Yaw')
plt.legend()
plt.show()

# 5.) Transform Body Accel to Inertial Frame
Ax_I_raw = np.zeros_like(Ax_final)
Ay_I_raw = np.zeros_like(Ay_final)
Az_I_raw = np.zeros_like(Az_final)

for i in range(len(Ax_final)):
    ab = np.array([Ax_final[i], Ay_final[i], Az_final[i]])
    phi, theta, psi = roll[i], pitch[i], yaw[i]

    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cs, ss = np.cos(psi), np.sin(psi)

    # Rotation Matrix: Body to Inertial
    TIB = np.array([
        [ct * cs, sp * st * cs - cp * ss, cp * st * cs + sp * ss],
        [ct * ss, sp * st * ss + cp * cs, cp * st * ss - cp * cs],
        [-st, sp * ct, cp * ct]
    ])

    aI = TIB @ ab
    Ax_I_raw[i], Ay_I_raw[i], Az_I_raw[i] = aI

# --- THE DYNAMIC FIX ---
# Calculate the 'Inertial Gravity' observed while sitting on the pad
pre_launch_mask = (time_t < tlaunch)

if np.any(pre_launch_mask):
    # This finds exactly what gravity looks like in your current coordinate setup
    grav_x = np.mean(Ax_I_raw[pre_launch_mask])
    grav_y = np.mean(Ay_I_raw[pre_launch_mask])
    grav_z = np.mean(Az_I_raw[pre_launch_mask])

    print(f"Detected Inertial Gravity: X={grav_x:.2f}, Y={grav_y:.2f}, Z={grav_z:.2f}")

    # Subtract the observed gravity from the entire flight
    Ax_I = Ax_I_raw - grav_x
    Ay_I = Ay_I_raw - grav_y
    Az_I = Az_I_raw - grav_z
else:
    # Fallback if no pre-launch data
    Ax_I, Ay_I, Az_I = Ax_I_raw, Ay_I_raw, (Az_I_raw - GEARTH)

# --- Visualization for Step 5 ---
plt.figure(figsize=(12, 6))
plt.grid(True, alpha=0.3)
plt.title('Inertial Acceleration (Gravity Subtracted)', fontweight='bold')
plt.xlabel('Time (sec)')
plt.ylabel('Acceleration (m/s^2)')

plt.plot(time_t, Ax_I, label='Ax (Inertial)')
plt.plot(time_t, Ay_I, label='Ay (Inertial)')
plt.plot(time_t, Az_I, label='Az (Inertial)')

plt.xlim(time_t[0], 900)  # Apply your 900s limit
plt.legend()
plt.show()

##################################################################
# --- 6.) FINAL DOUBLE INTEGRATION (Trapezoidal) ---
num_pts = len(time_t)
vx = np.zeros(num_pts)
vy = np.zeros(num_pts)
vz = np.zeros(num_pts)
px = np.zeros(num_pts)
py = np.zeros(num_pts)
pz = np.zeros(num_pts)

for i in range(1, num_pts):
    dt = time_t[i] - time_t[i - 1]

    # Velocity Integration
    vx[i] = vx[i - 1] + 0.5 * (Ax_I[i] + Ax_I[i - 1]) * dt
    vy[i] = vy[i - 1] + 0.5 * (Ay_I[i] + Ay_I[i - 1]) * dt
    vz[i] = vz[i - 1] + 0.5 * (Az_I[i] + Az_I[i - 1]) * dt

    # Position Integration
    px[i] = px[i - 1] + 0.5 * (vx[i] + vx[i - 1]) * dt
    py[i] = py[i - 1] + 0.5 * (vy[i] + vy[i - 1]) * dt
    pz[i] = pz[i - 1] + 0.5 * (vz[i] + vz[i - 1]) * dt

# Calculate Total Altitude (3D Distance from Start)
# This handles the fact that your "Up" is split between Y and Z
altitude = np.sqrt(px ** 2 + py ** 2 + pz ** 2)

print(f"Max Altitude Calculated: {np.max(altitude):.2f} meters")

# --- Final Trajectory Plot ---
plt.figure(figsize=(10, 6))
plt.plot(time_t, altitude, label='Total Altitude', color='black', linewidth=2)
plt.plot(time_t, pz, label='Position Z', alpha=0.7)
plt.plot(time_t, py, label='Position Y', alpha=0.7)
plt.title('Reconstructed Flight Altitude', fontweight='bold')
plt.ylabel('Meters (m)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)
plt.xlim(time_t[0], 900)
plt.ylim(-200, 1000)
plt.show()

# --- 6.5) Drift Compensation ---

# 1. Identify when the rocket has landed
# (We use the tland we detected earlier)
landed_mask = (time_t > tland)

if np.any(landed_mask):
    # Find the 'fake' velocity the rocket has at the moment it landed
    vx_drift = vx[time_t > tland][0]
    vy_drift = vy[time_t > tland][0]
    vz_drift = vz[time_t > tland][0]

    # Linearly remove the velocity drift
    # (Assuming the bias error was constant over the flight)
    t_duration = time_t - time_t[0]
    total_time = tland - time_t[0]

    # This creates a correction factor that grows over time
    vx_corr = vx - (vx_drift * (t_duration / total_time))
    vy_corr = vy - (vy_drift * (t_duration / total_time))
    vz_corr = vz - (vz_drift * (t_duration / total_time))

    # Re-integrate the corrected velocity to get corrected position
    px_f = np.zeros_like(px)
    py_f = np.zeros_like(py)
    pz_f = np.zeros_like(pz)

    for i in range(1, len(time_t)):
        dt = time_t[i] - time_t[i - 1]
        px_f[i] = px_f[i - 1] + vx_corr[i] * dt
        py_f[i] = py_f[i - 1] + vy_corr[i] * dt
        pz_f[i] = pz_f[i - 1] + vz_corr[i] * dt

        # Force position to stay constant after landing
        if time_t[i] > tland:
            px_f[i] = px_f[i - 1]
            py_f[i] = py_f[i - 1]
            pz_f[i] = pz_f[i - 1]

    # Update the altitude variable for plotting
    altitude_corrected = np.sqrt(px_f ** 2 + py_f ** 2 + pz_f ** 2)
else:
    altitude_corrected = altitude
    px_f, py_f, pz_f = px, py, pz

print("--- Drift Compensation Applied ---")

# --- 7. FINAL VISUALIZATION ---

# Calculate the Magnitude of the horizontal distance from the start
horizontal_dist = np.sqrt(px_f ** 2 + py_f ** 2)

plt.figure(figsize=(14, 10))

# Plot 1: Altitude (Vertical Performance)
plt.subplot(2, 1, 1)
# Use pz_f (Inertial Z) for altitude
plt.plot(time_t, pz_f, color='blue', linewidth=2.5, label='Vertical Position (Z)')

# Corrected line: use pz_f instead of pz_t
plt.fill_between(time_t, pz_f, color='blue', alpha=0.1)

plt.axvline(tlaunch, color='green', linestyle='--', alpha=0.7, label='Launch')
plt.axvline(tland, color='red', linestyle='--', alpha=0.7, label='Landing')

plt.title('Reconstructed Flight Trajectory (Drift Corrected)', fontsize=14, fontweight='bold')
plt.ylabel('Altitude (meters)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.xlim(time_t[0], 900)

# Plot 2: Horizontal Movement
plt.subplot(2, 1, 2)
plt.plot(time_t, horizontal_dist, color='purple', label='Total Horizontal Distance')
plt.plot(time_t, px_f, label='X Position (Inertial)', alpha=0.5)
plt.plot(time_t, py_f, label='Y Position (Inertial)', alpha=0.5)

plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Ground Distance (meters)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(loc='upper left')
plt.xlim(time_t[0], 900)

plt.tight_layout()
plt.show()

# --- 3D Flight Path ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(px_f, py_f, pz_f, label='3D Path', color='darkorange', lw=2)
ax.scatter(px_f[0], py_f[0], pz_f[0], color='green', s=100, label='Launch Pad')
ax.scatter(px_f[-1], py_f[-1], pz_f[-1], color='red', s=100, label='Final Position (900s)')

ax.set_xlabel('Inertial X (m)')
ax.set_ylabel('Inertial Y (m)')
ax.set_zlabel('Inertial Z (m)')
plt.title('3D Projectile Displacement')
plt.legend()
plt.show()

# show the sensor acceleration clipped
plt.figure(figsize=(12, 5))
plt.plot(time_t, Ax_t, label='Raw Ax (Noisy)', alpha=0.4)
plt.plot(time_t, Ax_f, label='Filtered Ax', linewidth=2)
plt.xlim(tlaunch - 0.5, tlaunch + 5.0)  # Zoom into the burn
plt.title('Sensor Health Check: Looking for Clipping or Aliasing')
plt.ylabel('m/s^2')
plt.legend()
plt.show()
