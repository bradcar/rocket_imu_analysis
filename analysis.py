import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# https://github.com/cmontalvo251/aerospace/blob/main/rockets/PLAR/post_launch_analysis.py

# --- CONSTANTS & UTILITIES ---
GEARTH = 9.81
filename = 'data/launch_data.txt'

def truncate(outvec, timevec, ts, te):
    """Crops the vector to the specified time window using a boolean mask."""
    mask = (timevec > ts) & (timevec < te)
    return outvec[mask]


def simple_firstorder_IIR_filter(vec, tau):
    """Manual low-pass filter (for comparison purposes)."""
    outvec = np.zeros_like(vec)
    outvec[0] = vec[0]
    for x in range(1, len(vec)):
        outvec[x] = (1 - tau) * outvec[x - 1] + tau * vec[x]
    return outvec


# --- 1. DATA LOADING & ANALYSIS ---
data = np.loadtxt(filename)
time = data[:, 0]
Ax, Ay, Az = data[:, 1], data[:, 2], data[:, 3]
gx, gy, gz = data[:, 4], data[:, 5], data[:, 6]

# Calculate Magnitude and Sampling Stats
A_mag = np.sqrt(Ax ** 2 + Ay ** 2 + Az ** 2)
dt_avg = np.mean(np.diff(time))
sample_frequency = 1.0 / dt_avg

print(f"File: {filename}")
print(f"Sampling Freq: {sample_frequency:.2f} Hz")
print(f"Average time step: {dt_avg:.4f} seconds ({dt_avg * 1000.:.1f} msec)")

# --- 2. ENHANCED EVENT DETECTION ---
# Find Launch: first time exceeding 3.5G
launch_indices = np.where(A_mag > 35.0)[0]
if len(launch_indices) == 0:
    tlaunch = time[0]
else:
    tlaunch = time[launch_indices[0]]

# Find Impact: highest G-force at least 2 seconds after launch
post_launch_mask = (time > tlaunch + 2.0)
if np.any(post_launch_mask):
    impact_idx = np.argmax(A_mag[post_launch_mask])
    tland = time[post_launch_mask][impact_idx]
else:
    tland = time[-1]

print(f"Detected Launch: {tlaunch:.2f}s")
print(f"Detected Impact: {tland:.2f}s")
print(f"Flight Duration: {tland - tlaunch:.2f}s")

# Plot Detection Check
plt.figure(figsize=(10, 4))
plt.plot(time, A_mag, label='Total Acceleration (A_mag)', color='gray', alpha=0.5)
plt.axvline(tlaunch, color='g', linestyle='--', label='Detected Launch')
plt.axvline(tland, color='r', linestyle='--', label='Detected Impact')
plt.title('Step 1: Launch and Impact Detection Check')
plt.ylabel('m/s^2')
plt.legend()
plt.xlim(tlaunch - 5, tland + 5)
plt.grid(True)
plt.show()

# Truncate data for flight window + buffer
buffer = 1.0
time_t = truncate(time, time, tlaunch - buffer, tland + buffer)
Ax_t, Ay_t, Az_t = [truncate(v, time, tlaunch - buffer, tland + buffer) for v in [Ax, Ay, Az]]
gx_t, gy_t, gz_t = [truncate(v, time, tlaunch - buffer, tland + buffer) for v in [gx, gy, gz]]
A_mag_t = truncate(A_mag, time, tlaunch - buffer, tland + buffer)

# --- 3. FILTERING ---
cutoff = sample_frequency / 4.0
b, a = signal.butter(2, cutoff / (0.5 * sample_frequency), btype='low')

# Calculate various filters for comparison plot
Ax_iir = simple_firstorder_IIR_filter(Ax_t, 0.5)
Ax_f = signal.filtfilt(b, a, Ax_t)
Ay_f, Az_f = signal.filtfilt(b, a, Ay_t), signal.filtfilt(b, a, Az_t)
gx_f, gy_f, gz_f = [signal.filtfilt(b, a, v) for v in [gx_t, gy_t, gz_t]]

# Plot Filter Comparison
plt.figure(figsize=(12, 6))
plt.scatter(time_t, Ax_t, color='black', s=8, alpha=0.2, label='Raw Data')
plt.plot(time_t, Ax_iir, color='red', label='IIR (Lagged)')
plt.plot(time_t, Ax_f, color='blue', linewidth=2, label='Butterworth (Zero Phase)')
plt.title('Step 2: Filter Phase-Lag Comparison')
plt.ylabel('m/s^2')
plt.legend()
plt.show()

# --- 4. CG TRANSLATION & BIAS REMOVAL ---
# Note: rx, ry, rz set to 0.0 as per previous setup
rx, ry, rz = 0.0, 0.0, 0.0
dt = dt_avg
gx_dot, gy_dot, gz_dot = np.gradient(gx_f, dt), np.gradient(gy_f, dt), np.gradient(gz_f, dt)

Ax_cg = Ax_f - (-(gy_f ** 2 + gz_f ** 2) * rx + (gx_f * gy_f - gz_dot) * ry + (gx_f * gz_f + gy_dot) * rz)
Ay_cg = Ay_f - ((gx_f * gy_f + gz_dot) * rx - (gx_f ** 2 + gz_f ** 2) * ry + (gy_f * gz_f - gx_dot) * rz)
Az_cg = Az_f - ((gx_f * gz_f - gy_dot) * rx + (gy_f * gz_f + gx_dot) * ry - (gx_f ** 2 + gy_f ** 2) * rz)

# Bias Removal
pre_launch_mask = (time_t < tlaunch)
if np.any(pre_launch_mask):
    Ax_bias, Ay_bias = np.mean(Ax_cg[pre_launch_mask]), np.mean(Ay_cg[pre_launch_mask])
    gx_b, gy_b, gz_b = np.mean(gx_f[pre_launch_mask]), np.mean(gy_f[pre_launch_mask]), np.mean(gz_f[pre_launch_mask])

    Ax_final, Ay_final, Az_final = Ax_cg - Ax_bias, Ay_cg - Ay_bias, Az_cg
    gx_final, gy_final, gz_final = gx_f - gx_b, gy_f - gy_b, gz_f - gz_b
else:
    Ax_final, Ay_final, Az_final = Ax_cg, Ay_cg, Az_cg
    gx_final, gy_final, gz_final = gx_f, gy_f, gz_f

# Bias Comparison Plot
plt.figure(figsize=(10, 4))
plt.plot(time_t, Ax_f, label='Raw Filtered Ax', alpha=0.5)
plt.plot(time_t, Ax_final, label='Bias Corrected Ax', color='blue')
plt.xlim(tlaunch - 0.8, tlaunch + 0.2)
plt.title('Step 3: Static Bias Correction (Pre-Launch Window)')
plt.legend()
plt.grid(True)
plt.show()

# --- 5. ATTITUDE ESTIMATION ---
num_pts = len(time_t)
q = np.array([[1.0, 0.0, 0.0, 0.0]] * num_pts)
twoKp = 1.0

for i in range(num_pts - 1):
    ax, ay, az = Ax_final[i], Ay_final[i], Az_final[i]
    gx, gy, gz = gx_final[i], gy_final[i], gz_final[i]

    norm = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    if norm > 0: ax, ay, az = ax / norm, ay / norm, az / norm

    vx = 2.0 * (q[i, 1] * q[i, 3] - q[i, 0] * q[i, 2])
    vy = 2.0 * (q[i, 0] * q[i, 1] + q[i, 2] * q[i, 3])
    vz = q[i, 0] ** 2 - 0.5 + q[i, 3] ** 2

    ex, ey, ez = (ay * vz - az * vy), (az * vx - ax * vz), (ax * vy - ay * vx)
    gx, gy, gz = gx + twoKp * ex, gy + twoKp * ey, gz + twoKp * ez

    dt_step = time_t[i + 1] - time_t[i]
    dq = 0.5 * np.array([
        -q[i, 1] * gx - q[i, 2] * gy - q[i, 3] * gz,
        q[i, 0] * gx + q[i, 2] * gz - q[i, 3] * gy,
        q[i, 0] * gy - q[i, 1] * gz + q[i, 3] * gx,
        q[i, 0] * gz + q[i, 1] * gy - q[i, 2] * gx
    ])
    q[i + 1] = q[i] + dq * dt_step
    q[i + 1] /= np.linalg.norm(q[i + 1])

roll = np.arctan2(2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
pitch = np.arcsin(np.clip(2 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1]), -1, 1))
yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))

# Euler Angles Plot
plt.figure(figsize=(10, 4))
plt.plot(time_t, np.degrees(pitch), label='Pitch (Degrees)', color='orange')
plt.plot(time_t, np.degrees(roll), label='Roll (Degrees)', alpha=0.5)
plt.title('Step 4: Attitude Estimation (Euler Angles)')
plt.ylabel('Degrees')
plt.legend()
plt.grid(True)
plt.show()

# --- 6. INERTIAL TRANSFORM ---
Ax_I, Ay_I, Az_I = np.zeros(num_pts), np.zeros(num_pts), np.zeros(num_pts)
for i in range(num_pts):
    cp, sp, ct, st, cs, ss = np.cos(roll[i]), np.sin(roll[i]), np.cos(pitch[i]), np.sin(pitch[i]), np.cos(
        yaw[i]), np.sin(yaw[i])
    TIB = np.array([
        [ct * cs, sp * st * cs - cp * ss, cp * st * cs + sp * ss],
        [ct * ss, sp * st * ss + cp * cs, cp * st * ss - cp * cs],
        [-st, sp * ct, cp * ct]
    ])
    Ax_I[i], Ay_I[i], Az_I[i] = TIB @ np.array([Ax_final[i], Ay_final[i], Az_final[i]])

# Dynamic Gravity Subtraction
if np.any(pre_launch_mask):
    Ax_I -= np.mean(Ax_I[pre_launch_mask])
    Ay_I -= np.mean(Ay_I[pre_launch_mask])
    Az_I -= np.mean(Az_I[pre_launch_mask])

# Inertial Plot
plt.figure(figsize=(10, 4))
plt.plot(time_t, Ax_I, label='Ax_Inertial')
plt.plot(time_t, Ay_I, label='Ay_Inertial')
plt.plot(time_t, Az_I, label='Az_Inertial', linewidth=2)
plt.title('Step 5: Inertial Acceleration (Gravity Subtracted)')
plt.ylabel('m/s^2')
plt.legend()
plt.grid(True)
plt.show()

# --- 7. INTEGRATION & DRIFT COMPENSATION ---
vx, vy, vz = np.zeros(num_pts), np.zeros(num_pts), np.zeros(num_pts)
px, py, pz = np.zeros(num_pts), np.zeros(num_pts), np.zeros(num_pts)

for i in range(1, num_pts):
    dt_step = time_t[i] - time_t[i - 1]
    vx[i] = vx[i - 1] + 0.5 * (Ax_I[i] + Ax_I[i - 1]) * dt_step
    vy[i] = vy[i - 1] + 0.5 * (Ay_I[i] + Ay_I[i - 1]) * dt_step
    vz[i] = vz[i - 1] + 0.5 * (Az_I[i] + Az_I[i - 1]) * dt_step
    px[i] = px[i - 1] + 0.5 * (vx[i] + vx[i - 1]) * dt_step
    py[i] = py[i - 1] + 0.5 * (vy[i] + vy[i - 1]) * dt_step
    pz[i] = pz[i - 1] + 0.5 * (vz[i] + vz[i - 1]) * dt_step

# Final Drift Neutralization
t_rel = time_t - time_t[0]
total_t = time_t[-1] - time_t[0]
vx_c, vy_c, vz_c = vx - (vx[-1] * (t_rel / total_t)), vy - (vy[-1] * (t_rel / total_t)), vz - (
            vz[-1] * (t_rel / total_t))

px_f, py_f, pz_f = np.zeros(num_pts), np.zeros(num_pts), np.zeros(num_pts)
for i in range(1, num_pts):
    dt_step = time_t[i] - time_t[i - 1]
    if time_t[i] < tland:
        px_f[i] = px_f[i - 1] + vx_c[i] * dt_step
        py_f[i] = py_f[i - 1] + vy_c[i] * dt_step
        pz_f[i] = pz_f[i - 1] + vz_c[i] * dt_step
    else:
        px_f[i], py_f[i], pz_f[i] = px_f[i - 1], py_f[i - 1], pz_f[i - 1]

# --- 8. FINAL VISUALIZATIONS ---
# Trajectory Plot
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(time_t, pz_f, color='blue', lw=2.5, label='Vertical Altitude (Z)')
plt.fill_between(time_t, pz_f, color='blue', alpha=0.1)
plt.axvline(tlaunch, color='g', linestyle='--')
plt.axvline(tland, color='r', linestyle='--')
plt.title('Step 6: Final Drift-Corrected Altitude')
plt.ylabel('Meters (m)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_t, np.sqrt(px_f ** 2 + py_f ** 2), label='Ground Distance (Horizontal)', color='purple')
plt.xlabel('Time (s)')
plt.ylabel('Meters (m)')
plt.legend()
plt.tight_layout()
plt.show()

# 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(px_f, py_f, pz_f, label='Flight Path', color='darkorange', lw=2)
ax.set_title('Step 7: 3D Flight Path')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
plt.show()

# Clipping Check
plt.figure(figsize=(10, 4))
plt.plot(time_t, Ax_t, alpha=0.3, label='Raw Ax')
plt.plot(time_t, Ax_f, color='blue', label='Filtered Ax')
plt.xlim(tlaunch - 0.5, tlaunch + 4.0)
plt.title('Step 8: Sensor Health Check (Thrust Phase Clipping)')
plt.ylabel('m/s^2')
plt.legend()
plt.show()