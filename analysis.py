# Rocket trajectory post-flight analysis.
"""
Rocket trajectory post-flight analysis.

Data flow: raw → truncated → filtered → bias-corrected → inertial.

Cast input data to FP64, Numpy defaults to 64-bit.

NAMING CONVENTION:
Preliminary data procesing
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

SOURCE Ideas:
# https://github.com/cmontalvo251/aerospace/blob/main/rockets/PLAR/post_launch_analysis.py
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
np.set_printoptions(precision=10)

# --- CONSTANTS & UTILITIES ---
G_EARTH = 9.81
filename = "data/launch_data.txt"
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

def add_2d_plot_note(note_text, ax=None, x=0.65, y=0.10, fontsize=9, color='green'):
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
        transform=ax.transAxes,        # Use axis-relative coordinates
        fontsize=fontsize,
        fontstyle='italic',
        color=color,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='green')
    )

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
print(f"Average time step: {dt_avg:.4f} seconds ({dt_avg*1000:.1f} msec)")

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

plt.figure(figsize=(10,4))
plt.plot(time, A_mag, label="Total Acceleration (A_mag)", color="gray", alpha=1.0)
plt.axvline(tlaunch, color="g", linestyle="--", label="Detected Launch")
plt.axvline(tland, color="r", linestyle="--", label="Detected Impact")
plt.title("Step 2: Launch and Impact Detection Check")
plt.ylabel("m/s^2")
plt.legend()
plt.xlim(tlaunch - 0.5, tland + 0.5)
plt.grid(True)
add_2d_plot_note("Actually launch & chute deploy")
plt.savefig(f"{plot_directory}/lauch-chute-plot.pdf")
plt.show()

# Hand-adjusted launch/land times
tlaunch = 874.6
tland = 882.9

print(f"Detected Launch: {tlaunch:.2f}s")
print(f"Detected Chute Deploy: {tland:.2f}s")
print(f"Flight Duration: {tland-tlaunch:.2f}s")

# Truncate data for flight window + buffer
buffer = 1.0
time_t = truncate(time, time, tlaunch - buffer, tland + buffer)
ax_t, ay_t, az_t = [truncate(v, time, tlaunch - buffer, tland + buffer) for v in [ax_b, ay_b, az_b]]
gr_t, gp_t, gy_t = [truncate(v, time, tlaunch - buffer, tland + buffer) for v in [gr_b, gp_b, gy_b]]
A_mag_t = truncate(A_mag, time, tlaunch - buffer, tland + buffer)

plt.figure(figsize=(10,4))
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

# --- 3. FILTERING ---
# only perform simple IIR filter to see the error in the acceleration x-axis
ax_iir = simple_firstorder_iir_filter(ax_t, 0.5)

# More accurate Butterworth filtering
cutoff = sample_frequency/4.0
sos = signal.butter(2, cutoff/(0.5*sample_frequency), btype="lowpass", output="sos")
ax_f, ay_f, az_f = [signal.sosfiltfilt(sos,v) for v in [ax_t, ay_t, az_t]]
gr_f, gp_f, gy_f = [signal.sosfiltfilt(sos,v) for v in [gr_t, gp_t, gy_t]]

plt.figure(figsize=(12,6))
plt.scatter(time_t, ax_t, color="black", s=8, alpha=1.0, label="Raw Data")
plt.plot(time_t, ax_iir, color="red", label="IIR (Lagged)")
plt.plot(time_t, ax_f, color="blue", linewidth=2, label="Butterworth (Zero Phase)")
plt.title("Step 3: Filter Phase-Lag Comparison")
plt.ylabel("m/s^2")
plt.legend()
add_2d_plot_note("IIR lags, only forward looking, use Butterworh filter", x=0.55)
plt.savefig(f"{plot_directory}/iir-buterworth-plot.pdf")
plt.show()

# --- 4. Sensor vs Center of Gravity TRANSLATION & BIAS REMOVAL ---
rx, ry, rz = 0.0, 0.0, 0.0
dt = dt_avg
gr_dot, gp_dot, gy_dot = np.gradient(gr_f, dt), np.gradient(gp_f, dt), np.gradient(gy_f, dt)

ax_cg = ax_f - (-(gp_f**2 + gy_f**2)*rx + (gr_f*gp_f - gy_dot)*ry + (gr_f*gy_f + gp_dot)*rz)
ay_cg = ay_f - ((gr_f*gp_f + gy_dot)*rx - (gr_f**2 + gy_f**2)*ry + (gp_f*gy_f - gr_dot)*rz)
az_cg = az_f - ((gr_f*gy_f - gp_dot)*rx + (gp_f*gy_f + gr_dot)*ry - (gr_f**2 + gp_f**2)*rz)

pre_launch_mask = (time_t < tlaunch)
if np.any(pre_launch_mask):
    ax_bias, ay_bias = np.mean(ax_cg[pre_launch_mask]), np.mean(ay_cg[pre_launch_mask])
    gr_b, gp_b, gy_b = np.mean(gr_f[pre_launch_mask]), np.mean(gp_f[pre_launch_mask]), np.mean(gy_f[pre_launch_mask])

    ax_final, ay_final, az_final = ax_cg - ax_bias, ay_cg - ay_bias, az_cg
    gr_final, gp_final, gy_final = gr_f - gr_b, gp_f - gp_b, gy_f - gy_b
else:
    ax_final, ay_final, az_final = ax_cg, ay_cg, az_cg
    gr_final, gp_final, gy_final = gr_f, gp_f, gy_f

plt.figure(figsize=(10,4))
plt.plot(time_t, ax_f, label="Raw Filtered Ax", alpha=0.5)
plt.plot(time_t, ax_final, label="Bias Corrected Ax", color="blue")
plt.xlim(tlaunch-0.8, tlaunch+0.2)
plt.title("Step 4: Static Bias Correction (Pre-Launch Window) Center Gravity & Bias Correction")
plt.legend()
plt.grid(True)
add_2d_plot_note("orig IMU data should zero bias")
plt.savefig(f"{plot_directory}/imu-bias-plot.pdf")
plt.show()

# --- 5. ATTITUDE ESTIMATION ---
num_pts = len(time_t)
q = np.array([[1.0,0.0,0.0,0.0]]*num_pts, dtype=np.float64)
twoKp = 1.0

for i in range(num_pts-1):
    ax, ay, az = ax_final[i], ay_final[i], az_final[i]
    gr, gp, gy = gr_final[i], gp_final[i], gy_final[i]

    norm = np.sqrt(ax**2 + ay**2 + az**2)
    if norm>0: ax, ay, az = ax/norm, ay/norm, az/norm

    vx = 2.0*(q[i,1]*q[i,3] - q[i,0]*q[i,2])
    vy = 2.0*(q[i,0]*q[i,1] + q[i,2]*q[i,3])
    vz = q[i,0]**2 - 0.5 + q[i,3]**2

    ex, ey, ez = (ay*vz - az*vy), (az*vx - ax*vz), (ax*vy - ay*vx)
    gr, gp, gy = gr + twoKp*ex, gp + twoKp*ey, gy + twoKp*ez

    dt_step = time_t[i+1] - time_t[i]
    dq = 0.5*np.array([
        -q[i,1]*gr - q[i,2]*gp - q[i,3]*gy,
        q[i,0]*gr + q[i,2]*gy - q[i,3]*gp,
        q[i,0]*gp - q[i,1]*gy + q[i,3]*gr,
        q[i,0]*gy + q[i,1]*gp - q[i,2]*gr
    ], dtype=np.float64)
    q[i+1] = q[i] + dq*dt_step
    q[i+1] /= np.linalg.norm(q[i+1])

roll = np.arctan2(2*(q[:,0]*q[:,1] + q[:,2]*q[:,3]), 1-2*(q[:,1]**2 + q[:,2]**2))
pitch = np.arcsin(np.clip(2*(q[:,0]*q[:,2] - q[:,3]*q[:,1]), -1,1))
yaw = np.arctan2(2*(q[:,0]*q[:,3] + q[:,1]*q[:,2]), 1-2*(q[:,2]**2 + q[:,3]**2))

plt.figure(figsize=(10,4))
plt.plot(time_t, np.degrees(pitch), label="Pitch (Degrees)", color="orange")
plt.plot(time_t, np.degrees(roll), label="Roll (Degrees)", alpha=0.5)
plt.title("Step 5: Attitude Estimation (Euler Angles)")
plt.ylabel("Degrees")
plt.legend()
plt.grid(True)
add_2d_plot_note("pitch & roll more as go higher, gimbal flip is discontinuity", x=0.1)
plt.savefig(f"{plot_directory}/roll-pitch-plot.pdf")
plt.show()

# --- 6. INERTIAL TRANSFORM ---
ax_I, ay_I, az_I = np.zeros(num_pts, dtype=np.float64), np.zeros(num_pts, dtype=np.float64), np.zeros(num_pts, dtype=np.float64)
for i in range(num_pts):
    cp, sp, ct, st, cs, ss = np.cos(roll[i]), np.sin(roll[i]), np.cos(pitch[i]), np.sin(pitch[i]), np.cos(yaw[i]), np.sin(yaw[i])
    TIB = np.array([
        [ct*cs, sp*st*cs - cp*ss, cp*st*cs + sp*ss],
        [ct*ss, sp*st*ss + cp*cs, cp*st*ss - sp*cs],
        [-st, sp*ct, cp*ct]
    ], dtype=np.float64)
    ax_I[i], ay_I[i], az_I[i] = TIB @ np.array([ax_final[i], ay_final[i], az_final[i]], dtype=np.float64)

if np.any(pre_launch_mask):
    ax_I -= np.mean(ax_I[pre_launch_mask])
    ay_I -= np.mean(ay_I[pre_launch_mask])
    az_I -= np.mean(az_I[pre_launch_mask])

plt.figure(figsize=(10,4))
plt.plot(time_t, ax_I, label="Ax_Inertial")
plt.plot(time_t, ay_I, label="Ay_Inertial")
plt.plot(time_t, az_I, label="Az_Inertial", linewidth=2)
plt.title("Step 6: Inertial Acceleration (Gravity Subtracted)")
plt.ylabel("m/s^2")
plt.legend()
plt.grid(True)
add_2d_plot_note("Strange: Ay_I goe neg, offset by Ax_I?", x=0.02)
plt.savefig(f"{plot_directory}/inertial-acceleration-plot.pdf")
plt.show()

# --- 7. INTEGRATION & DRIFT COMPENSATION ---
vx, vy, vz = np.zeros(num_pts, dtype=np.float64), np.zeros(num_pts, dtype=np.float64), np.zeros(num_pts, dtype=np.float64)
px, py, pz = np.zeros(num_pts, dtype=np.float64), np.zeros(num_pts, dtype=np.float64), np.zeros(num_pts, dtype=np.float64)

for i in range(1,num_pts):
    dt_step = time_t[i]-time_t[i-1]
    vx[i] = vx[i-1] + 0.5*(ax_I[i]+ax_I[i-1])*dt_step
    vy[i] = vy[i-1] + 0.5*(ay_I[i]+ay_I[i-1])*dt_step
    vz[i] = vz[i-1] + 0.5*(az_I[i]+az_I[i-1])*dt_step
    px[i] = px[i-1] + 0.5*(vx[i]+vx[i-1])*dt_step
    py[i] = py[i-1] + 0.5*(vy[i]+vy[i-1])*dt_step
    pz[i] = pz[i-1] + 0.5*(vz[i]+vz[i-1])*dt_step

t_rel = time_t - time_t[0]
total_t = time_t[-1] - time_t[0]
vx_c, vy_c, vz_c = vx - (vx[-1]*(t_rel/total_t)), vy - (vy[-1]*(t_rel/total_t)), vz - (vz[-1]*(t_rel/total_t))

px_f, py_f, pz_f = np.zeros(num_pts, dtype=np.float64), np.zeros(num_pts, dtype=np.float64), np.zeros(num_pts, dtype=np.float64)
for i in range(1,num_pts):
    dt_step = time_t[i]-time_t[i-1]
    if time_t[i]<tland:
        px_f[i] = px_f[i-1] + vx_c[i]*dt_step
        py_f[i] = py_f[i-1] + vy_c[i]*dt_step
        pz_f[i] = pz_f[i-1] + vz_c[i]*dt_step
    else:
        px_f[i], py_f[i], pz_f[i] = px_f[i-1], py_f[i-1], pz_f[i-1]

# --- 8. FINAL VISUALIZATIONS ---
# Clipping Check
plt.figure(figsize=(10, 4))
plt.plot(time_t, ax_t, alpha=0.3, label="Raw Ax")
plt.plot(time_t, ax_f, color="blue", label="Filtered Ax")
plt.xlim(tlaunch - 0.5, tlaunch + 4.0)
plt.title("Step 8: Sensor Health Check (Thrust Phase Clipping)")
plt.ylabel("m/s^2")
plt.legend()
plt.grid(True)
add_2d_plot_note("We know massive clip at launch, looks minimized here", x=0.1)
plt.savefig(f"{plot_directory}/acceleration-clipping-plot.pdf")
plt.show()

# Stacked plot of altitude and ground drift horizontal
plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.plot(time_t, pz_f, color="blue", lw=2.5, label="Vertical Altitude (Z)")
plt.fill_between(time_t, pz_f, color="blue", alpha=0.1)
plt.axvline(tlaunch, color="g", linestyle="--", label="Detected Launch")
plt.axvline(tland, color="r", linestyle="--", label="Detected Chute")
plt.title("Step 8: Final Drift-Corrected Altitude")
plt.ylabel("Meters (m)")
add_2d_plot_note("likely altitude too low due to acceleration clipping at launch", x=0.4)
plt.legend()
# stacked
plt.subplot(2,1,2)
plt.plot(time_t, np.sqrt(px_f**2 + py_f**2), label="Ground Distance (Horizontal)", color="purple")
plt.title("Step 8: Ground Distance (Horizontal)")
plt.xlabel("Time (s)")
plt.ylabel("Meters (m)")
plt.legend()
plt.tight_layout()
add_2d_plot_note("seems reasonable drift scale", x=0.4)
plt.savefig(f"{plot_directory}/dirft-plot.pdf")
plt.show()

# 3D Plot flight path
# set x-axis and y-axis to same range
x_min, x_max = px_f.min(), px_f.max()
y_min, y_max = py_f.min(), py_f.max()
x_center = 0.5*(x_max+x_min)
y_center = 0.5*(y_max+y_min)
span = max(x_max-x_min, y_max-y_min)
pad = 0.10*span
half = 0.5*span + pad

fig = plt.figure(figsize=(10,8))
ax3d = fig.add_subplot(111, projection="3d")
ax3d.plot(px_f, py_f, pz_f, label="Flight Path", color="darkorange", lw=2)
ax3d.set_title("Step 8: 3D Flight Path - seems reasonable")
ax3d.set_xlabel("X (m)")
ax3d.set_ylabel("Y (m)")
ax3d.set_zlabel("Z (m)")
ax3d.set_xlim(x_center-half, x_center+half)
ax3d.set_ylim(y_center-half, y_center+half)
plt.savefig(f"{plot_directory}/flight-path.pdf")
plt.show()
