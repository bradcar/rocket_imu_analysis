# animate_projectile.py
"""
Animate projectile with vpython
"""

import os
from math import pi

from vpython import sphere, box, arrow, color, rate, scene, label

from mylib.quaternion_functions import *

# --- CONSTANTS & UTILITIES ---
G_EARTH = 9.81
frames_directory = "video"


def animate_projectile(time_t, px_f, py_f, pz_f, vz, q, ax_f):
    """
     VPython: 3D Rocket Launch Visualization

    :param time_t: time stamp at position i
    :param px_f: position in X
    :param py_f: position in Y
    :param pz_f: position in Z
    :param vz: vertical velocity
    :param q: quaternion for body
    :param ax_f: accleration up axis before rotation
    :return:
    """

    # set up ability to capture png outputs
    os.makedirs("video", exist_ok=True)
    scene.capture("frame.png")

    # Set scene
    scene.title = "Rocket Launch Simulation"

    # light gray
    scene.background = vector(0.8, 0.8, 0.8)

    # pixels
    scene.width = 800
    scene.height = 600

    # Camera settings
    scene.camera.pos = vector(-150, -145, 32)
    scene.camera.axis = vector(150, 145, 75)
    scene.up = vector(0, 0, 1)

    # World Reference orintation
    world_x = vector(1, 0, 0)
    world_y = vector(0, 1, 0)
    world_z = vector(0, 0, 1)

    # Ground plane
    ground = box(pos=vector(0, 0, 0), size=vector(200, 200, 1), color=color.green, opacity=0.5)

    # World axes for reference
    x_axis = arrow(pos=vector(0, 0, 0), axis=vector(50, 0, 0), shaftwidth=0.8, color=color.red)
    y_axis = arrow(pos=vector(0, 0, 0), axis=vector(0, 50, 0), shaftwidth=0.8, color=color.green)
    z_axis = arrow(pos=vector(0, 0, 0), axis=vector(0, 0, 50), shaftwidth=0.8, color=color.blue)
    label(pos=x_axis.pos + x_axis.axis, text="X", color=color.red, box=False, height=12)
    label(pos=y_axis.pos + y_axis.axis, text="Y", color=color.green, box=False, height=12)

    # Rocket body
    rocket = sphere(pos=vector(px_f[0], py_f[0], pz_f[0]), radius=3, color=color.blue, make_trail=True,
                    trail_color=color.yellow, retain=500)

    # Rocket Arrows
    arrow_scale = 25
    arrow_body_x = arrow(pos=rocket.pos, axis=arrow_scale * world_x, shaftwidth=1.5, color=color.red)
    arrow_body_y = arrow(pos=rocket.pos, axis=arrow_scale * world_y, shaftwidth=1.5, color=color.green)
    arrow_body_z = arrow(pos=rocket.pos, axis=arrow_scale * world_z, shaftwidth=1.5, color=color.blue)

    # Numerical simulation quantity display of Altitude, timestamp, and velocity
    data_label = label(pos=vector(100, -50, 25), text="", color=color.black, box=True, background=vector(0.9, 0.9, 0.9),
                       opacity=1.0, height=12)
    omega_label = label(pos=vector(-50, 100, 25), text="", color=color.black, box=True,
                        background=vector(0.9, 0.9, 0.9),
                        opacity=1.0, height=12)

    # --- Animation loop ---
    for i in range(len(px_f)):
        rate(2)

        # Update rocket position
        rocket.pos = vector(px_f[i], py_f[i], pz_f[i])

        # Relative quaternion from initial orientation
        q_rel = quaternion_multiply(quaternion_conjugate(q[0]), q[i])

        # Rotate world axes to body axes
        body_x = quaternion_rotate(q_rel, world_x)
        body_y = quaternion_rotate(q_rel, world_y)
        body_z = quaternion_rotate(q_rel, world_z)

        # Attach arrows to rocket
        arrow_body_x.pos = rocket.pos
        arrow_body_y.pos = rocket.pos
        arrow_body_z.pos = rocket.pos

        # Update arrow orientation
        arrow_body_x.axis = arrow_scale * body_x
        arrow_body_y.axis = arrow_scale * body_y
        arrow_body_z.axis = arrow_scale * body_z

        # Numerical simulation quantity update for Altitude, timestamp, and velocity
        flight_time = time_t[i] - time_t[0]
        data_label.text = (
            f"Alt={pz_f[i]:.1f} m\n"
            f"t={flight_time:.1f} s\n"
            f"v={vz[i] + G_EARTH:.1f} m/s"
        )

        # Calc Angular velocity from quaternion change, change to degree/sec
        rad2deg = 180.0 / pi
        if i > 0:
            dt = time_t[i] - time_t[i - 1]
            wx, wy, wz = quaternion_to_omega(q[i - 1], q[i], dt)
            wx *= rad2deg
            wy *= rad2deg
            wz *= rad2deg
        else:
            wx, wy, wz = 0.0, 0.0, 0.0

        omega_label.text = (
            "deg/s (world)\n"
            f"x° = {wx:.0f}°/s\n"
            f"y° = {wy:.0f}°/s\n"
            f"z° = {wz:.0f}°/s"
        )

        # Trail effect changes
        # Thick red trail during rocket burn: detected by < 4 sec and high acceleration (>0.5 m/s^2)
        if flight_time < 4 and ax_f[i] > 0.5:
            rocket.trail_color = color.red
            rocket.trail_radius = 2.0
        # Coasting phase shown in thin yellow trail
        else:
            rocket.trail_color = color.yellow
            rocket.trail_radius = 0.5

        # Save png of each from frame
        scene.capture(f"frame_{i:04d}")
