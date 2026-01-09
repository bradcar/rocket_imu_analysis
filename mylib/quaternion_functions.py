# quaternion_functions.py
"""
Quaternion helper functions - Hamilton names/order r, i, j, k
"""

import numpy as np


def quaternion_rotate(q, v):
    """
    rotate a quaternion
    :param q:
    :param v:
    :return:
    """
    q0, q1, q2, q3 = q
    v = np.asarray(v, dtype=np.float64)
    qv = np.array([q1, q2, q3], dtype=np.float64)

    # Rodrigues-style quaternion rotation
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + q0 * v)


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions
    :param q1:
    :param q2:
    :return:
    """
    r1, i1, j1, k1 = q1
    r2, i2, j2, k2 = q2
    return (
        r1 * r2 - i1 * i2 - j1 * j2 - k1 * k2,
        r1 * i2 + i1 * r2 + j1 * k2 - k1 * j2,
        r1 * j2 - i1 * k2 + j1 * r2 + k1 * i2,
        r1 * k2 + i1 * j2 - j1 * i2 + k1 * r2
    )


def quaternion_conjugate(q):
    """
    Conjugate Quaternion
    :param q:
    :return:
    """
    qr, qi, qj, qk = q
    return qr, -qi, -qj, -qk


def quaternion_normalize(qr: float, qi: float, qj: float, qk: float) -> tuple:
    """
    quaternion_normalize
    :param qr:
    :param qi:
    :param qj:
    :param qk:
    :return:
    """
    mag_sq = qr ** 2 + qi ** 2 + qj ** 2 + qk ** 2
    if mag_sq < 0.000001:  # Check for near-zero magnitude
        return 1.0, 0.0, 0.0, 0.0
    n = np.sqrt(mag_sq)
    return qr / n, qi / n, qj / n, qk / n


def quaternion_to_omega(q_prev, q_curr, dt):
    """
    from two quaternions q_prev and q_curr at a time delta, create omega in rad/s for x, y, z
    :param q_prev: quaternion at previous timestamp
    :param q_curr: quaternion at current timestamp
    :param dt: delta t between timestamps
    :return:
    """
    # Relative rotation quaternion
    dq = quaternion_multiply(quaternion_conjugate(q_prev), q_curr)

    # Normalize & clamp scalar
    mag = (dq[0] ** 2 + dq[1] ** 2 + dq[2] ** 2 + dq[3] ** 2) ** 0.5
    dq = [x / mag for x in dq]
    w = max(-1.0, min(1.0, dq[0]))

    # Rotation angle with
    theta = 2.0 * np.acos(w)
    if theta < 1e-6 or dt <= 0:
        return 0.0, 0.0, 0.0

    # Rotation axis
    s = np.sin(theta / 2.0)
    axis = (dq[1] / s, dq[2] / s, dq[3] / s)

    # Angular velocity (world frame)
    return tuple(theta / dt * a for a in axis)
