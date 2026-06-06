"""Forward kinematics helpers for the end-effector body ``tool0`` (Menagerie UR5e scene)."""

from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import NDArray


def tool0_body_id(model: mujoco.MjModel) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool0")
    if bid < 0:
        raise RuntimeError("MJCF missing body 'tool0'")
    return bid


def tool0_position(model: mujoco.MjModel, data: mujoco.MjData) -> NDArray[np.float64]:
    """World-frame COM position of ``tool0`` (length 3)."""
    bid = tool0_body_id(model)
    return np.asarray(data.xpos[bid], dtype=np.float64).copy()


def tool0_rotation_matrix(model: mujoco.MjModel, data: mujoco.MjData) -> NDArray[np.float64]:
    """World-from-body rotation for ``tool0`` (3×3, row-major layout consistent with ``data.xmat``)."""
    bid = tool0_body_id(model)
    return np.asarray(data.xmat[bid], dtype=np.float64).reshape(3, 3).copy()


def tool0_se3_matrix(model: mujoco.MjModel, data: mujoco.MjData) -> NDArray[np.float64]:
    """4×4 homogeneous transform (world from tool0 body frame)."""
    R = tool0_rotation_matrix(model, data)
    p = tool0_position(model, data)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T
