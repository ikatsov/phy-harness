"""Policy for ``task: shoulder_rotation`` — TCP follows an explicit 180° hinge arc, IK fills joint targets.

At episode start (step 0):
1. Record ``tool0`` pose and the shoulder-lift hinge axis/pivot from the current ``MjData``.
2. Build the full sequence of ``N`` TCP targets as SE(3) matrices along the arc (closed-form Rodrigues).
3. Solve IK for all targets at once with ``track_tool_se3_trajectory`` → ``q_plan (N, nq)``.

Every subsequent step just reads ``q_plan[step]`` — no per-step IK, no accumulated arc state.

Requires ``pip install -e ".[ik]"``.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from robot_manipulation_sim.env import UR5GripperEnv
from robot_manipulation_sim.ik import check_ik_dependencies
from robot_manipulation_sim.ik.tool_pose import tool0_se3_matrix

SHOULDER_ARC_TOTAL_RAD = math.pi   # full sweep angle (rad)
SHOULDER_ARC_DURATION_S = 4.0      # wall-clock time to complete the arc
_SHOULDER_ARC_SIGN = -1.0           # direction (negative → lift-joint more negative)

SHOULDER_IK_INNER_ITERS = 30

_episode_start_ctrl: list[float] | None = None
_latch_qpos: np.ndarray | None = None
_q_plan: np.ndarray | None = None   # (N, nq) — precomputed IK joint trajectory


def reset() -> None:
    """Optional hook: called by ``simulate_policy`` after each ``env.reset()``."""
    global _episode_start_ctrl, _latch_qpos, _q_plan
    _episode_start_ctrl = None
    _latch_qpos = None
    _q_plan = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _actuator_qadr(env: UR5GripperEnv, act_idx: int) -> int:
    jid = int(env.model.actuator_trnid[act_idx, 0])
    return int(env.model.jnt_qposadr[jid])


def _build_episode_latch(obs: dict[str, Any], env: UR5GripperEnv) -> list[float]:
    """Episode-start ctrl target for every actuator (joint qpos or tendon ctrl)."""
    lat: list[float] = []
    for i in range(env.nu):
        if int(env.model.actuator_trntype[i]) == 0:
            lat.append(float(obs["qpos"][_actuator_qadr(env, i)]))
        else:
            lat.append(float(obs["ctrl"][i]))
    return lat


def _shoulder_lift_hinge(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """World-frame unit axis and pivot point of the shoulder_lift hinge."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_lift_joint")
    if jid < 0:
        raise RuntimeError("MJCF missing joint 'shoulder_lift_joint'")
    bid = int(model.jnt_bodyid[jid])
    R = np.asarray(data.xmat[bid], dtype=np.float64).reshape(3, 3)
    u = R @ np.asarray(model.jnt_axis[jid], dtype=np.float64)
    u /= np.linalg.norm(u)
    pivot = np.asarray(data.xanchor[jid], dtype=np.float64).reshape(3)
    return u, pivot


def _tcp_poses_along_arc(T0: np.ndarray, u: np.ndarray, pivot: np.ndarray,
                          thetas: np.ndarray) -> np.ndarray:
    """Return (N, 4, 4) SE(3) targets: TCP rotated about hinge ``(u, pivot)`` by each angle in ``thetas``.

    Closed-form Rodrigues: R(θ) = I + sin θ K + (1−cos θ) K²
    where K is the skew-symmetric matrix of unit axis ``u``.
    """
    R0 = T0[:3, :3]
    p0 = T0[:3, 3]
    x, y, z = u
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    K2 = K @ K
    poses = np.empty((len(thetas), 4, 4), dtype=np.float64)
    for i, th in enumerate(thetas):
        Rd = np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * K2
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rd @ R0
        T[:3, 3] = pivot + Rd @ (p0 - pivot)
        poses[i] = T
    return poses


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def policy(obs: dict[str, Any], step: int, env: UR5GripperEnv) -> np.ndarray:
    """On step 0: build arc, solve IK plan. Every step: look up pre-solved joint targets."""
    check_ik_dependencies()
    global _episode_start_ctrl, _latch_qpos, _q_plan

    if step == 0 or _episode_start_ctrl is None:
        from robot_manipulation_sim.ik.service import MujocoMinkIkService

        _episode_start_ctrl = _build_episode_latch(obs, env)
        _latch_qpos = np.asarray(obs["qpos"], dtype=np.float64).copy()

        # --- build arc SE(3) sequence ---
        data0 = mujoco.MjData(env.model)
        data0.qpos[:] = obs["qpos"]
        mujoco.mj_forward(env.model, data0)
        T0 = tool0_se3_matrix(env.model, data0)
        u, pivot = _shoulder_lift_hinge(env.model, data0)

        dt = float(env.control_dt)
        N = max(1, round(SHOULDER_ARC_DURATION_S / dt))
        thetas = float(_SHOULDER_ARC_SIGN) * float(SHOULDER_ARC_TOTAL_RAD) * np.arange(1, N + 1) / N
        arc_se3 = _tcp_poses_along_arc(T0, u, pivot, thetas)  # (N, 4, 4)

        # --- solve IK for the whole arc in one call ---
        ik = MujocoMinkIkService(
            env.model,
            inner_iters=SHOULDER_IK_INNER_ITERS,
            control_dt=dt,
            position_cost=2.2,
            orientation_cost=0.22,
            frame_lm_damping=0.42,
            posture_cost=0.034,
            damping_cost=2.5e-3,
        )
        _q_plan = ik.track_tool_se3_trajectory(_latch_qpos, arc_se3, _latch_qpos)  # (N, nq)

    assert _q_plan is not None and _latch_qpos is not None and _episode_start_ctrl is not None

    lo = env.model.actuator_ctrlrange[:, 0]
    hi = env.model.actuator_ctrlrange[:, 1]
    ctrl = np.empty(env.nu, dtype=np.float64)

    k = min(step, len(_q_plan) - 1)
    q_cmd = _latch_qpos.copy()
    q_cmd[:6] = _q_plan[k, :6]

    for i in range(env.nu):
        if int(env.model.actuator_trntype[i]) == 0:
            qadr = _actuator_qadr(env, i)
            ctrl[i] = float(np.clip(float(q_cmd[qadr]), lo[i], hi[i]))
        else:
            ctrl[i] = float(np.clip(_episode_start_ctrl[i], lo[i], hi[i]))
    return ctrl
