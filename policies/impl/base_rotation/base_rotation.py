"""Policy for ``task: base_rotation`` — slow rotation about the robot base (vertical axis).

Paired bundle (same stem ``base_rotation`` under ``policies/impl/base_rotation/``):
  ``base_rotation.yaml`` — ``task_spec`` + ``task_analyzers`` (validation loads it when the main config sets ``task: base_rotation``).
  ``base_rotation.py`` — this module (pass to ``simulate_policy.py``).
  ``joints_csv_base_rotation.py`` — task-specific rollout analyzer (loaded by ``type: joints_csv_base_rotation``).

Uses **differential IK** (``mink`` via ``robot_manipulation_sim.ik``) on ``tool0``: each step applies a small
world-frame rotation about **+Z** through the robot ``base`` body XY position (vertical axis through the
UR mount), matching the nominal ``BASE_ROTATION_SPEED`` tangential rate. Joint **position** ``ctrl`` is
filled from the IK configuration (same actuator interface as before).

Non-arm ``qpos`` (gripper + box) follow the posture reference taken from the episode latch. Requires
``pip install -e ".[ik]"``.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from robot_manipulation_sim.env import UR5GripperEnv
from robot_manipulation_sim.ik import check_ik_dependencies
from robot_manipulation_sim.ik.tool_pose import tool0_se3_matrix

# rad/s nominal rotation about world +Z through the base (same scale as the legacy pan-only slew).
BASE_ROTATION_SPEED = 0.7
# Differential IK substeps per control tick (trade accuracy vs CPU).
BASE_ROTATION_IK_INNER_ITERS = 24

_episode_start_ctrl: list[float] | None = None
_pan_actuator: int | None = None
_pan_qadr: int | None = None
_latch_qpos: np.ndarray | None = None
_axis_xy: tuple[float, float] | None = None
_ik_svc: Any = None
_ik_model_id: int | None = None


def reset() -> None:
    """Optional hook: ``simulate_policy`` calls ``reset()`` after ``env.reset()`` each episode."""
    global _episode_start_ctrl, _pan_qadr, _latch_qpos, _axis_xy, _ik_svc, _ik_model_id
    _episode_start_ctrl = None
    _pan_qadr = None
    _latch_qpos = None
    _axis_xy = None
    _ik_svc = None
    _ik_model_id = None


def _shoulder_pan_actuator(env: UR5GripperEnv) -> int:
    global _pan_actuator
    if _pan_actuator is None:
        aid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_shoulder_pan")
        if aid < 0:
            raise RuntimeError("MJCF missing position actuator 'a_shoulder_pan' (shoulder pan)")
        _pan_actuator = aid
    return _pan_actuator


def _actuator_qadr(env: UR5GripperEnv, act_idx: int) -> int:
    jid = int(env.model.actuator_trnid[act_idx, 0])
    return int(env.model.jnt_qposadr[jid])


def _build_episode_latch(obs: dict[str, Any], env: UR5GripperEnv) -> list[float]:
    lat: list[float] = []
    for i in range(env.nu):
        if int(env.model.actuator_trntype[i]) == 0:  # mjTRN_JOINT
            qadr = _actuator_qadr(env, i)
            lat.append(float(obs["qpos"][qadr]))
        else:
            lat.append(float(obs["ctrl"][i]))
    return lat


def _Rz(theta: float) -> np.ndarray:
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rotate_point_about_vertical_axis(p: np.ndarray, ax: float, ay: float, theta: float) -> np.ndarray:
    """Rotate ``p`` about world +Z through vertical line ``(ax, ay, *)``."""
    qx, qy = float(p[0]) - ax, float(p[1]) - ay
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([ax + c * qx - s * qy, ay + s * qx + c * qy, float(p[2])], dtype=np.float64)


def _get_ik_service(env: UR5GripperEnv) -> Any:
    global _ik_svc, _ik_model_id
    mid = id(env.model)
    if _ik_svc is None or _ik_model_id != mid:
        from robot_manipulation_sim.ik.service import MujocoMinkIkService

        _ik_svc = MujocoMinkIkService(
            env.model,
            inner_iters=BASE_ROTATION_IK_INNER_ITERS,
            control_dt=float(env.control_dt),
            position_cost=2.0,
            orientation_cost=0.18,
            frame_lm_damping=0.32,
            posture_cost=0.025,
            damping_cost=2e-3,
        )
        _ik_model_id = mid
    return _ik_svc


def policy(obs: dict[str, Any], step: int, env: UR5GripperEnv) -> np.ndarray:
    """One control step: IK on ``tool0`` for a small world-Z rotation; ``ctrl`` = joint targets from IK."""
    check_ik_dependencies()
    global _episode_start_ctrl, _pan_qadr, _latch_qpos, _axis_xy

    active = _shoulder_pan_actuator(env)
    lo = env.model.actuator_ctrlrange[:, 0]
    hi = env.model.actuator_ctrlrange[:, 1]
    ctrl = np.empty(env.nu, dtype=np.float64)
    pan_step = BASE_ROTATION_SPEED * float(env.control_dt)

    if step == 0 or _episode_start_ctrl is None:
        _episode_start_ctrl = _build_episode_latch(obs, env)
        _pan_qadr = int(_actuator_qadr(env, active))
        _latch_qpos = np.asarray(obs["qpos"], dtype=np.float64).copy()
        data0 = mujoco.MjData(env.model)
        data0.qpos[:] = obs["qpos"]
        mujoco.mj_forward(env.model, data0)
        bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if bid < 0:
            raise RuntimeError("MJCF missing body 'base' (needed for pan rotation axis)")
        _axis_xy = (float(data0.xpos[bid, 0]), float(data0.xpos[bid, 1]))

    if _latch_qpos is None or _axis_xy is None or _pan_qadr is None:
        raise RuntimeError("base_rotation: internal latch not initialized")

    data = mujoco.MjData(env.model)
    data.qpos[:] = obs["qpos"]
    mujoco.mj_forward(env.model, data)
    T_cur = tool0_se3_matrix(env.model, data)
    ax, ay = _axis_xy
    p_new = _rotate_point_about_vertical_axis(T_cur[:3, 3], ax, ay, pan_step)
    R_new = _Rz(pan_step) @ T_cur[:3, :3]
    T_tgt = np.eye(4, dtype=np.float64)
    T_tgt[:3, :3] = R_new
    T_tgt[:3, 3] = p_new

    posture = _latch_qpos.copy()
    posture[_pan_qadr] = float(obs["qpos"][_pan_qadr])

    ik = _get_ik_service(env)
    q_new = ik.step_toward_tool_se3(np.asarray(obs["qpos"], dtype=np.float64), T_tgt, posture)
    # Joint-space latch: only ``shoulder_pan`` may deviate from the episode start; other arm joints stay
    # at their latched angles (same contract as the pre-IK policy). Pan value comes from the IK solution.
    q_cmd = _latch_qpos.copy()
    q_cmd[_pan_qadr] = float(q_new[_pan_qadr])

    for i in range(env.nu):
        if int(env.model.actuator_trntype[i]) == 0:
            qadr = _actuator_qadr(env, i)
            ctrl[i] = float(np.clip(float(q_cmd[qadr]), lo[i], hi[i]))
        else:
            ctrl[i] = float(np.clip(_episode_start_ctrl[i], lo[i], hi[i]))
    return ctrl
