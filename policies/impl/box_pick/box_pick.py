"""Simple box-pick policy using linear Cartesian IK phase targets."""

from __future__ import annotations

import logging
from typing import Any

import mujoco
import numpy as np

from robot_manipulation_sim.env import UR5GripperEnv
from robot_manipulation_sim.ik import check_ik_dependencies
from robot_manipulation_sim.ik.tool_pose import tool0_se3_matrix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enable wrist RGB in the rollout video (obs["images"] is read by this policy
# indirectly via the vision service's detection camera render).
# ---------------------------------------------------------------------------
ENABLE_RGB: bool = True

# ---------------------------------------------------------------------------
# Box geometry / detection parameters
# ---------------------------------------------------------------------------

# Box centre height [m] above the table at rest.
# MJCF: box geom size="0.025 0.04 0.025" → half-height = 0.025 m.  The box is
# placed at z=0.035 in the scene XML but the physics engine settles it to
# z ≈ 0.025 m (half-height = resting centre).  Use 0.025 for IK and detection.
BOX_CENTER_Z: float = 0.025

# HSV colour range for the orange box (OpenCV scale: H÷2, S/V in [0, 255]).
# MJCF box colour: rgba(0.85, 0.55, 0.2, 1) ≈ RGB(217, 140, 51).
# H≈25°/2≈12.5 in OpenCV scale; broad range to handle lighting variation.
BOX_HSV_LOWER: np.ndarray = np.array([8, 80, 80], dtype=np.float32)
BOX_HSV_UPPER: np.ndarray = np.array([35, 255, 255], dtype=np.float32)

# Minimum connected-pixel area to accept a detection [pixels²].
BOX_MIN_DETECTION_AREA: int = 80

# Camera used for box detection.
DETECTION_CAMERA: str = "topdown"
DETECTION_WIDTH: int = 640
DETECTION_HEIGHT: int = 480

# ---------------------------------------------------------------------------
# Grasp orientation
# ---------------------------------------------------------------------------

# Target tool0 rotation for a top-down grasp.
# Derivation: at home, body-Z = world-Y (approach dir).  For a top-down grasp we need:
#   body-Z → world-[0, 0, -1]  (approach from above)
#   body-X → world-[1, 0, 0]   (finger-spread along world X; fits 5 cm box face)
#   body-Y = body-Z × body-X → world-[0, -1, 0]
# R[i,j] = i-th world component of j-th body axis (MuJoCo xmat convention).
R_GRASP: np.ndarray = np.array(
    [
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ],
    dtype=np.float64,
)

# Measured distance from tool0 origin to follower (fingertip) body in world -Z [m].
# Measured by FK: at any arm config with body-Z = world -Z (R_GRASP), the
# left/right follower bodies are consistently ~0.115 m below tool0 in world Z.
GRIPPER_FINGER_OFFSET_M: float = 0.115

# ---------------------------------------------------------------------------
# Z heights for each phase [m] — these are tool0 body frame origin in world Z.
# With R_GRASP, fingertips are GRIPPER_FINGER_OFFSET_M below tool0 (world -Z).
#   fingertip_z = tool0_z - GRIPPER_FINGER_OFFSET_M
# So tool0_z = target_fingertip_z + GRIPPER_FINGER_OFFSET_M.
# ---------------------------------------------------------------------------

# Grasp height: tool0 positioned so fingertips reach box centre (z=BOX_CENTER_Z).
# fingertips at 0.025 m + driver bodies at 0.025 + 0.066 = 0.091 m → clears box top 0.050 m.
Z_GRASP: float = BOX_CENTER_Z + GRIPPER_FINGER_OFFSET_M   # = 0.140 m

# Pre-grasp clearance above box for safe lateral approach.
Z_PRE_GRASP: float = Z_GRASP + 0.12   # = 0.260 m

# Post-lift height for transporting the box.
Z_LIFT: float = 0.42   # well above any table clutter

# ---------------------------------------------------------------------------
# Placement target
# ---------------------------------------------------------------------------

# World-frame (x, y) to place the box.  Chosen inside the UR5e workspace but
# clearly separated from the nominal box start zone (x≈0.52, y≈0.0±0.04).
PLACE_XY: np.ndarray = np.array([0.30, 0.35], dtype=np.float64)

# ---------------------------------------------------------------------------
# Phase durations [s] → steps = round(duration / control_dt)
# Total default: 12 s = 600 steps at control_dt = 0.02 s.
# ---------------------------------------------------------------------------
T_CRANE: float = 2.0      # phase 0 crane settle (qpos[0] → home ctrl-target joints)
T_PRE: float = 3.0        # phase 1 approach (crane pose → pre-grasp above box)
T_DESCEND: float = 1.5    # phase 2 descend
T_GRIP: float = 0.6       # phase 3 grip (stationary, gripper closes)
T_LIFT: float = 1.5       # phase 4 lift
T_TRANSPORT: float = 2.0  # phase 5 transport
T_LOWER: float = 1.5      # phase 6 lower
T_RELEASE: float = 0.8    # phase 7 release (stationary, gripper opens)

# ---------------------------------------------------------------------------
# IK parameters (same philosophy as path_tracing: incremental tracking)
# ---------------------------------------------------------------------------
IK_INNER_ITERS: int = 1           # incremental; prevent jump-and-converge artefacts
IK_POSITION_COST: float = 2.0
IK_ORIENTATION_COST: float = 0.8  # higher than path_tracing — orientation matters for grasping
IK_FRAME_LM_DAMPING: float = 0.35
IK_POSTURE_COST: float = 0.1
IK_DAMPING_COST: float = 2.5e-3

# ---------------------------------------------------------------------------
# Gripper commands
# ---------------------------------------------------------------------------
# Do **not** infer open/close from XML gain/bias alone — the settled finger geometry is
# measured in ``tests/test_gripper_control_finger_geometry.py``. For the bundled scene:
#   low ``ctrl`` (≈0)  → wider pad opening (release / spread),
#   high ``ctrl`` (≈255) → narrower opening (clamp / grasp).
# Default ``UR5GripperEnv._home`` uses ``0`` so the episode starts physically consistent.
GRIPPER_RELEASE: float = 0.0    # settled: maximum pad spread in the safe low-ctrl band
GRIPPER_GRIP: float = 255.0     # settled: minimum pad spread (fingers closed on object)

# Aliases used in the gripper plan (kept for readability in build_plan).
GRIPPER_OPEN = GRIPPER_RELEASE
GRIPPER_CLOSED = GRIPPER_GRIP

# ---------------------------------------------------------------------------
# Episode state (cleared by reset())
# ---------------------------------------------------------------------------
_episode_start_ctrl: list[float] | None = None
_latch_qpos: np.ndarray | None = None
_q_plan: np.ndarray | None = None        # (N, nq) arm joint trajectory
_gripper_plan: np.ndarray | None = None  # (N,)   gripper command per step
_detection_ok: bool = False


def reset() -> None:
    """Called by ``simulate_policy`` before each episode."""
    global _episode_start_ctrl, _latch_qpos, _q_plan, _gripper_plan, _detection_ok
    _episode_start_ctrl = None
    _latch_qpos = None
    _q_plan = None
    _gripper_plan = None
    _detection_ok = False


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _make_se3(pos: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Build a 4×4 SE(3) matrix from a 3-vector position and 3×3 rotation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def _interpolate_se3_targets(
    p_start: np.ndarray,
    p_end: np.ndarray,
    R_fixed: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Return ``(n_steps, 4, 4)`` SE3 targets: linearly interpolate position, fixed rotation.

    ``t ∈ (0, 1]`` so the last step lands exactly on ``p_end``.
    """
    t = np.arange(1, n_steps + 1, dtype=np.float64) / n_steps
    positions = p_start[np.newaxis] + t[:, np.newaxis] * (p_end - p_start)
    n = len(positions)
    T = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    T[:, :3, :3] = R_fixed
    T[:, :3, 3] = positions
    return T


# ---------------------------------------------------------------------------
# Helpers (shared with other policies)
# ---------------------------------------------------------------------------

def _actuator_qadr(env: UR5GripperEnv, act_idx: int) -> int:
    jid = int(env.model.actuator_trnid[act_idx, 0])
    return int(env.model.jnt_qposadr[jid])


def _build_episode_latch(obs: dict[str, Any], env: UR5GripperEnv) -> list[float]:
    """Episode-start ctrl target for every actuator."""
    lat: list[float] = []
    for i in range(env.nu):
        if int(env.model.actuator_trntype[i]) == 0:
            lat.append(float(obs["qpos"][_actuator_qadr(env, i)]))
        else:
            lat.append(float(obs["ctrl"][i]))
    return lat


def _make_ik_service(env: UR5GripperEnv) -> Any:
    from robot_manipulation_sim.ik.service import MujocoMinkIkService  # noqa: PLC0415

    return MujocoMinkIkService(
        env.model,
        inner_iters=IK_INNER_ITERS,
        control_dt=float(env.control_dt),
        position_cost=IK_POSITION_COST,
        orientation_cost=IK_ORIENTATION_COST,
        frame_lm_damping=IK_FRAME_LM_DAMPING,
        posture_cost=IK_POSTURE_COST,
        damping_cost=IK_DAMPING_COST,
    )


def _gripper_idx(env: UR5GripperEnv) -> int:
    """Index of the gripper (tendon-type) actuator."""
    for i in range(env.nu):
        if int(env.model.actuator_trntype[i]) != 0:
            return i
    raise RuntimeError("No tendon actuator found in model (gripper).")


# ---------------------------------------------------------------------------
# Box detection
# ---------------------------------------------------------------------------

def detect_box_world_xy(
    env: UR5GripperEnv,
    obs: dict[str, Any],
) -> np.ndarray | None:
    """Detect the orange box in the topdown camera and return its (x, y, z) world position.

    Uses HSV colour detection via :class:`VisionService`.  Returns ``None`` if no
    detection passes the minimum area threshold.
    """
    from robot_manipulation_sim.vision.service import VisionService  # noqa: PLC0415

    svc = VisionService(
        env.model,
        camera_name=DETECTION_CAMERA,
        width=DETECTION_WIDTH,
        height=DETECTION_HEIGHT,
    )
    image = svc.render_detection_image(env.data)
    detections = svc.detect_by_color(
        image,
        BOX_HSV_LOWER,
        BOX_HSV_UPPER,
        min_area=BOX_MIN_DETECTION_AREA,
    )
    if not detections:
        return None
    best = detections[0]  # already sorted by area descending
    try:
        world_pos = svc.unproject_pixel_to_world(env.data, best.center_uv, target_z=BOX_CENTER_Z)
    except ValueError as exc:
        logger.warning("box_pick: back-projection failed: %s", exc)
        return None
    return world_pos


# ---------------------------------------------------------------------------
# Plan builder
# ---------------------------------------------------------------------------

def build_plan(
    env: UR5GripperEnv,
    obs: dict[str, Any],
    box_world_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the full 8-phase arm-joint + gripper plan.

    Parameters
    ----------
    env:
        Live environment (model + fresh data).
    obs:
        Step-0 observation (must contain ``qpos`` and ``ctrl``).
    box_world_pos:
        3-element world-frame position of the box (X, Y, Z) from detection.

    Returns
    -------
    q_plan : (N, nq)
        Full joint-position trajectory.
    gripper_plan : (N,)
        Gripper command for each step.

    Notes
    -----
    After ``env.reset()`` the arm joint positions (``obs["qpos"][:6]``) are all zero
    (the MJCF default) while the position-actuator targets (``obs["ctrl"][:6]``) are
    the home values ``[-π/2, -π/2, π/2, -π/2, -π/2, 0]``.  At those home ctrl-target
    joints the UR5e tool0 is already in the ``R_GRASP`` orientation (body-Z = world −Z).
    Phase 0 linearly interpolates the joint targets from all-zero to the home ctrl-target
    ("crane pose") over ``T_CRANE`` seconds so the actuators drive the arm there smoothly.
    All subsequent IK phases then start from the crane pose, which is already in
    ``R_GRASP`` orientation, so no configuration branching / folding occurs.
    """
    check_ik_dependencies()

    latch_qpos = np.asarray(obs["qpos"], dtype=np.float64)
    ctrl0 = np.asarray(obs["ctrl"], dtype=np.float64)
    dt = float(env.control_dt)

    # Step counts for each phase.
    n_crane = max(1, round(T_CRANE / dt))
    n_pre = max(1, round(T_PRE / dt))
    n_desc = max(1, round(T_DESCEND / dt))
    n_grip = max(1, round(T_GRIP / dt))
    n_lift = max(1, round(T_LIFT / dt))
    n_trans = max(1, round(T_TRANSPORT / dt))
    n_lower = max(1, round(T_LOWER / dt))
    n_rel = max(1, round(T_RELEASE / dt))

    # Crane qpos: arm joints at ctrl-target values.  At this configuration the
    # tool0 body-Z is already world -Z (R_GRASP), so all subsequent IK phases
    # operate in the correct orientation half-space without branch switching.
    q_crane = latch_qpos.copy()
    q_crane[:6] = ctrl0[:6]

    # FK to recover the crane tool0 position (used as the IK approach start).
    data_crane = mujoco.MjData(env.model)
    data_crane.qpos[:] = q_crane
    mujoco.mj_forward(env.model, data_crane)
    T_crane_fk = tool0_se3_matrix(env.model, data_crane)
    p_crane = T_crane_fk[:3, 3].copy()

    # Target positions for each phase transition.
    bx, by = float(box_world_pos[0]), float(box_world_pos[1])
    px, py = float(PLACE_XY[0]), float(PLACE_XY[1])

    p_pregrasp = np.array([bx, by, Z_PRE_GRASP], dtype=np.float64)
    p_grasp = np.array([bx, by, Z_GRASP], dtype=np.float64)
    p_lift = np.array([bx, by, Z_LIFT], dtype=np.float64)
    p_trans = np.array([px, py, Z_LIFT], dtype=np.float64)
    p_lower = np.array([px, py, Z_GRASP], dtype=np.float64)

    ik = _make_ik_service(env)

    # Phase 0: crane settle — linearly interpolate arm joints from all-zero qpos
    # to the home ctrl-target joints.  The position actuators track this smoothly.
    t_ramp = np.arange(1, n_crane + 1, dtype=np.float64) / n_crane
    q_settle = latch_qpos[np.newaxis, :] + t_ramp[:, np.newaxis] * (q_crane - latch_qpos)[np.newaxis, :]

    # Phase 1: crane → pre-grasp.  Both ends already have R_GRASP orientation,
    # so incremental IK stays in the unfolded (crane) configuration branch.
    tgts_pre = _interpolate_se3_targets(p_crane, p_pregrasp, R_GRASP, n_pre)
    q_pre = ik.track_tool_se3_trajectory(q_crane, tgts_pre, q_crane, rolling_posture=True)

    # Phase 2: pre-grasp → grasp height.
    tgts_desc = _interpolate_se3_targets(p_pregrasp, p_grasp, R_GRASP, n_desc)
    q_desc = ik.track_tool_se3_trajectory(q_pre[-1], tgts_desc, q_pre[-1], rolling_posture=True)

    # Phase 3: stationary at grasp (gripper closes) — repeat last q.
    q_grip = np.tile(q_desc[-1], (n_grip, 1))

    # Phase 4: lift up.
    tgts_lift = _interpolate_se3_targets(p_grasp, p_lift, R_GRASP, n_lift)
    q_lift = ik.track_tool_se3_trajectory(q_desc[-1], tgts_lift, q_desc[-1], rolling_posture=True)

    # Phase 5: transport to placement XY.
    tgts_trans = _interpolate_se3_targets(p_lift, p_trans, R_GRASP, n_trans)
    q_trans = ik.track_tool_se3_trajectory(q_lift[-1], tgts_trans, q_lift[-1], rolling_posture=True)

    # Phase 6: lower to place height.
    tgts_lower = _interpolate_se3_targets(p_trans, p_lower, R_GRASP, n_lower)
    q_lower = ik.track_tool_se3_trajectory(q_trans[-1], tgts_lower, q_trans[-1], rolling_posture=True)

    # Phase 7: stationary at place (gripper opens) — repeat last q.
    q_rel = np.tile(q_lower[-1], (n_rel, 1))

    q_plan = np.concatenate([q_settle, q_pre, q_desc, q_grip, q_lift, q_trans, q_lower, q_rel], axis=0)

    # Build gripper plan.  Empirical mapping (same MJCF): low ``ctrl`` → wider pad opening,
    # high ``ctrl`` → narrower — see ``tests/test_gripper_control_finger_geometry.py``.
    # Fingers stay fully open through crane/approach/descent; ramp to close at grasp height.
    g_crane = np.full(n_crane, GRIPPER_RELEASE)                   # fingers wide open during settle
    g_pre = np.full(n_pre, GRIPPER_RELEASE)                       # fingers wide open during approach
    g_desc = np.full(n_desc, GRIPPER_RELEASE)                     # fingers wide open during descent
    g_grip = np.linspace(GRIPPER_RELEASE, GRIPPER_GRIP, n_grip)  # actuator closes at grasp height
    g_lift = np.full(n_lift, GRIPPER_GRIP)                        # actuator holds grip during lift
    g_trans = np.full(n_trans, GRIPPER_GRIP)                      # actuator holds grip during transport
    g_lower = np.full(n_lower, GRIPPER_GRIP)                      # actuator holds grip during lower
    g_rel = np.linspace(GRIPPER_GRIP, GRIPPER_RELEASE, n_rel)    # ramp to low ctrl = release
    gripper_plan = np.concatenate([g_crane, g_pre, g_desc, g_grip, g_lift, g_trans, g_lower, g_rel])

    assert len(q_plan) == len(gripper_plan), "q_plan and gripper_plan must have the same length"
    return q_plan, gripper_plan


# ---------------------------------------------------------------------------
# Policy entry point
# ---------------------------------------------------------------------------

def policy(obs: dict[str, Any], step: int, env: UR5GripperEnv) -> np.ndarray:
    """Step 0: detect box, build IK plan. Every step: return precomputed ctrl."""
    check_ik_dependencies()
    global _episode_start_ctrl, _latch_qpos, _q_plan, _gripper_plan, _detection_ok

    if step == 0 or _episode_start_ctrl is None:
        _episode_start_ctrl = _build_episode_latch(obs, env)
        _latch_qpos = np.asarray(obs["qpos"], dtype=np.float64).copy()

        box_pos = detect_box_world_xy(env, obs)
        if box_pos is None:
            logger.warning(
                "box_pick: no box detected — arm will stay at home for the full episode."
            )
            _detection_ok = False
            n_total = max(1, round((T_CRANE + T_PRE + T_DESCEND + T_GRIP + T_LIFT + T_TRANSPORT + T_LOWER + T_RELEASE) / float(env.control_dt)))
            _q_plan = np.tile(_latch_qpos, (n_total, 1))
            _gripper_plan = np.full(n_total, GRIPPER_RELEASE)
        else:
            logger.info("box_pick: detected box at world (%.3f, %.3f, %.3f)", *box_pos)
            _detection_ok = True
            _q_plan, _gripper_plan = build_plan(env, obs, box_pos)

    assert _q_plan is not None and _gripper_plan is not None
    assert _episode_start_ctrl is not None and _latch_qpos is not None

    lo = env.model.actuator_ctrlrange[:, 0]
    hi = env.model.actuator_ctrlrange[:, 1]
    ctrl = np.empty(env.nu, dtype=np.float64)

    k = min(step, len(_q_plan) - 1)
    q_cmd = _latch_qpos.copy()
    q_cmd[:6] = _q_plan[k, :6]  # first 6 DOFs are UR5e arm joints

    gripper_cmd = float(_gripper_plan[k])
    g_idx = _gripper_idx(env)

    for i in range(env.nu):
        if i == g_idx:
            ctrl[i] = float(np.clip(gripper_cmd, lo[i], hi[i]))
        elif int(env.model.actuator_trntype[i]) == 0:
            qadr = _actuator_qadr(env, i)
            ctrl[i] = float(np.clip(float(q_cmd[qadr]), lo[i], hi[i]))
        else:
            ctrl[i] = float(np.clip(_episode_start_ctrl[i], lo[i], hi[i]))
    return ctrl
