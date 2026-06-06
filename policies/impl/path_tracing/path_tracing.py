"""Policy for ``task: path_tracing`` — gripper TCP traces all 12 edges of a virtual cube.

Algorithm
---------
At episode start (step 0) the policy builds a precomputed joint-space plan (``_q_plan``)
and every subsequent step returns ``_q_plan[step]``, clamped at the last frame.

The plan is built in **two IK phases**, solved in sequence:

Phase 1 — Approach
  Linear interpolation of tool0 position from the episode-start pose to ``CUBE_CORNER0_WORLD``.
  Orientation is held fixed at the episode-start value throughout both phases.
  Duration: ``APPROACH_DURATION_S``  (→ ``n_approach = round(APPROACH_DURATION_S / control_dt)`` steps).

Phase 2 — Cube trace
  16-segment connected path that covers all 12 cube edges (4 edges repeated for continuity):
    0→1→2→3→0→4→5→6→7→4→0→3→7→6→2→1→5
  Duration: ``CUBE_DURATION_S``  (→ ``n_trace = round(CUBE_DURATION_S / control_dt)`` steps).

The trace IK is warm-started from the approach endpoint, giving < 1.2 mm position tracking error.
At default ``control_dt = 0.02 s``: 125 + 375 = 500 steps total — fits the default sim window.

Cube geometry
-------------
  Corner i = ``CUBE_CORNER0_WORLD + CUBE_SIDE_M * _UNIT_CORNERS[i]``
  Axes aligned with world frame (+X, +Y, +Z):
    Bottom (z=0): 0=(0,0,0)  1=(1,0,0)  2=(1,1,0)  3=(0,1,0)
    Top    (z=1): 4=(0,0,1)  5=(1,0,1)  6=(1,1,1)  7=(0,1,1)
  All 8 corners within 0.72 m of the UR5e base (reach ≈ 0.85 m); min z ≥ 0.20 m.

Requires ``pip install -e ".[ik"]``.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from robot_manipulation_sim.env import UR5GripperEnv
from robot_manipulation_sim.ik import check_ik_dependencies
from robot_manipulation_sim.ik.tool_pose import tool0_se3_matrix

# ---------------------------------------------------------------------------
# Cube geometry parameters
# ---------------------------------------------------------------------------

# World-frame position of cube corner 0.
# The cube extends CUBE_SIDE_M in world +X, +Y, +Z from this point.
# Chosen so all 8 corners are well within the UR5e workspace and the approach
# distance from the home configuration is ≈ 0.28 m (a gentle straight-line lift).
CUBE_CORNER0_WORLD: np.ndarray = np.array([-0.45, -0.25, 0.20], dtype=np.float64)

# Cube edge length [m].
CUBE_SIDE_M: float = 0.15

# ---------------------------------------------------------------------------
# Timing parameters
# ---------------------------------------------------------------------------

# Approach: straight-line move from home tool0 pose to cube corner 0.
APPROACH_DURATION_S: float = 2.5   # → 125 steps at control_dt = 0.02 s

# Cube trace: follow the full 16-segment edge path.
CUBE_DURATION_S: float = 7.5       # → 375 steps at control_dt = 0.02 s
                                    # Total 500 steps = default simulation window.

# ---------------------------------------------------------------------------
# IK parameters
# ---------------------------------------------------------------------------

IK_INNER_ITERS: int = 1   # 1 iter = incremental tracking mode; prevents the "converge-and-jump"
                           # problem where 30 iters fully converge each step to a potentially
                           # far-away IK branch, creating 1.5+ rad joint jumps between steps.
IK_POSITION_COST: float = 2.0
IK_ORIENTATION_COST: float = 0.05  # lower than default → tighter position tracking (< 1.2 mm)
IK_FRAME_LM_DAMPING: float = 0.35
IK_POSTURE_COST: float = 0.1      # raised from 0.034; with rolling posture (prev step's q as ref)
                                   # a stronger posture anchor discourages per-step null-space drift
IK_DAMPING_COST: float = 2.5e-3

# ---------------------------------------------------------------------------
# Cube corner layout (unit cube; scale by CUBE_SIDE_M, offset by CUBE_CORNER0_WORLD)
# ---------------------------------------------------------------------------
_UNIT_CORNERS: np.ndarray = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],   # bottom face  (z=0)
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],   # top face     (z=1)
], dtype=np.float64)

# Connected path through all 12 cube edges.
# Each consecutive pair differs in exactly one coordinate (valid cube edge).
# Unique edges: (0,1)(1,2)(2,3)(0,3)(0,4)(4,5)(5,6)(6,7)(4,7)(3,7)(2,6)(1,5) — all 12.
# Repeated edges for path continuity: (0,4)(0,3)(6,7)(1,2).
_PATH_CORNER_INDICES: tuple[int, ...] = (0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 0, 3, 7, 6, 2, 1, 5)

# ---------------------------------------------------------------------------
# Episode state (cleared by reset())
# ---------------------------------------------------------------------------
_episode_start_ctrl: list[float] | None = None
_latch_qpos: np.ndarray | None = None
_q_plan: np.ndarray | None = None   # (n_approach + n_trace, nq) — precomputed IK trajectory


def reset() -> None:
    """Called by ``simulate_policy`` before each episode."""
    global _episode_start_ctrl, _latch_qpos, _q_plan
    _episode_start_ctrl = None
    _latch_qpos = None
    _q_plan = None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def cube_corners_world() -> np.ndarray:
    """Return (8, 3) world-frame positions of the cube corners."""
    return CUBE_CORNER0_WORLD[np.newaxis] + CUBE_SIDE_M * _UNIT_CORNERS


def _positions_to_se3(positions: np.ndarray, R_fixed: np.ndarray) -> np.ndarray:
    """Convert (N, 3) positions array to (N, 4, 4) SE(3) targets with fixed rotation R_fixed."""
    n = len(positions)
    T = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    T[:, :3, :3] = R_fixed[np.newaxis]
    T[:, :3, 3] = positions
    return T


def _approach_se3_targets(
    p_start: np.ndarray,
    R_fixed: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """(n_steps, 4, 4) SE3 targets: linearly interpolate position from p_start to CUBE_CORNER0_WORLD.

    Parameterised as t ∈ (0, 1]; the last step lands exactly on CUBE_CORNER0_WORLD.
    """
    t = np.arange(1, n_steps + 1, dtype=np.float64) / n_steps
    positions = p_start[np.newaxis] + t[:, np.newaxis] * (CUBE_CORNER0_WORLD - p_start)
    return _positions_to_se3(positions, R_fixed)


def _cube_trace_se3_targets(R_fixed: np.ndarray, n_steps: int) -> np.ndarray:
    """(n_steps, 4, 4) SE3 targets: uniformly-timed path along the 16-segment cube edge path.

    Time t ∈ (0, n_segs] is sampled at equal intervals; each cube segment gets equal time.
    The last step lands exactly on the final path corner.
    """
    corners = cube_corners_world()
    n_segs = len(_PATH_CORNER_INDICES) - 1
    t = np.arange(1, n_steps + 1, dtype=np.float64) * n_segs / n_steps
    positions = np.empty((n_steps, 3), dtype=np.float64)
    for i, ti in enumerate(t):
        seg = min(int(ti), n_segs - 1)
        alpha = ti - seg                          # ∈ [0, 1)
        pa = corners[_PATH_CORNER_INDICES[seg]]
        pb = corners[_PATH_CORNER_INDICES[seg + 1]]
        positions[i] = pa + alpha * (pb - pa)
    return _positions_to_se3(positions, R_fixed)


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------

def _actuator_qadr(env: UR5GripperEnv, act_idx: int) -> int:
    jid = int(env.model.actuator_trnid[act_idx, 0])
    return int(env.model.jnt_qposadr[jid])


def _build_episode_latch(obs: dict[str, Any], env: UR5GripperEnv) -> list[float]:
    """Episode-start ctrl target for every actuator (joint qpos for position actuators)."""
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


# ---------------------------------------------------------------------------
# Policy entry point
# ---------------------------------------------------------------------------

def policy(obs: dict[str, Any], step: int, env: UR5GripperEnv) -> np.ndarray:
    """Step 0: solve two-phase IK plan. Every step: return precomputed q_plan[step]."""
    check_ik_dependencies()
    global _episode_start_ctrl, _latch_qpos, _q_plan

    if step == 0 or _episode_start_ctrl is None:
        _episode_start_ctrl = _build_episode_latch(obs, env)
        _latch_qpos = np.asarray(obs["qpos"], dtype=np.float64).copy()

        # Forward kinematics at episode start → tool0 pose
        data0 = mujoco.MjData(env.model)
        data0.qpos[:] = obs["qpos"]
        mujoco.mj_forward(env.model, data0)
        T0 = tool0_se3_matrix(env.model, data0)
        p0 = T0[:3, 3].copy()
        R0 = T0[:3, :3].copy()

        dt = float(env.control_dt)
        n_approach = max(1, round(APPROACH_DURATION_S / dt))
        n_trace = max(1, round(CUBE_DURATION_S / dt))

        ik = _make_ik_service(env)

        # Phase 1 — approach: home → cube corner 0
        # Fixed posture = home qpos anchors the approach against drifting to far-away IK branches.
        approach_targets = _approach_se3_targets(p0, R0, n_approach)
        q_approach = ik.track_tool_se3_trajectory(
            _latch_qpos, approach_targets, _latch_qpos, rolling_posture=True
        )

        # Phase 2 — cube trace: rolling posture (each step anchors to previous q).
        # This prevents per-corner null-space drift that would otherwise cause large joint jumps.
        trace_targets = _cube_trace_se3_targets(R0, n_trace)
        q_trace = ik.track_tool_se3_trajectory(
            q_approach[-1], trace_targets, q_approach[-1], rolling_posture=True
        )

        _q_plan = np.concatenate([q_approach, q_trace], axis=0)  # (n_approach + n_trace, nq)

    assert _q_plan is not None and _latch_qpos is not None and _episode_start_ctrl is not None

    lo = env.model.actuator_ctrlrange[:, 0]
    hi = env.model.actuator_ctrlrange[:, 1]
    ctrl = np.empty(env.nu, dtype=np.float64)

    k = min(step, len(_q_plan) - 1)
    q_cmd = _latch_qpos.copy()
    q_cmd[:6] = _q_plan[k, :6]   # first 6 DOFs are the UR5e arm joints

    for i in range(env.nu):
        if int(env.model.actuator_trntype[i]) == 0:
            qadr = _actuator_qadr(env, i)
            ctrl[i] = float(np.clip(float(q_cmd[qadr]), lo[i], hi[i]))
        else:
            ctrl[i] = float(np.clip(_episode_start_ctrl[i], lo[i], hi[i]))
    return ctrl
