"""Policy for ``task: inspection`` — auto-generate a scan path from mesh feature edges.

At episode start this policy:
1. Settles the arm into a "crane/home" pose (same idea as box_pick) to avoid folded IK branches.
2. Extracts feature edges directly from the inspected mesh (boundary or high-dihedral edges).
3. Filters to accessible upper edges and builds an ordered scan toolpath from these edge points.
4. Solves IK once and replays cached controls.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import mujoco
import numpy as np

from robot_manipulation_sim.env import UR5GripperEnv
from robot_manipulation_sim.ik import check_ik_dependencies
from robot_manipulation_sim.ik.tool_pose import tool0_se3_matrix

# Scan geometry parameters.
INSPECTION_GEOM_NAME: str = "inspection_bumper_geom"
EDGE_DIHEDRAL_DEG_MIN: float = 35.0
EDGE_LENGTH_MIN_M: float = 0.008
SCAN_WAYPOINTS: int = 24
ACCESS_Z_QUANTILE_MIN: float = 0.82
ACCESS_Z_MAX_M: float = 0.55
ACCESS_XY_RADIUS_MIN_M: float = 0.20
ACCESS_XY_RADIUS_MAX_M: float = 0.75

# Keep gripper tip above detected edge points by a small hover gap [m].
TIP_HOVER_M: float = 0.04
# tool0->fingertip distance when scan orientation is top-down.
GRIPPER_FINGER_OFFSET_M: float = 0.115
# Keep backwards-compatible name used by tests/metrics as the *effective tool0 stand-off*.
SCAN_STANDOFF_M: float = TIP_HOVER_M + GRIPPER_FINGER_OFFSET_M

# Timing.
CRANE_DURATION_S: float = 1.0
APPROACH_DURATION_S: float = 1.2
SCAN_SEGMENT_DURATION_S: float = 0.5

# IK tuning (tracking mode, similar to path_tracing/box_pick).
IK_INNER_ITERS: int = 1
IK_POSITION_COST: float = 2.0
IK_ORIENTATION_COST: float = 0.45
IK_FRAME_LM_DAMPING: float = 0.35
IK_POSTURE_COST: float = 0.08
IK_DAMPING_COST: float = 2.5e-3

# Gripper command: keep closed during inspection so fingertip pose is stable.
GRIPPER_CLOSED_CTRL: float = 255.0

# Top-down scan orientation (same convention as box_pick).
R_SCAN: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)

_episode_start_ctrl: list[float] | None = None
_latch_qpos: np.ndarray | None = None
_q_plan: np.ndarray | None = None
_scan_targets_world: np.ndarray | None = None


def reset() -> None:
    """Called by ``simulate_policy`` before each episode."""
    global _episode_start_ctrl, _latch_qpos, _q_plan, _scan_targets_world
    _episode_start_ctrl = None
    _latch_qpos = None
    _q_plan = None
    _scan_targets_world = None


def _actuator_qadr(env: UR5GripperEnv, act_idx: int) -> int:
    jid = int(env.model.actuator_trnid[act_idx, 0])
    return int(env.model.jnt_qposadr[jid])


def _build_episode_latch(obs: dict[str, Any], env: UR5GripperEnv) -> list[float]:
    lat: list[float] = []
    for i in range(env.nu):
        if int(env.model.actuator_trntype[i]) == 0:
            lat.append(float(obs["qpos"][_actuator_qadr(env, i)]))
        else:
            lat.append(float(obs["ctrl"][i]))
    return lat


def _gripper_idx(env: UR5GripperEnv) -> int:
    for i in range(env.nu):
        if int(env.model.actuator_trntype[i]) != 0:
            return i
    raise RuntimeError("No tendon actuator found in model (gripper).")


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


def _inspection_mesh_local(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, INSPECTION_GEOM_NAME)
    if gid < 0:
        raise RuntimeError(f"inspection geom {INSPECTION_GEOM_NAME!r} not found")
    mid = int(model.geom_dataid[gid])
    if mid < 0:
        raise RuntimeError(f"inspection geom {INSPECTION_GEOM_NAME!r} has no mesh data")
    va = int(model.mesh_vertadr[mid])
    vn = int(model.mesh_vertnum[mid])
    fa = int(model.mesh_faceadr[mid])
    fn = int(model.mesh_facenum[mid])
    verts = np.asarray(model.mesh_vert[va : va + vn], dtype=np.float64)
    faces = np.asarray(model.mesh_face[fa : fa + fn], dtype=np.int32)
    if verts.shape[0] < 3 or faces.shape[0] < 1:
        raise RuntimeError("inspection mesh has insufficient vertices/faces")
    return verts, faces


def _feature_edge_midpoints_local(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    # Face normals.
    tri = verts[faces]  # (F,3,3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    nn = np.where(nn < 1e-12, 1.0, nn)
    n = n / nn

    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        edge_faces[tuple(sorted((int(a), int(b))))].append(fi)
        edge_faces[tuple(sorted((int(b), int(c))))].append(fi)
        edge_faces[tuple(sorted((int(c), int(a))))].append(fi)

    out: list[np.ndarray] = []
    cos_th = np.cos(np.deg2rad(float(EDGE_DIHEDRAL_DEG_MIN)))
    for (i0, i1), adj in edge_faces.items():
        p0, p1 = verts[i0], verts[i1]
        edge_len = float(np.linalg.norm(p1 - p0))
        if edge_len < EDGE_LENGTH_MIN_M:
            continue
        is_feature = False
        if len(adj) == 1:
            is_feature = True
        elif len(adj) >= 2:
            c = float(np.clip(np.dot(n[adj[0]], n[adj[1]]), -1.0, 1.0))
            if c <= cos_th:
                is_feature = True
        if is_feature:
            out.append(0.5 * (p0 + p1))
    if not out:
        raise RuntimeError("no mesh feature edges found for inspection path")
    return np.asarray(out, dtype=np.float64)


def _local_points_to_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    points_local: np.ndarray,
) -> np.ndarray:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, INSPECTION_GEOM_NAME)
    R = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
    t = np.asarray(data.geom_xpos[gid], dtype=np.float64).reshape(3)
    return (R @ points_local.T).T + t[np.newaxis]


def _accessible_scan_waypoints(points_world: np.ndarray) -> np.ndarray:
    z = points_world[:, 2]
    z_min = float(np.quantile(z, ACCESS_Z_QUANTILE_MIN))
    r_xy = np.linalg.norm(points_world[:, :2], axis=1)
    mask = (
        (z >= z_min)
        & (z <= ACCESS_Z_MAX_M)
        & (r_xy >= ACCESS_XY_RADIUS_MIN_M)
        & (r_xy <= ACCESS_XY_RADIUS_MAX_M)
    )
    cand = points_world[mask]
    if cand.shape[0] < 3:
        cand = points_world

    # Order points by local continuity (nearest-neighbor contour walk), not pure axis sort.
    # Axis sort on curved/round features tends to create left-right zig-zag jumps.
    xy = cand[:, :2]
    ctr = np.mean(xy, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(xy - ctr, full_matrices=False)
    axis = vh[0]
    proj = (xy - ctr) @ axis
    start = int(np.argmin(proj))

    n_pts = int(cand.shape[0])
    ordered_idx = [start]
    remaining = np.ones(n_pts, dtype=bool)
    remaining[start] = False
    for _ in range(n_pts - 1):
        cur = ordered_idx[-1]
        rem_idx = np.flatnonzero(remaining)
        if rem_idx.size == 0:
            break
        d2 = np.sum((cand[rem_idx] - cand[cur]) ** 2, axis=1)
        nxt = int(rem_idx[int(np.argmin(d2))])
        ordered_idx.append(nxt)
        remaining[nxt] = False
    ordered = cand[np.asarray(ordered_idx, dtype=np.int32)]

    # Arc-length resampling for smooth progression along the extracted contour.
    seg = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    if seg.size == 0 or float(np.sum(seg)) < 1e-8:
        out = ordered[:1]
    else:
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])
        m = min(SCAN_WAYPOINTS, ordered.shape[0])
        t = np.linspace(0.0, total, m)
        out = np.empty((m, 3), dtype=np.float64)
        j = 0
        for i, ti in enumerate(t):
            while j + 1 < len(s) and s[j + 1] < ti:
                j += 1
            if j + 1 >= len(s):
                out[i] = ordered[-1]
                continue
            ds = s[j + 1] - s[j]
            a = 0.0 if ds < 1e-10 else (ti - s[j]) / ds
            out[i] = ordered[j] + a * (ordered[j + 1] - ordered[j])
    if out.shape[0] < 2:
        raise RuntimeError("insufficient scan waypoints after mesh-edge filtering")
    return out


def accessible_edge_points_world(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return ordered world-frame scan waypoints derived from mesh feature edges."""
    verts, faces = _inspection_mesh_local(model)
    edge_mid_local = _feature_edge_midpoints_local(verts, faces)
    edge_mid_world = _local_points_to_world(model, data, edge_mid_local)
    return _accessible_scan_waypoints(edge_mid_world)


def _positions_to_se3(positions: np.ndarray, R_fixed: np.ndarray) -> np.ndarray:
    n = int(positions.shape[0])
    T = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    T[:, :3, :3] = R_fixed[np.newaxis]
    T[:, :3, 3] = positions
    return T


def _line_positions(p_start: np.ndarray, p_end: np.ndarray, n_steps: int) -> np.ndarray:
    t = np.arange(1, n_steps + 1, dtype=np.float64) / n_steps
    return p_start[np.newaxis] + t[:, np.newaxis] * (p_end - p_start)


def _polyline_positions(points: np.ndarray, n_per_segment: int) -> np.ndarray:
    if points.shape[0] < 2:
        return points.copy()
    out = []
    for i in range(points.shape[0] - 1):
        out.append(_line_positions(points[i], points[i + 1], n_per_segment))
    return np.concatenate(out, axis=0)


def policy(obs: dict[str, Any], step: int, env: UR5GripperEnv) -> np.ndarray:
    """Step 0: build and solve scan trajectory. Later steps: replay cached controls."""
    check_ik_dependencies()
    global _episode_start_ctrl, _latch_qpos, _q_plan, _scan_targets_world

    if step == 0 or _episode_start_ctrl is None:
        _episode_start_ctrl = _build_episode_latch(obs, env)
        _latch_qpos = np.asarray(obs["qpos"], dtype=np.float64).copy()

        edges_world = accessible_edge_points_world(env.model, env.data)
        stand_off = np.array([0.0, 0.0, SCAN_STANDOFF_M], dtype=np.float64)
        _scan_targets_world = edges_world + stand_off[np.newaxis]

        data0 = mujoco.MjData(env.model)
        data0.qpos[:] = obs["qpos"]
        mujoco.mj_forward(env.model, data0)
        T0 = tool0_se3_matrix(env.model, data0)
        p0 = T0[:3, 3].copy()

        dt = float(env.control_dt)
        n_crane = max(1, round(CRANE_DURATION_S / dt))
        n_approach = max(1, round(APPROACH_DURATION_S / dt))
        n_seg = max(1, round(SCAN_SEGMENT_DURATION_S / dt))

        # Crane settle first: move arm joints from current qpos to episode-start ctrl targets.
        q_crane = _latch_qpos.copy()
        q_crane[:6] = np.asarray(obs["ctrl"], dtype=np.float64)[:6]
        t_ramp = np.arange(1, n_crane + 1, dtype=np.float64) / n_crane
        q_settle = _latch_qpos[np.newaxis, :] + t_ramp[:, np.newaxis] * (q_crane - _latch_qpos)[
            np.newaxis, :
        ]

        data_crane = mujoco.MjData(env.model)
        data_crane.qpos[:] = q_crane
        mujoco.mj_forward(env.model, data_crane)
        T_crane = tool0_se3_matrix(env.model, data_crane)
        p_crane = T_crane[:3, 3].copy()

        approach_pos = _line_positions(p_crane, _scan_targets_world[0], n_approach)
        scan_pos = _polyline_positions(_scan_targets_world, n_seg)

        ik = _make_ik_service(env)
        q_approach = ik.track_tool_se3_trajectory(
            q_crane,
            _positions_to_se3(approach_pos, R_SCAN),
            q_crane,
            rolling_posture=True,
        )
        q_scan = ik.track_tool_se3_trajectory(
            q_approach[-1],
            _positions_to_se3(scan_pos, R_SCAN),
            q_approach[-1],
            rolling_posture=True,
        )
        _q_plan = np.concatenate([q_settle, q_approach, q_scan], axis=0)

    assert _episode_start_ctrl is not None and _latch_qpos is not None and _q_plan is not None

    lo = env.model.actuator_ctrlrange[:, 0]
    hi = env.model.actuator_ctrlrange[:, 1]
    ctrl = np.empty(env.nu, dtype=np.float64)

    k = min(step, len(_q_plan) - 1)
    q_cmd = _latch_qpos.copy()
    q_cmd[:6] = _q_plan[k, :6]
    g_idx = _gripper_idx(env)

    for i in range(env.nu):
        if i == g_idx:
            ctrl[i] = float(np.clip(GRIPPER_CLOSED_CTRL, lo[i], hi[i]))
        elif int(env.model.actuator_trntype[i]) == 0:
            qadr = _actuator_qadr(env, i)
            ctrl[i] = float(np.clip(float(q_cmd[qadr]), lo[i], hi[i]))
        else:
            ctrl[i] = float(np.clip(_episode_start_ctrl[i], lo[i], hi[i]))
    return ctrl
