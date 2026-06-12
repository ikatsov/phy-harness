"""Cable routing policy with modular waypoint-driven planning.

High-level behavior:
1) detect cable tip + gate centers (with temporal stabilization),
2) build a pick-and-lift plan,
3) follow a polyline route through all gates while holding the cube.
"""

from dataclasses import dataclass
import os
from typing import Any

import mujoco
import numpy as np

from robot_manipulation_sim.env import UR5GripperEnv
from robot_manipulation_sim.ik import check_ik_dependencies
from robot_manipulation_sim.ik.tool_pose import tool0_se3_matrix
from robot_manipulation_sim.state import RobotStateService
from robot_manipulation_sim.vision import (
    ColorDetectionOperation,
    ColorRange,
    PositionStabilizer,
    VisionService,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimingConfig:
    initial_settle_s: float = 1.0
    crane_s: float = 0.6
    approach_pick_s: float = 1.0
    descend_pick_s: float = 0.5
    final_descend_pick_s: float = 0.4
    close_gripper_s: float = 1.0
    lift_s: float = 0.8


@dataclass(frozen=True)
class HeightConfig:
    tip_z_m: float = 0.016
    tool_to_fingertip_z_m: float = 0.115
    pick_contact_undershoot_m: float = -0.010
    transport_z_m: float = 0.36
    gate_approach_z_m: float = 0.34
    gate_thread_z_m: float = 0.30
    route_arc_z_m: float = 0.36
    first_gate_thread_z_m: float = 0.325

    @property
    def pick_approach_z_m(self) -> float:
        return self.tip_z_m + self.tool_to_fingertip_z_m + 0.08

    @property
    def pick_contact_z_m(self) -> float:
        return self.tip_z_m + self.tool_to_fingertip_z_m - self.pick_contact_undershoot_m

    @property
    def pick_precontact_z_m(self) -> float:
        return self.pick_contact_z_m + 0.025


@dataclass(frozen=True)
class RouteConfig:
    gate_radius_m: float = 0.035
    first_gate_long_approach_m: float = 0.20
    first_gate_arc_points: int = 11
    gate_arc_points: int = 5
    points_per_segment: int = 8


@dataclass(frozen=True)
class PerceptionConfig:
    settle_guard_steps: int = 10
    stable_window: int = 15
    tip_xy_std_max_m: float = 0.020
    peg_xy_std_max_m: float = 0.010
    tip_accept_z_min_m: float = -0.01
    tip_accept_z_max_m: float = 0.20
    tip_unproject_z_m: float = 0.016
    tip_max_jump_m: float = 0.05
    max_wait_after_settle_s: float = 4.0
    peg_expected_xy: dict[str, np.ndarray] | None = None
    peg_max_dist_from_expected_m: float = 0.12


@dataclass(frozen=True)
class CalibrationConfig:
    topdown_to_grasp_A: np.ndarray
    topdown_to_grasp_b: np.ndarray


@dataclass(frozen=True)
class IkConfig:
    inner_iters: int = 8
    position_cost: float = 2.0
    orientation_cost: float = 0.9
    frame_lm_damping: float = 0.35
    posture_cost: float = 0.03
    damping_cost: float = 2.5e-3


@dataclass(frozen=True)
class PolicyConfig:
    timing: TimingConfig
    heights: HeightConfig
    route: RouteConfig
    perception: PerceptionConfig
    calibration: CalibrationConfig
    ik: IkConfig
    gripper_open: float = 0.0
    gripper_closed: float = 255.0
    topdown_width: int = 1920
    topdown_height: int = 1440
    tip_max_area_px: float = 0.0
    top_down_R: np.ndarray | None = None


def _build_config() -> PolicyConfig:
    heights = HeightConfig()
    perception = PerceptionConfig(
        peg_expected_xy={
            "peg_cyan": np.array([0.40, -0.24], dtype=np.float64),
            "peg_magenta": np.array([0.52, -0.08], dtype=np.float64),
            "peg_yellow": np.array([0.58, 0.10], dtype=np.float64),
        }
    )
    w = int(os.getenv("CABLE_ROUTING_TOPDOWN_WIDTH", "1920"))
    h = int(os.getenv("CABLE_ROUTING_TOPDOWN_HEIGHT", "1440"))
    return PolicyConfig(
        timing=TimingConfig(),
        heights=heights,
        route=RouteConfig(),
        perception=perception,
        calibration=CalibrationConfig(
            topdown_to_grasp_A=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            topdown_to_grasp_b=np.array([0.013, -0.018], dtype=np.float64),
        ),
        ik=IkConfig(),
        topdown_width=w,
        topdown_height=h,
        tip_max_area_px=150.0 * (w / 640.0) * (h / 480.0) * 1.5,
        top_down_R=np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64),
    )


CFG = _build_config()


# Public constants kept for compatibility with existing tools/tests.
ENABLE_RGB: bool = True
CABLE_END_LABEL: str = "cable_tip"
PEG_ROUTE_ORDER: tuple[str, ...] = ("peg_cyan", "peg_magenta", "peg_yellow")
GRIPPER_OPEN: float = CFG.gripper_open
GRIPPER_CLOSED: float = CFG.gripper_closed
T_INITIAL_SETTLE: float = CFG.timing.initial_settle_s
T_CRANE: float = CFG.timing.crane_s
T_APPROACH_PICK: float = CFG.timing.approach_pick_s
T_DESCEND_PICK: float = CFG.timing.descend_pick_s
T_FINAL_DESCEND_PICK: float = CFG.timing.final_descend_pick_s
T_CLOSE_GRIPPER: float = CFG.timing.close_gripper_s
T_LIFT: float = CFG.timing.lift_s
T_ROUTE_SEGMENT: float = 0.8  # compatibility only; route is waypoint-polyline driven
T_APPROACH: float = T_APPROACH_PICK
T_DESCEND: float = T_DESCEND_PICK
T_GRIP: float = T_CLOSE_GRIPPER
T_WRIST_ALIGN: float = 0.0
T_WRIST_RECENTER: float = 0.0


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------


class EpisodeState:
    def __init__(
        self,
        *,
        episode_start_ctrl: list[float],
        latch_qpos: np.ndarray,
        vision: VisionService,
        vision_profile: Any,
        stabilizer: PositionStabilizer,
        state_service: RobotStateService,
    ) -> None:
        self.episode_start_ctrl = episode_start_ctrl
        self.latch_qpos = latch_qpos
        self.vision = vision
        self.vision_profile = vision_profile
        self.stabilizer = stabilizer
        self.state_service = state_service
        self.q_plan: np.ndarray | None = None
        self.g_plan: np.ndarray | None = None
        self.plan_start_step: int | None = None
        self.rebase_trigger_idx: int | None = None
        self.rebase_done: bool = False
        self.last_tip_world: np.ndarray | None = None


_episode: EpisodeState | None = None


def reset() -> None:
    global _episode
    _episode = None


# ---------------------------------------------------------------------------
# Vision profile + perception
# ---------------------------------------------------------------------------


class CableRoutingVisionProfile:
    def __init__(self) -> None:
        self.cable_tip = ColorDetectionOperation(
            label=CABLE_END_LABEL,
            ranges=(ColorRange(np.array([0, 35, 35], dtype=np.float64), np.array([24, 255, 255], dtype=np.float64)),),
            min_area_px=4,
        )
        self.peg_pairs = (
            ColorDetectionOperation(
                label="peg_cyan",
                ranges=(ColorRange(np.array([78, 70, 70], dtype=np.float64), np.array([110, 255, 255], dtype=np.float64)),),
                min_area_px=40,
            ),
            ColorDetectionOperation(
                label="peg_magenta",
                ranges=(ColorRange(np.array([132, 70, 70], dtype=np.float64), np.array([170, 255, 255], dtype=np.float64)),),
                min_area_px=40,
            ),
            ColorDetectionOperation(
                label="peg_yellow",
                ranges=(ColorRange(np.array([20, 80, 80], dtype=np.float64), np.array([42, 255, 255], dtype=np.float64)),),
                min_area_px=40,
            ),
        )


def _apply_topdown_to_grasp_xy(xy: np.ndarray) -> np.ndarray:
    return CFG.calibration.topdown_to_grasp_A @ np.asarray(xy, dtype=np.float64) + CFG.calibration.topdown_to_grasp_b


def _unproject_tip_world(episode: EpisodeState, data: mujoco.MjData, uv: tuple[int, int], *, target_z: float) -> np.ndarray:
    world = episode.vision.unproject_pixel_to_world(data, uv, target_z=target_z)
    world = np.asarray(world, dtype=np.float64)
    world[:2] = _apply_topdown_to_grasp_xy(world[:2])
    return world


def _fallback_scene_estimate(env: UR5GripperEnv) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    tip_bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "cable_tip_marker")
    if tip_bid >= 0:
        tip_world = np.asarray(env.data.xpos[tip_bid], dtype=np.float64).copy()
    else:
        tip_world = np.array([0.834, -0.20, CFG.perception.tip_unproject_z_m], dtype=np.float64)
    tip_world[2] = float(np.clip(tip_world[2], CFG.perception.tip_accept_z_min_m, CFG.perception.tip_accept_z_max_m))
    peg_world = {
        k: np.array([float(v[0]), float(v[1]), 0.17], dtype=np.float64)
        for k, v in (CFG.perception.peg_expected_xy or {}).items()
    }
    return tip_world, peg_world


def _collect_stable_perception(env: UR5GripperEnv, episode: EpisodeState) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    try:
        img = episode.vision.render_detection_image(env.data)
    except Exception:
        return _fallback_scene_estimate(env)

    tip_candidates = episode.vision.detect_with_operation(img, episode.vision_profile.cable_tip)
    tip_candidates = [d for d in tip_candidates if float(d.area) <= CFG.tip_max_area_px]
    if not tip_candidates:
        return _fallback_scene_estimate(env)

    if episode.last_tip_world is None:
        tip_det = max(tip_candidates, key=lambda d: float(d.area))
    else:
        tip_det = min(
            tip_candidates,
            key=lambda d: float(
                np.linalg.norm(
                    _unproject_tip_world(episode, env.data, d.center_uv, target_z=CFG.perception.tip_unproject_z_m)[:2]
                    - episode.last_tip_world[:2]
                )
            ),
        )
    tip_world = _unproject_tip_world(episode, env.data, tip_det.center_uv, target_z=CFG.perception.tip_unproject_z_m)

    peg_world: dict[str, np.ndarray] = {}
    expected = CFG.perception.peg_expected_xy or {}
    for op in episode.vision_profile.peg_pairs:
        candidates = episode.vision.detect_with_operation(img, op)
        if not candidates:
            return _fallback_scene_estimate(env)
        worlds = [np.asarray(episode.vision.unproject_pixel_to_world(env.data, d.center_uv, target_z=0.17), dtype=np.float64) for d in candidates]
        expected_xy = expected[op.label]
        best_world = worlds[int(np.argmin([float(np.linalg.norm(w[:2] - expected_xy)) for w in worlds]))]
        if float(np.linalg.norm(best_world[:2] - expected_xy)) > CFG.perception.peg_max_dist_from_expected_m:
            return _fallback_scene_estimate(env)
        peg_world[op.label] = best_world

    if not (CFG.perception.tip_accept_z_min_m <= float(tip_world[2]) <= CFG.perception.tip_accept_z_max_m):
        return _fallback_scene_estimate(env)
    if episode.last_tip_world is not None and float(np.linalg.norm(tip_world[:2] - episode.last_tip_world[:2])) > CFG.perception.tip_max_jump_m:
        return _fallback_scene_estimate(env)

    episode.last_tip_world = tip_world.copy()
    episode.stabilizer.push(tip_world, peg_world)
    stable = episode.stabilizer.stable_means(
        primary_xy_std_max=CFG.perception.tip_xy_std_max_m,
        labeled_xy_std_max=CFG.perception.peg_xy_std_max_m,
    )
    if stable is not None:
        return stable
    early = episode.stabilizer.early_means(min_samples=max(3, CFG.perception.stable_window // 3))
    if early is not None:
        return early
    return _fallback_scene_estimate(env)


# ---------------------------------------------------------------------------
# Waypoint plan builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanTrajectory:
    q_plan: np.ndarray
    g_plan: np.ndarray
    rebase_trigger_idx: int


def _interpolate_se3_positions(p_start: np.ndarray, p_end: np.ndarray, n_steps: int) -> np.ndarray:
    t = np.arange(1, n_steps + 1, dtype=np.float64) / max(n_steps, 1)
    pos = p_start[np.newaxis] + t[:, np.newaxis] * (p_end - p_start)
    T = np.tile(np.eye(4, dtype=np.float64), (n_steps, 1, 1))
    T[:, :3, :3] = np.asarray(CFG.top_down_R, dtype=np.float64)
    T[:, :3, 3] = pos
    return T


def _sample_polyline(points: np.ndarray, points_per_segment: int) -> np.ndarray:
    chunks = [np.linspace(points[i], points[i + 1], points_per_segment, endpoint=False) for i in range(len(points) - 1)]
    chunks.append(points[-1][np.newaxis])
    return np.concatenate(chunks, axis=0)


def _build_gate_route_waypoints(cable_tip_world: np.ndarray, peg_world: dict[str, np.ndarray]) -> np.ndarray:
    z = CFG.heights
    r = CFG.route
    p0 = np.asarray(cable_tip_world, dtype=np.float64).copy()
    p0[2] = z.transport_z_m
    pts: list[np.ndarray] = [p0]
    prev = p0[:2].copy()
    for i, label in enumerate(PEG_ROUTE_ORDER):
        gate = np.asarray(peg_world[label], dtype=np.float64)[:2]
        if i == 0:
            run = max(float(np.linalg.norm(prev - gate)), r.first_gate_long_approach_m)
            p_above = np.array([gate[0] + run, gate[1], z.gate_approach_z_m], dtype=np.float64)
            theta = np.linspace(0.0, 0.5 * np.pi, max(2, r.first_gate_arc_points), endpoint=True)
            arc = np.stack(
                [
                    gate[0] + r.gate_radius_m * np.cos(theta),
                    gate[1] + r.gate_radius_m * np.sin(theta),
                    np.full_like(theta, z.route_arc_z_m),
                ],
                axis=1,
            )
            p_thread = np.array([gate[0], gate[1] + r.gate_radius_m, z.first_gate_thread_z_m], dtype=np.float64)
            p_out = np.array([p_thread[0], p_thread[1], z.gate_approach_z_m], dtype=np.float64)
            pts.extend([p_above, *list(arc), p_thread, p_out])
            prev = p_out[:2].copy()
            continue
        if i == 1:
            p_above = np.array([gate[0] + r.gate_radius_m, gate[1], z.gate_approach_z_m], dtype=np.float64)
            p_thread = np.array([gate[0] - r.gate_radius_m, gate[1], z.gate_thread_z_m], dtype=np.float64)
            p_out = np.array([p_thread[0], p_thread[1], z.gate_approach_z_m], dtype=np.float64)
            pts.extend([p_above, p_thread, p_out])
            prev = p_out[:2].copy()
            continue
        p_above = np.array([gate[0], gate[1] - r.gate_radius_m, z.gate_approach_z_m], dtype=np.float64)
        theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, max(2, r.gate_arc_points), endpoint=True)
        arc = np.stack(
            [
                gate[0] + r.gate_radius_m * np.cos(theta),
                gate[1] + r.gate_radius_m * np.sin(theta),
                np.full_like(theta, z.route_arc_z_m),
            ],
            axis=1,
        )
        p_thread = np.array([gate[0], gate[1] + r.gate_radius_m, z.gate_thread_z_m], dtype=np.float64)
        p_out = np.array([p_thread[0], p_thread[1], z.gate_approach_z_m], dtype=np.float64)
        pts.extend([p_above, *list(arc), p_thread, p_out])
        prev = p_out[:2].copy()
    return np.stack(pts, axis=0)


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _wrap_to_near(prev: np.ndarray, q_new: np.ndarray, arm_dofs: int = 6) -> np.ndarray:
    out = np.asarray(q_new, dtype=np.float64).copy()
    for j in range(min(arm_dofs, len(out), len(prev))):
        delta = out[j] - prev[j]
        out[j] = prev[j] + ((delta + np.pi) % (2.0 * np.pi) - np.pi)
    return out


def _stabilize_joint_branch(q_plan: np.ndarray, arm_dofs: int = 6) -> np.ndarray:
    q = np.asarray(q_plan, dtype=np.float64).copy()
    for i in range(1, len(q)):
        q[i] = _wrap_to_near(q[i - 1], q[i], arm_dofs=arm_dofs)
    return q


def _make_ik_service(env: UR5GripperEnv) -> Any:
    from robot_manipulation_sim.ik.service import MujocoMinkIkService  # noqa: PLC0415

    return MujocoMinkIkService(
        env.model,
        inner_iters=CFG.ik.inner_iters,
        control_dt=float(env.control_dt),
        position_cost=CFG.ik.position_cost,
        orientation_cost=CFG.ik.orientation_cost,
        frame_lm_damping=CFG.ik.frame_lm_damping,
        posture_cost=CFG.ik.posture_cost,
        damping_cost=CFG.ik.damping_cost,
    )


def build_plan(env: UR5GripperEnv, obs: dict[str, Any], cable_tip_world: np.ndarray, peg_world: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Build the full joint/gripper trajectory from pick to final gate route."""
    tr = build_plan_trajectory(env, obs, cable_tip_world, peg_world)
    return tr.q_plan, tr.g_plan


def build_plan_trajectory(
    env: UR5GripperEnv,
    obs: dict[str, Any],
    cable_tip_world: np.ndarray,
    peg_world: dict[str, np.ndarray],
) -> PlanTrajectory:
    check_ik_dependencies()
    q_start = np.asarray(obs["qpos"], dtype=np.float64).copy()
    ctrl0 = np.asarray(obs["ctrl"], dtype=np.float64).copy()
    dt = float(env.control_dt)
    n_crane = max(1, round(CFG.timing.crane_s / dt))
    n_app = max(1, round(CFG.timing.approach_pick_s / dt))
    n_desc = max(1, round(CFG.timing.descend_pick_s / dt))
    n_final_desc = max(1, round(CFG.timing.final_descend_pick_s / dt))
    n_grip = max(1, round(CFG.timing.close_gripper_s / dt))
    n_lift = max(1, round(CFG.timing.lift_s / dt))

    q_crane = q_start.copy()
    q_crane[:6] = ctrl0[:6]
    t = np.arange(1, n_crane + 1, dtype=np.float64) / n_crane
    q_settle = q_start[np.newaxis, :] + t[:, np.newaxis] * (q_crane - q_start)[np.newaxis, :]

    data_crane = mujoco.MjData(env.model)
    data_crane.qpos[:] = q_crane
    mujoco.mj_forward(env.model, data_crane)
    p_crane = tool0_se3_matrix(env.model, data_crane)[:3, 3].copy()

    tip_xy = np.asarray(cable_tip_world[:2], dtype=np.float64)
    z = CFG.heights
    p_pick_approach = np.array([tip_xy[0], tip_xy[1], z.pick_approach_z_m], dtype=np.float64)
    p_pick_precontact = np.array([tip_xy[0], tip_xy[1], z.pick_precontact_z_m], dtype=np.float64)
    p_pick_contact = np.array([tip_xy[0], tip_xy[1], z.pick_contact_z_m], dtype=np.float64)
    p_lift = np.array([tip_xy[0], tip_xy[1], z.transport_z_m], dtype=np.float64)

    ik = _make_ik_service(env)
    q_app = ik.track_tool_se3_trajectory(
        q_crane,
        _interpolate_se3_positions(p_crane, p_pick_approach, n_app),
        np.tile(q_crane, (n_app, 1)),
        rolling_posture=False,
    )
    q_desc = ik.track_tool_se3_trajectory(
        q_app[-1],
        _interpolate_se3_positions(p_pick_approach, p_pick_precontact, n_desc),
        np.tile(q_crane, (n_desc, 1)),
        rolling_posture=False,
    )
    q_final_desc = ik.track_tool_se3_trajectory(
        q_desc[-1],
        _interpolate_se3_positions(p_pick_precontact, p_pick_contact, n_final_desc),
        np.tile(q_crane, (n_final_desc, 1)),
        rolling_posture=False,
    )
    q_close_hold = np.tile(q_final_desc[-1], (n_grip, 1))
    q_lift = ik.track_tool_se3_trajectory(
        q_final_desc[-1],
        _interpolate_se3_positions(p_pick_contact, p_lift, n_lift),
        np.tile(q_crane, (n_lift, 1)),
        rolling_posture=False,
    )

    route_waypoints = _build_gate_route_waypoints(cable_tip_world, peg_world)
    route_positions = _sample_polyline(route_waypoints, CFG.route.points_per_segment)
    route_targets = np.tile(np.eye(4, dtype=np.float64), (route_positions.shape[0], 1, 1))
    route_targets[:, :3, :3] = np.asarray(CFG.top_down_R, dtype=np.float64)
    route_targets[:, :3, 3] = route_positions
    q_route = ik.track_tool_se3_trajectory(
        q_lift[-1],
        route_targets,
        np.tile(q_crane, (route_targets.shape[0], 1)),
        rolling_posture=False,
    )

    q_plan = np.concatenate([q_settle, q_app, q_desc, q_final_desc, q_close_hold, q_lift, q_route], axis=0)
    q_plan = _stabilize_joint_branch(q_plan)

    close_tau = _smoothstep01(np.linspace(0.0, 1.0, n_grip))
    close_profile = GRIPPER_OPEN + (GRIPPER_CLOSED - GRIPPER_OPEN) * close_tau
    g_plan = np.concatenate(
        [
            np.full(n_crane, GRIPPER_OPEN),
            np.full(n_app, GRIPPER_OPEN),
            np.full(n_desc, GRIPPER_OPEN),
            np.full(n_final_desc, GRIPPER_OPEN),
            close_profile,
            np.full(n_lift, GRIPPER_CLOSED),
            np.full(len(q_route), GRIPPER_CLOSED),
        ],
        axis=0,
    )

    rebase_trigger_idx = n_crane + n_app + n_desc + n_final_desc - 1
    return PlanTrajectory(q_plan=q_plan, g_plan=g_plan, rebase_trigger_idx=rebase_trigger_idx)


def _plan_ctrl(k_rel: int, env: UR5GripperEnv, episode: EpisodeState) -> np.ndarray:
    assert episode.q_plan is not None and episode.g_plan is not None
    q_cmd = episode.latch_qpos.copy()
    idx = min(k_rel, len(episode.q_plan) - 1)
    q_cmd[:6] = episode.q_plan[idx, :6]
    return episode.state_service.ctrl_from_q_and_gripper(
        q_cmd,
        float(episode.g_plan[idx]),
        nu=env.nu,
        episode_start_ctrl=episode.episode_start_ctrl,
    )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def _build_episode_state(obs: dict[str, Any], env: UR5GripperEnv) -> EpisodeState:
    state_service = RobotStateService(env.model)
    latch = state_service.build_episode_latch(obs, nu=env.nu)
    return EpisodeState(
        episode_start_ctrl=latch,
        latch_qpos=np.asarray(obs["qpos"], dtype=np.float64).copy(),
        vision=VisionService(env.model, camera_name="topdown", width=CFG.topdown_width, height=CFG.topdown_height),
        vision_profile=CableRoutingVisionProfile(),
        stabilizer=PositionStabilizer(labels=PEG_ROUTE_ORDER, window_size=CFG.perception.stable_window),
        state_service=state_service,
    )


def _settle_ctrl(env: UR5GripperEnv, episode: EpisodeState) -> np.ndarray:
    lo = env.model.actuator_ctrlrange[:, 0]
    hi = env.model.actuator_ctrlrange[:, 1]
    return np.asarray(
        [float(np.clip(episode.episode_start_ctrl[i], lo[i], hi[i])) for i in range(env.nu)],
        dtype=np.float64,
    )


def policy(obs: dict[str, Any], step: int, env: UR5GripperEnv) -> np.ndarray:
    """Entry point for simulation policy execution."""
    check_ik_dependencies()
    global _episode
    if step == 0 or _episode is None:
        _episode = _build_episode_state(obs, env)
    episode = _episode

    settle_steps = max(1, round(CFG.timing.initial_settle_s / float(env.control_dt)))
    if step < settle_steps:
        return _settle_ctrl(env, episode)

    if episode.q_plan is None or episode.g_plan is None or episode.plan_start_step is None:
        if (step - settle_steps) < CFG.perception.settle_guard_steps:
            return _settle_ctrl(env, episode)
        stable = _collect_stable_perception(env, episode)
        if stable is None:
            waited_steps = step - settle_steps - CFG.perception.settle_guard_steps
            waited_limit = max(1, round(CFG.perception.max_wait_after_settle_s / float(env.control_dt)))
            if waited_steps < waited_limit:
                return _settle_ctrl(env, episode)
            raise RuntimeError("cable_routing: stable top-camera detections not achieved within timeout")
        tip_world, peg_world = stable
        episode.latch_qpos = np.asarray(obs["qpos"], dtype=np.float64).copy()
        plan = build_plan_trajectory(env, obs, tip_world, peg_world)
        episode.q_plan = plan.q_plan
        episode.g_plan = plan.g_plan
        episode.rebase_trigger_idx = plan.rebase_trigger_idx
        episode.rebase_done = False
        episode.plan_start_step = step

    assert episode.plan_start_step is not None and episode.q_plan is not None and episode.rebase_trigger_idx is not None
    k_rel = step - episode.plan_start_step
    if not episode.rebase_done and k_rel == (episode.rebase_trigger_idx + 1):
        q_now = np.asarray(obs["qpos"], dtype=np.float64)
        delta = q_now[:6] - episode.q_plan[episode.rebase_trigger_idx, :6]
        if episode.rebase_trigger_idx + 1 < len(episode.q_plan):
            episode.q_plan[episode.rebase_trigger_idx + 1 :, :6] += delta[np.newaxis, :]
            tail = _stabilize_joint_branch(episode.q_plan[episode.rebase_trigger_idx:, :].copy())
            episode.q_plan[episode.rebase_trigger_idx:, :6] = tail[:, :6]
        episode.rebase_done = True

    return _plan_ctrl(k_rel, env, episode)
