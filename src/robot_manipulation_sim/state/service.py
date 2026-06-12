"""Generic robot-state queries and actuator/control helpers.

The service intentionally avoids task-specific assumptions and provides small,
composable primitives for policies:

- actuator/joint mapping utilities
- world/body-frame point transforms
- generic hole/probe latch scoring in a body-local frame
- conversion from joint+gripper command to full actuator control vector
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True)
class HoleLatchSpec:
    """Axis-aligned box-face hole model in a body-local frame.

    The modeled object is a box centered at the body origin with face planes at
    +/- ``half_size_m`` along each axis. A point is considered latched if:

    - its distance to the nearest hole axis on a face is <= ``hole_radius_m``
    - its distance to that face plane is <= ``face_tol_m``
    """

    half_size_m: float
    hole_radius_m: float
    face_tol_m: float


class RobotStateService:
    """Model-level helpers for generic control/state operations."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model

    def actuator_qadr(self, act_idx: int) -> int:
        """Return qpos address driven by a position actuator."""
        jid = int(self.model.actuator_trnid[act_idx, 0])
        return int(self.model.jnt_qposadr[jid])

    def gripper_actuator_index(self) -> int:
        """Return the first non-position actuator index (parallel gripper in this repo)."""
        for i in range(int(self.model.nu)):
            if int(self.model.actuator_trntype[i]) != 0:
                return i
        raise RuntimeError("No tendon actuator found in model (gripper).")

    def build_episode_latch(self, obs: dict[str, Any], *, nu: int) -> list[float]:
        """Build per-actuator episode latch target from an observation."""
        out: list[float] = []
        for i in range(nu):
            if int(self.model.actuator_trntype[i]) == 0:
                out.append(float(obs["qpos"][self.actuator_qadr(i)]))
            else:
                out.append(float(obs["ctrl"][i]))
        return out

    def ctrl_from_q_and_gripper(
        self,
        q_cmd: np.ndarray,
        g_cmd: float,
        *,
        nu: int,
        episode_start_ctrl: list[float],
    ) -> np.ndarray:
        """Map arm q-command + scalar gripper command to a full actuator ctrl vector."""
        lo = self.model.actuator_ctrlrange[:, 0]
        hi = self.model.actuator_ctrlrange[:, 1]
        g_idx = self.gripper_actuator_index()
        ctrl = np.empty(nu, dtype=np.float64)
        for i in range(nu):
            if i == g_idx:
                ctrl[i] = float(np.clip(g_cmd, lo[i], hi[i]))
            elif int(self.model.actuator_trntype[i]) == 0:
                ctrl[i] = float(np.clip(float(q_cmd[self.actuator_qadr(i)]), lo[i], hi[i]))
            else:
                ctrl[i] = float(np.clip(episode_start_ctrl[i], lo[i], hi[i]))
        return ctrl

    def body_world_pose(self, data: mujoco.MjData, body_name: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (position, rotation-matrix) for a named body in world frame."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            return None
        p = np.asarray(data.xpos[bid], dtype=np.float64)
        r = np.asarray(data.xmat[bid], dtype=np.float64).reshape(3, 3)
        return p, r

    def geom_world_position(self, data: mujoco.MjData, geom_name: str) -> np.ndarray | None:
        """Return world position for a named geom."""
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if gid < 0:
            return None
        return np.asarray(data.geom_xpos[gid], dtype=np.float64)

    @staticmethod
    def world_to_body_local(
        body_pos_world: np.ndarray,
        body_rot_world: np.ndarray,
        point_world: np.ndarray,
    ) -> np.ndarray:
        """Transform one world-frame point into the body-local frame."""
        return np.asarray(body_rot_world, dtype=np.float64).T @ (
            np.asarray(point_world, dtype=np.float64) - np.asarray(body_pos_world, dtype=np.float64)
        )

    def geom_points_in_body_local(
        self,
        data: mujoco.MjData,
        *,
        body_name: str,
        geom_names: tuple[str, ...],
    ) -> list[np.ndarray] | None:
        """Return named geom world positions expressed in a named body frame."""
        pose = self.body_world_pose(data, body_name)
        if pose is None:
            return None
        p_body, r_body = pose
        out: list[np.ndarray] = []
        for name in geom_names:
            p = self.geom_world_position(data, name)
            if p is None:
                return None
            out.append(self.world_to_body_local(p_body, r_body, p))
        return out

    def latch_eval_for_local_point(
        self,
        local_point: np.ndarray,
        spec: HoleLatchSpec,
    ) -> tuple[bool, str, float, float]:
        """Evaluate one local-frame point against the nearest face hole model.

        Returns: ``(latched, face_label, radial_err, face_err)``.
        """
        return self._best_hole_face_for_point(local_point, spec)

    def all_local_points_latched(
        self,
        local_points: list[np.ndarray],
        spec: HoleLatchSpec,
    ) -> bool:
        """True if every local-frame point satisfies hole latch constraints."""
        if len(local_points) == 0:
            return False
        return bool(all(self._best_hole_face_for_point(p, spec)[0] for p in local_points))

    def both_pegs_latched(
        self,
        data: mujoco.MjData,
        *,
        tip_body_name: str,
        left_peg_geom_name: str,
        right_peg_geom_name: str,
        spec: HoleLatchSpec,
    ) -> bool:
        """Compatibility helper: two named peg geoms latched in one tip-body frame."""
        local_points = self.geom_points_in_body_local(
            data,
            body_name=tip_body_name,
            geom_names=(left_peg_geom_name, right_peg_geom_name),
        )
        if local_points is None:
            return False
        return self.all_local_points_latched(local_points, spec)

    @staticmethod
    def _best_hole_face_for_point(
        p_tip_local: np.ndarray,
        spec: HoleLatchSpec,
    ) -> tuple[bool, str, float, float]:
        p = np.asarray(p_tip_local, dtype=np.float64)
        faces = (
            ("+x", 0, 1.0, (1, 2)),
            ("-x", 0, -1.0, (1, 2)),
            ("+y", 1, 1.0, (0, 2)),
            ("-y", 1, -1.0, (0, 2)),
            ("+z", 2, 1.0, (0, 1)),
            ("-z", 2, -1.0, (0, 1)),
        )
        best = (False, "none", 1e9, 1e9)
        for label, axis, sign, (r1, r2) in faces:
            face_err = abs(p[axis] - sign * spec.half_size_m)
            radial = float(np.hypot(p[r1], p[r2]))
            latched = radial <= spec.hole_radius_m and face_err <= spec.face_tol_m
            score = face_err + radial
            if latched and (not best[0] or score < best[2] + best[3]):
                best = (True, label, radial, face_err)
            elif (not best[0]) and score < best[2] + best[3]:
                best = (False, label, radial, face_err)
        return best
