"""Differential IK on the simulator ``MjModel`` using `mink <https://github.com/kevinzakka/mink>`_."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

from robot_manipulation_sim.ik import check_ik_dependencies
from robot_manipulation_sim.ik.tool_pose import tool0_position


@dataclass
class MujocoMinkIkService:
    """Local differential IK using the same MJCF as ``UR5GripperEnv``.

    Only the first six velocity DOFs (arm hinges) are integrated; gripper + box DOFs are held at the
    values from the posture reference ``q_posture`` each step (matches how policies latch non-driven joints).
    """

    model: mujoco.MjModel
    tool_body: str = "tool0"
    inner_iters: int = 40
    control_dt: float = 0.02
    position_cost: float = 2.0
    orientation_cost: float = 0.12
    frame_lm_damping: float = 0.25
    posture_cost: float = 5e-4
    damping_cost: float = 1e-3
    configuration_limit_gain: float = 0.95
    solver_damping: float = 1e-2

    def __post_init__(self) -> None:
        check_ik_dependencies()
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.tool_body)
        if bid < 0:
            raise RuntimeError(f"MJCF missing body {self.tool_body!r}")

    def track_tool_se3_trajectory(
        self,
        q_init: NDArray[np.float64],
        targets_se3: NDArray[np.float64],
        posture_q: NDArray[np.float64],
        *,
        warm_start: bool = True,
        rolling_posture: bool = False,
    ) -> NDArray[np.float64]:
        """Return ``q`` trajectory with one row per target row (same length as ``targets_se3``).

        Row ``k`` is the configuration **after** converging toward ``targets_se3[k]`` (chain warm-starts
        from row ``k-1`` when ``warm_start``).

        ``posture_q[k]`` is the full ``nq`` reference used by the soft ``PostureTask`` (typically the policy
        ``qpos`` at the same index).

        When ``rolling_posture=True``, the posture reference is updated to the solved ``q`` after each
        outer step (``posture_q[k+1] = q_solved_k``).  The caller-supplied ``posture_q`` provides only the
        initial reference for step 0; subsequent steps anchor to the previous solution, encouraging small
        incremental joint changes and preventing null-space drift across long trajectories.
        """
        from mink import (  # noqa: PLC0415
            Configuration,
            ConfigurationLimit,
            DampingTask,
            FrameTask,
            PostureTask,
            SE3,
            solve_ik,
        )

        T = int(targets_se3.shape[0])
        posture_q = np.asarray(posture_q, dtype=np.float64)
        if posture_q.ndim == 1:
            posture_q = np.tile(posture_q.reshape(1, -1), (T, 1))
        if posture_q.shape != (T, int(self.model.nq)):
            raise ValueError(
                f"posture_q must be shape ({T}, {self.model.nq}) or ({self.model.nq},), got {posture_q.shape}"
            )
        out = np.empty((T, int(self.model.nq)), dtype=np.float64)
        q = np.asarray(q_init, dtype=np.float64).reshape(-1).copy()
        if q.shape[0] != int(self.model.nq):
            raise ValueError(f"q_init length {q.shape[0]} != model.nq {self.model.nq}")

        # rolling_posture: start with caller's reference; updated to solved q after each outer step.
        rolling_ref = np.asarray(posture_q[0], dtype=np.float64).reshape(-1).copy()

        lim = ConfigurationLimit(self.model, gain=float(self.configuration_limit_gain))
        dt = float(self.control_dt)

        for k in range(T):
            if not warm_start and k == 0:
                q = np.asarray(q_init, dtype=np.float64).reshape(-1).copy()
            cfg = Configuration(self.model, q)
            task = FrameTask(
                self.tool_body,
                "body",
                position_cost=float(self.position_cost),
                orientation_cost=float(self.orientation_cost),
                lm_damping=float(self.frame_lm_damping),
            )
            task.set_target(SE3.from_matrix(np.asarray(targets_se3[k], dtype=np.float64)))
            post = PostureTask(self.model, cost=float(self.posture_cost))
            post_ref = rolling_ref if rolling_posture else np.asarray(posture_q[k], dtype=np.float64).reshape(-1)
            post.set_target(post_ref)
            damp = DampingTask(self.model, cost=float(self.damping_cost))
            for _ in range(int(self.inner_iters)):
                v = solve_ik(
                    cfg,
                    [task, post, damp],
                    dt=dt,
                    solver="daqp",
                    limits=[lim],
                    damping=float(self.solver_damping),
                )
                v = np.asarray(v, dtype=np.float64).reshape(-1)
                v[6:] = 0.0
                cfg.integrate_inplace(v, dt)
            q = np.asarray(cfg.data.qpos, dtype=np.float64).copy()
            # Keep passive / non-arm coordinates identical to the policy reference (numerical drift guard).
            q[6:] = np.asarray(posture_q[k], dtype=np.float64).reshape(-1)[6:]
            out[k] = q
            if rolling_posture:
                rolling_ref = q.copy()
        return out

    def step_toward_tool_se3(
        self,
        q_current: NDArray[np.float64],
        target_se3: NDArray[np.float64],
        posture_q: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply ``inner_iters`` differential-IK substeps from ``q_current`` toward one ``4×4`` target.

        Gripper + box ``qpos`` rows are reset from ``posture_q[6:]`` after solving (same convention as
        :meth:`track_tool_se3_trajectory`).
        """
        from mink import (  # noqa: PLC0415
            Configuration,
            ConfigurationLimit,
            DampingTask,
            FrameTask,
            PostureTask,
            SE3,
            solve_ik,
        )

        q = np.asarray(q_current, dtype=np.float64).reshape(-1).copy()
        if q.shape[0] != int(self.model.nq):
            raise ValueError(f"q_current length {q.shape[0]} != model.nq {self.model.nq}")
        posture_q = np.asarray(posture_q, dtype=np.float64).reshape(-1).copy()
        if posture_q.shape[0] != int(self.model.nq):
            raise ValueError("posture_q must have length model.nq")

        lim = ConfigurationLimit(self.model, gain=float(self.configuration_limit_gain))
        dt = float(self.control_dt)
        cfg = Configuration(self.model, q)
        task = FrameTask(
            self.tool_body,
            "body",
            position_cost=float(self.position_cost),
            orientation_cost=float(self.orientation_cost),
            lm_damping=float(self.frame_lm_damping),
        )
        task.set_target(SE3.from_matrix(np.asarray(target_se3, dtype=np.float64)))
        post = PostureTask(self.model, cost=float(self.posture_cost))
        post.set_target(posture_q)
        damp = DampingTask(self.model, cost=float(self.damping_cost))
        for _ in range(int(self.inner_iters)):
            v = solve_ik(
                cfg,
                [task, post, damp],
                dt=dt,
                solver="daqp",
                limits=[lim],
                damping=float(self.solver_damping),
            )
            v = np.asarray(v, dtype=np.float64).reshape(-1)
            v[6:] = 0.0
            cfg.integrate_inplace(v, dt)
        q_out = np.asarray(cfg.data.qpos, dtype=np.float64).copy()
        q_out[6:] = posture_q[6:]
        return q_out

    def position_errors_vs_targets(
        self,
        q_trajectory: NDArray[np.float64],
        targets_se3: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Per-row Euclidean error (m) between FK ``tool0`` position and ``targets_se3[:, :3, 3]``."""
        data = mujoco.MjData(self.model)
        errs = np.empty((q_trajectory.shape[0],), dtype=np.float64)
        for i in range(q_trajectory.shape[0]):
            data.qpos[:] = q_trajectory[i]
            mujoco.mj_forward(self.model, data)
            p = tool0_position(self.model, data)
            t = np.asarray(targets_se3[i, :3, 3], dtype=np.float64).reshape(3)
            errs[i] = float(np.linalg.norm(p - t))
        return errs
