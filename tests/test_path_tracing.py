"""Validation tests for the ``path_tracing`` policy.

Tests are structured to validate each layer of the implementation:
  1. Cube geometry helpers — corner positions, SE3 target shapes, path validity.
  2. IK quality — approach and cube trace position errors against precomputed plan.
  3. Policy integration — runs N steps, checks gripper latch and tool0 trajectory.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

from robot_manipulation_sim.env import UR5GripperEnv
from robot_manipulation_sim.ik.tool_pose import tool0_se3_matrix

pytest.importorskip("mink")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_policy_module():
    path = Path(__file__).resolve().parents[1] / "policies" / "impl" / "path_tracing" / "path_tracing.py"
    spec = importlib.util.spec_from_file_location("path_tracing", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _actuator_qadr(env: UR5GripperEnv, act_idx: int) -> int:
    jid = int(env.model.actuator_trnid[act_idx, 0])
    return int(env.model.jnt_qposadr[jid])


@pytest.fixture(scope="module")
def mod():
    return _load_policy_module()


@pytest.fixture(scope="module")
def env_obs():
    env = UR5GripperEnv(enable_rgb=False, seed=0)
    obs = env.reset(box_xy_noise=0.0)
    return env, obs


# ---------------------------------------------------------------------------
# 1. Geometry helpers
# ---------------------------------------------------------------------------

class TestCubeGeometry:
    def test_corner_count_and_shape(self, mod):
        corners = mod.cube_corners_world()
        assert corners.shape == (8, 3)

    def test_corner_0_at_cube_corner0_world(self, mod):
        corners = mod.cube_corners_world()
        np.testing.assert_allclose(corners[0], mod.CUBE_CORNER0_WORLD, atol=1e-9)

    def test_all_corners_offset_by_side_m(self, mod):
        """Every corner pair that differs in exactly one unit index differs by CUBE_SIDE_M."""
        unit = mod._UNIT_CORNERS
        corners = mod.cube_corners_world()
        for i in range(8):
            for j in range(i + 1, 8):
                diff_units = np.sum(unit[i] != unit[j])
                if diff_units == 1:
                    dist = float(np.linalg.norm(corners[i] - corners[j]))
                    assert abs(dist - mod.CUBE_SIDE_M) < 1e-9, \
                        f"corners {i}↔{j}: expected side {mod.CUBE_SIDE_M} m, got {dist}"

    def test_path_only_uses_cube_edges(self, mod):
        """Every consecutive pair in _PATH_CORNER_INDICES is an adjacent cube corner."""
        unit = mod._UNIT_CORNERS
        path = mod._PATH_CORNER_INDICES
        for k in range(len(path) - 1):
            diff = int(np.sum(unit[path[k]] != unit[path[k + 1]]))
            assert diff == 1, \
                f"path step {k}: corners {path[k]}→{path[k+1]} are not adjacent (diff={diff})"

    def test_path_covers_all_12_edges(self, mod):
        """All 12 unique cube edges appear at least once in _PATH_CORNER_INDICES."""
        unit = mod._UNIT_CORNERS
        path = mod._PATH_CORNER_INDICES
        traversed: set[frozenset] = set()
        for k in range(len(path) - 1):
            traversed.add(frozenset({path[k], path[k + 1]}))
        # Enumerate all 12 unit-cube edges (pairs differing in one coordinate)
        all_edges: set[frozenset] = set()
        for i in range(8):
            for j in range(i + 1, 8):
                if np.sum(unit[i] != unit[j]) == 1:
                    all_edges.add(frozenset({i, j}))
        assert len(all_edges) == 12
        missing = all_edges - traversed
        assert not missing, f"path misses edges: {missing}"

    def test_approach_targets_shape(self, mod, env_obs):
        env, obs = env_obs
        data0 = mujoco.MjData(env.model)
        data0.qpos[:] = obs["qpos"]
        mujoco.mj_forward(env.model, data0)
        T0 = tool0_se3_matrix(env.model, data0)
        targets = mod._approach_se3_targets(T0[:3, 3], T0[:3, :3], 50)
        assert targets.shape == (50, 4, 4)
        # Last target lands on CUBE_CORNER0_WORLD
        np.testing.assert_allclose(targets[-1, :3, 3], mod.CUBE_CORNER0_WORLD, atol=1e-9)

    def test_cube_trace_targets_shape_and_endpoints(self, mod, env_obs):
        env, obs = env_obs
        data0 = mujoco.MjData(env.model)
        data0.qpos[:] = obs["qpos"]
        mujoco.mj_forward(env.model, data0)
        T0 = tool0_se3_matrix(env.model, data0)
        targets = mod._cube_trace_se3_targets(T0[:3, :3], 160)
        assert targets.shape == (160, 4, 4)
        # First target is ε past corner 0 (t=n_segs/n_steps > 0 so not exactly corner 0)
        # Last target should land on the final path corner (index PATH[-1])
        corners = mod.cube_corners_world()
        final_corner = corners[mod._PATH_CORNER_INDICES[-1]]
        np.testing.assert_allclose(targets[-1, :3, 3], final_corner, atol=1e-9)

    def test_orientation_constant_throughout(self, mod, env_obs):
        env, obs = env_obs
        data0 = mujoco.MjData(env.model)
        data0.qpos[:] = obs["qpos"]
        mujoco.mj_forward(env.model, data0)
        T0 = tool0_se3_matrix(env.model, data0)
        R0 = T0[:3, :3]
        for targets in [
            mod._approach_se3_targets(T0[:3, 3], R0, 30),
            mod._cube_trace_se3_targets(R0, 48),
        ]:
            for t in targets:
                np.testing.assert_allclose(t[:3, :3], R0, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. IK quality
# ---------------------------------------------------------------------------

class TestIkQuality:
    """Verify IK planning errors stay within acceptable bounds.

    The policy uses tracking IK (inner_iters=1, rolling_posture=True): each outer step makes
    a single Newton step toward the target, with the posture reference updated to the previous
    solved q.  All tests here mirror this setup: they call track_tool_se3_trajectory with
    rolling_posture=True and use step counts proportional to the actual policy so that per-step
    target distances are representative.
    """

    @pytest.fixture(autouse=True)
    def setup(self, mod, env_obs):
        self.mod = mod
        self.env, obs = env_obs
        data0 = mujoco.MjData(self.env.model)
        data0.qpos[:] = obs["qpos"]
        mujoco.mj_forward(self.env.model, data0)
        T0 = tool0_se3_matrix(self.env.model, data0)
        self.p0 = T0[:3, 3].copy()
        self.R0 = T0[:3, :3].copy()
        self.q0 = np.asarray(obs["qpos"], dtype=np.float64).copy()

    def _make_ik(self):
        from robot_manipulation_sim.ik.service import MujocoMinkIkService
        return MujocoMinkIkService(
            self.env.model,
            inner_iters=self.mod.IK_INNER_ITERS,
            position_cost=self.mod.IK_POSITION_COST,
            orientation_cost=self.mod.IK_ORIENTATION_COST,
            frame_lm_damping=self.mod.IK_FRAME_LM_DAMPING,
            posture_cost=self.mod.IK_POSTURE_COST,
            damping_cost=self.mod.IK_DAMPING_COST,
        )

    def test_approach_ik_max_position_error(self):
        """Approach: use actual policy step count (n_app=125) so per-step distance matches."""
        n_app = 125
        targets = self.mod._approach_se3_targets(self.p0, self.R0, n_app)
        ik = self._make_ik()
        qp = ik.track_tool_se3_trajectory(self.q0, targets, self.q0, rolling_posture=True)
        errs = ik.position_errors_vs_targets(qp, targets)
        assert float(np.max(errs)) < 5e-3, \
            f"approach IK max error {np.max(errs)*1000:.2f} mm exceeds 5 mm"

    def test_cube_trace_ik_max_position_error(self):
        """Two-phase solve matching policy step counts (n_app=125, n_trace=375).

        Tracking IK accuracy depends on per-step target distance; using actual counts ensures
        the test is representative of real policy behavior.
        """
        n_app, n_trace = 125, 375
        app_targets = self.mod._approach_se3_targets(self.p0, self.R0, n_app)
        ik = self._make_ik()
        q_app = ik.track_tool_se3_trajectory(
            self.q0, app_targets, self.q0, rolling_posture=True
        )

        trace_targets = self.mod._cube_trace_se3_targets(self.R0, n_trace)
        q_trace = ik.track_tool_se3_trajectory(
            q_app[-1], trace_targets, q_app[-1], rolling_posture=True
        )
        errs = ik.position_errors_vs_targets(q_trace, trace_targets)
        assert float(np.max(errs)) < 5e-3, \
            f"cube trace IK max error {np.max(errs)*1000:.2f} mm exceeds 5 mm"

    def test_all_cube_corners_reachable(self):
        """Each of the 8 cube corners is reachable by tracking from corner 0 in ≤50 small steps.

        Tracking IK (inner_iters=1) follows a sequence of small targets; testing corner
        reachability in one step would incorrectly measure convergence speed, not reachability.
        Instead, we interpolate 50 steps from corner 0 to each target corner and verify the
        final position error is within 5 mm.
        """
        n_app = 125
        app_targets = self.mod._approach_se3_targets(self.p0, self.R0, n_app)
        ik = self._make_ik()
        q_c0 = ik.track_tool_se3_trajectory(
            self.q0, app_targets, self.q0, rolling_posture=True
        )[-1]

        corners = self.mod.cube_corners_world()
        n_interp = 50   # small enough for test speed; large enough for accurate tracking
        max_err = 0.0
        for i, corner in enumerate(corners):
            t = np.arange(1, n_interp + 1, dtype=np.float64) / n_interp
            positions = corners[0][np.newaxis] + t[:, np.newaxis] * (corner - corners[0])
            tgts = np.tile(np.eye(4, dtype=np.float64), (n_interp, 1, 1))
            tgts[:, :3, :3] = self.R0
            tgts[:, :3, 3] = positions
            q_seq = ik.track_tool_se3_trajectory(q_c0, tgts, q_c0, rolling_posture=True)
            final_err = ik.position_errors_vs_targets(q_seq[-1:], tgts[-1:])
            max_err = max(max_err, float(final_err[0]))

        assert max_err < 5e-3, \
            f"max corner reachability error {max_err*1000:.2f} mm exceeds 5 mm"


# ---------------------------------------------------------------------------
# 3. Policy integration
# ---------------------------------------------------------------------------

class TestPolicyIntegration:
    N_STEPS = 150   # enough to cover approach (125 steps) and start of trace

    @pytest.fixture(autouse=True)
    def setup(self, mod):
        self.mod = mod
        reset_fn = getattr(mod, "reset", None)
        if callable(reset_fn):
            reset_fn()
        self.env = UR5GripperEnv(enable_rgb=False, seed=0)
        self.obs = self.env.reset(box_xy_noise=0.0)

    def _actuator_qadr(self, act_idx):
        return _actuator_qadr(self.env, act_idx)

    def test_policy_returns_correct_ctrl_shape(self):
        ctrl = self.mod.policy(self.obs, 0, self.env)
        assert ctrl.shape == (self.env.nu,), f"Expected ({self.env.nu},), got {ctrl.shape}"

    def test_gripper_ctrl_held_at_latch(self):
        """Gripper actuator (tendon type) must hold the episode-start setpoint every step."""
        gripper_acts = [
            i for i in range(self.env.nu)
            if int(self.env.model.actuator_trntype[i]) != 0  # not mjTRN_JOINT
        ]
        obs = self.obs
        q0_gripper = {i: float(obs["ctrl"][i]) for i in gripper_acts}
        for step in range(self.N_STEPS):
            ctrl = self.mod.policy(obs, step, self.env)
            for i in gripper_acts:
                assert abs(float(ctrl[i]) - q0_gripper[i]) < 1e-5, \
                    f"gripper actuator {i} drifted at step {step}"
            obs = self.env.step(ctrl)

    def test_tool0_reaches_cube_corner0_after_approach(self):
        """After completing the approach phase, tool0 should be within 15 mm of CUBE_CORNER0_WORLD."""
        n_approach = max(1, round(self.mod.APPROACH_DURATION_S / float(self.env.control_dt)))
        obs = self.obs
        for step in range(n_approach):
            ctrl = self.mod.policy(obs, step, self.env)
            obs = self.env.step(ctrl)

        data = mujoco.MjData(self.env.model)
        data.qpos[:] = obs["qpos"]
        mujoco.mj_forward(self.env.model, data)
        T = tool0_se3_matrix(self.env.model, data)
        dist = float(np.linalg.norm(T[:3, 3] - self.mod.CUBE_CORNER0_WORLD))
        assert dist < 0.015, \
            f"tool0 is {dist*1000:.1f} mm from cube corner 0 after approach (limit: 15 mm)"

    def test_ctrl_within_actuator_limits(self):
        """Every ctrl command must be within the model's ctrlrange."""
        lo = self.env.model.actuator_ctrlrange[:, 0]
        hi = self.env.model.actuator_ctrlrange[:, 1]
        obs = self.obs
        for step in range(self.N_STEPS):
            ctrl = self.mod.policy(obs, step, self.env)
            for i in range(self.env.nu):
                assert lo[i] <= ctrl[i] <= hi[i], \
                    f"actuator {i} ctrl={ctrl[i]:.4f} outside [{lo[i]:.4f}, {hi[i]:.4f}] at step {step}"
            obs = self.env.step(ctrl)
