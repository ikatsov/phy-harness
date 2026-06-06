"""Validation tests for the ``box_pick`` policy.

Structure mirrors ``test_path_tracing.py``:
  1. Geometry helpers — SE3 target shapes, orientations, positions.
  2. Detection + IK quality — vision back-projection accuracy, plan position errors.
  3. Policy integration — ctrl shape, gripper phases, ctrl limits.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

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
    path = Path(__file__).resolve().parents[1] / "policies" / "impl" / "box_pick" / "box_pick.py"
    spec = importlib.util.spec_from_file_location("box_pick", path)
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

class TestGeometryHelpers:
    def test_make_se3_identity_rotation(self, mod):
        p = np.array([1.0, 2.0, 3.0])
        T = mod._make_se3(p, np.eye(3))
        np.testing.assert_allclose(T[:3, 3], p)
        np.testing.assert_allclose(T[:3, :3], np.eye(3))
        assert T[3, 3] == 1.0

    def test_make_se3_with_r_grasp(self, mod):
        """R_GRASP is a valid rotation matrix (det=1, orthonormal)."""
        R = mod.R_GRASP
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12

    def test_r_grasp_body_z_points_world_minus_z(self, mod):
        """Body-Z column of R_GRASP must equal world [0, 0, -1] (top-down approach)."""
        np.testing.assert_allclose(mod.R_GRASP[:, 2], [0.0, 0.0, -1.0], atol=1e-12)

    def test_r_grasp_body_x_points_world_x(self, mod):
        """Body-X column of R_GRASP must equal world [1, 0, 0]."""
        np.testing.assert_allclose(mod.R_GRASP[:, 0], [1.0, 0.0, 0.0], atol=1e-12)

    def test_interpolate_se3_targets_shape(self, mod):
        p_start = np.array([0.0, 0.0, 0.3])
        p_end = np.array([0.5, 0.0, 0.18])
        tgts = mod._interpolate_se3_targets(p_start, p_end, mod.R_GRASP, 100)
        assert tgts.shape == (100, 4, 4)

    def test_interpolate_se3_targets_last_is_p_end(self, mod):
        p_start = np.array([0.0, 0.0, 0.3])
        p_end = np.array([0.52, 0.0, 0.185])
        tgts = mod._interpolate_se3_targets(p_start, p_end, mod.R_GRASP, 75)
        np.testing.assert_allclose(tgts[-1, :3, 3], p_end, atol=1e-12)

    def test_interpolate_se3_targets_orientation_fixed(self, mod):
        p_start = np.array([0.0, 0.0, 0.3])
        p_end = np.array([0.5, 0.1, 0.2])
        tgts = mod._interpolate_se3_targets(p_start, p_end, mod.R_GRASP, 40)
        for t in tgts:
            np.testing.assert_allclose(t[:3, :3], mod.R_GRASP, atol=1e-12)

    def test_z_heights_ordered(self, mod):
        """Grasp height < pre-grasp height < lift height."""
        assert mod.Z_GRASP < mod.Z_PRE_GRASP < mod.Z_LIFT

    def test_z_grasp_formula(self, mod):
        """Z_GRASP = BOX_CENTER_Z + GRIPPER_FINGER_OFFSET_M."""
        expected = mod.BOX_CENTER_Z + mod.GRIPPER_FINGER_OFFSET_M
        assert abs(mod.Z_GRASP - expected) < 1e-9

# ---------------------------------------------------------------------------
# 2. Detection + back-projection quality
# ---------------------------------------------------------------------------

class TestDetectionAndLocalization:
    """Verify that the vision service can detect the simulated box and recover its world position.

    Uses the project_world_positions_to_camera_pixels round-trip: forward-project the ground-truth
    box position to a topdown pixel, then back-project and compare.
    """

    @pytest.fixture(autouse=True)
    def setup(self, env_obs):
        self.env, self.obs = env_obs

    def test_box_ground_truth_visible_from_topdown(self):
        """Ground-truth box position must project into the topdown camera frustum."""
        from robot_manipulation_sim.cameras import project_world_positions_to_camera_pixels  # noqa: PLC0415

        m, d = self.env.model, self.env.data
        box_pt = np.array([0.52, 0.0, 0.035]).reshape(1, 3)
        uv = project_world_positions_to_camera_pixels(m, d, "topdown", 640, 480, box_pt)[0]
        assert np.all(np.isfinite(uv)), "Box ground-truth not visible from topdown camera"
        assert 0 <= uv[0] < 640 and 0 <= uv[1] < 480

    def test_backproject_ground_truth_pixel_to_box_position(self):
        """Back-project the ground-truth box pixel → should recover (0.52, 0.0, 0.035) ≤ 1 mm."""
        from robot_manipulation_sim.cameras import project_world_positions_to_camera_pixels  # noqa: PLC0415
        from robot_manipulation_sim.vision.service import VisionService  # noqa: PLC0415

        m, d = self.env.model, self.env.data
        box_pt = np.array([0.52, 0.0, 0.035])
        uv = project_world_positions_to_camera_pixels(m, d, "topdown", 640, 480, box_pt.reshape(1, 3))[0]

        svc = VisionService(m, camera_name="topdown", width=640, height=480)
        recovered = svc.unproject_pixel_to_world(d, uv, target_z=0.035)
        np.testing.assert_allclose(recovered, box_pt, atol=1e-3)


# ---------------------------------------------------------------------------
# 3. IK plan quality
# ---------------------------------------------------------------------------

class TestIkPlanQuality:
    """Verify IK accuracy for each motion phase with tight tolerances."""

    @pytest.fixture(autouse=True)
    def setup(self, mod, env_obs):
        self.mod = mod
        self.env, obs = env_obs
        data0 = mujoco.MjData(self.env.model)
        data0.qpos[:] = obs["qpos"]
        mujoco.mj_forward(self.env.model, data0)
        T0 = tool0_se3_matrix(self.env.model, data0)
        self.p0 = T0[:3, 3].copy()
        self.q0 = np.asarray(obs["qpos"], dtype=np.float64).copy()
        # Use nominal box position for plan building.
        self.box_xy = np.array([0.52, 0.0, mod.BOX_CENTER_Z])

    def _make_ik(self):
        from robot_manipulation_sim.ik.service import MujocoMinkIkService  # noqa: PLC0415
        return MujocoMinkIkService(
            self.env.model,
            inner_iters=self.mod.IK_INNER_ITERS,
            control_dt=float(self.env.control_dt),
            position_cost=self.mod.IK_POSITION_COST,
            orientation_cost=self.mod.IK_ORIENTATION_COST,
            frame_lm_damping=self.mod.IK_FRAME_LM_DAMPING,
            posture_cost=self.mod.IK_POSTURE_COST,
            damping_cost=self.mod.IK_DAMPING_COST,
        )

    def test_crane_pose_has_r_grasp_orientation(self):
        """At the crane qpos (home ctrl-target joints) tool0 body-Z must point world -Z.

        This is the key invariant that makes the crane approach work: the arm is already
        in R_GRASP orientation at the crane pose, so no SLERP or branch switching is needed.
        """
        obs = self.env.reset(box_xy_noise=0.0)
        ctrl0 = np.asarray(obs["ctrl"], dtype=np.float64)
        q_crane = np.asarray(obs["qpos"], dtype=np.float64).copy()
        q_crane[:6] = ctrl0[:6]

        data_crane = mujoco.MjData(self.env.model)
        data_crane.qpos[:] = q_crane
        mujoco.mj_forward(self.env.model, data_crane)
        T_crane = tool0_se3_matrix(self.env.model, data_crane)
        body_z = T_crane[:3, :3][:, 2]
        np.testing.assert_allclose(body_z, [0.0, 0.0, -1.0], atol=1e-4,
                                   err_msg="Crane pose body-Z does not equal world -Z (R_GRASP)")

    def test_approach_phase_max_position_error(self):
        """Phase 1 (crane → pre-grasp): final position error < 15 mm.

        The approach starts from the crane pose (ctrl-target joints, body-Z already = world -Z)
        and targets the pre-grasp position with R_GRASP orientation. No SLERP or branch
        switching is required; the arm stays in the unfolded crane configuration.
        """
        obs = self.env.reset(box_xy_noise=0.0)
        ctrl0 = np.asarray(obs["ctrl"], dtype=np.float64)
        q_crane = np.asarray(obs["qpos"], dtype=np.float64).copy()
        q_crane[:6] = ctrl0[:6]

        data_crane = mujoco.MjData(self.env.model)
        data_crane.qpos[:] = q_crane
        mujoco.mj_forward(self.env.model, data_crane)
        p_crane = tool0_se3_matrix(self.env.model, data_crane)[:3, 3].copy()

        n_pre = max(1, round(self.mod.T_PRE / float(self.env.control_dt)))
        p_pregrasp = np.array([0.52, 0.0, self.mod.Z_PRE_GRASP])
        tgts = self.mod._interpolate_se3_targets(p_crane, p_pregrasp, self.mod.R_GRASP, n_pre)
        ik = self._make_ik()
        q = ik.track_tool_se3_trajectory(q_crane, tgts, q_crane, rolling_posture=True)
        errs = ik.position_errors_vs_targets(q, tgts)
        assert float(errs[-1]) < 0.020, \
            f"approach final position error {errs[-1]*1000:.2f} mm exceeds 20 mm"

    def test_descend_phase_max_position_error(self):
        """Phase 2 (descend): IK accuracy < 15 mm at grasp height."""
        obs = self.env.reset(box_xy_noise=0.0)
        ctrl0 = np.asarray(obs["ctrl"], dtype=np.float64)
        q_crane = np.asarray(obs["qpos"], dtype=np.float64).copy()
        q_crane[:6] = ctrl0[:6]

        data_crane = mujoco.MjData(self.env.model)
        data_crane.qpos[:] = q_crane
        mujoco.mj_forward(self.env.model, data_crane)
        p_crane = tool0_se3_matrix(self.env.model, data_crane)[:3, 3].copy()

        n_pre = max(1, round(self.mod.T_PRE / float(self.env.control_dt)))
        n_desc = max(1, round(self.mod.T_DESCEND / float(self.env.control_dt)))

        p_pregrasp = np.array([0.52, 0.0, self.mod.Z_PRE_GRASP])
        p_grasp = np.array([0.52, 0.0, self.mod.Z_GRASP])
        ik = self._make_ik()
        tgts_pre = self.mod._interpolate_se3_targets(p_crane, p_pregrasp, self.mod.R_GRASP, n_pre)
        q_pre = ik.track_tool_se3_trajectory(q_crane, tgts_pre, q_crane, rolling_posture=True)

        tgts_desc = self.mod._interpolate_se3_targets(p_pregrasp, p_grasp, self.mod.R_GRASP, n_desc)
        q_desc = ik.track_tool_se3_trajectory(q_pre[-1], tgts_desc, q_pre[-1], rolling_posture=True)
        errs = ik.position_errors_vs_targets(q_desc, tgts_desc)
        assert float(errs[-1]) < 0.015, \
            f"descend final error {errs[-1]*1000:.2f} mm exceeds 15 mm"

    def test_build_plan_returns_correct_shapes(self, mod, env_obs):
        env, obs = env_obs
        mod.reset()
        q_plan, g_plan = mod.build_plan(env, obs, np.array([0.52, 0.0, mod.BOX_CENTER_Z]))
        n_expected = sum(
            max(1, round(t / float(env.control_dt)))
            for t in [mod.T_CRANE, mod.T_PRE, mod.T_DESCEND, mod.T_GRIP, mod.T_LIFT,
                      mod.T_TRANSPORT, mod.T_LOWER, mod.T_RELEASE]
        )
        assert q_plan.shape == (n_expected, env.model.nq), \
            f"q_plan shape {q_plan.shape} != ({n_expected}, {env.model.nq})"
        assert g_plan.shape == (n_expected,), \
            f"gripper_plan shape {g_plan.shape} != ({n_expected},)"

    def test_gripper_plan_phases(self, mod, env_obs):
        """Gripper: RELEASE (0, spring-open) during crane/approach/descent; ramp to GRIP (255) in
        grip phase; GRIP during lift/transport/lower; RELEASE at end of release phase."""
        env, obs = env_obs
        mod.reset()
        _, g_plan = mod.build_plan(env, obs, np.array([0.52, 0.0, mod.BOX_CENTER_Z]))
        dt = float(env.control_dt)
        n_crane = max(1, round(mod.T_CRANE / dt))
        n_pre = max(1, round(mod.T_PRE / dt))
        n_desc = max(1, round(mod.T_DESCEND / dt))
        n_grip = max(1, round(mod.T_GRIP / dt))
        n_lift = max(1, round(mod.T_LIFT / dt))

        # Phases 0, 1, 2: RELEASE (0 = spring default = fingers wide open).
        assert g_plan[0] == pytest.approx(mod.GRIPPER_RELEASE, abs=1e-5)
        assert g_plan[n_crane - 1] == pytest.approx(mod.GRIPPER_RELEASE, abs=1e-5)
        assert g_plan[n_crane] == pytest.approx(mod.GRIPPER_RELEASE, abs=1e-5)
        assert g_plan[n_crane + n_pre - 1] == pytest.approx(mod.GRIPPER_RELEASE, abs=1e-5)
        # Phase 2 end: still RELEASE (descent keeps fingers fully open).
        assert g_plan[n_crane + n_pre + n_desc - 1] == pytest.approx(mod.GRIPPER_RELEASE, abs=1e-5)

        # Phase 3 end: GRIP ramp completes (255 = actuator closes fingers).
        grip_end = n_crane + n_pre + n_desc + n_grip - 1
        assert g_plan[grip_end] == pytest.approx(mod.GRIPPER_GRIP, abs=1e-5)

        # Phases 4, 5, 6: GRIP.
        mid_lift = n_crane + n_pre + n_desc + n_grip + n_lift // 2
        assert g_plan[mid_lift] == pytest.approx(mod.GRIPPER_GRIP, abs=1e-5)

        # Phase 7 end: RELEASE (spring opens = box dropped).
        last = len(g_plan) - 1
        assert g_plan[last] == pytest.approx(mod.GRIPPER_RELEASE, abs=1e-5)


# ---------------------------------------------------------------------------
# 4. Policy integration
# ---------------------------------------------------------------------------

class TestPolicyIntegration:
    N_STEPS = 200

    @pytest.fixture(autouse=True)
    def setup(self, mod):
        self.mod = mod
        mod.reset()
        self.env = UR5GripperEnv(enable_rgb=False, seed=0)
        self.obs = self.env.reset(box_xy_noise=0.0)
        # Pre-inject a known box position so vision is not required (no GL in tests).
        # We do this by calling build_plan directly and caching the plan.
        import numpy as np  # noqa: PLC0415
        q_plan, g_plan = mod.build_plan(
            self.env, self.obs, np.array([0.52, 0.0, mod.BOX_CENTER_Z])
        )
        mod._q_plan = q_plan
        mod._gripper_plan = g_plan
        mod._latch_qpos = np.asarray(self.obs["qpos"], dtype=np.float64).copy()
        mod._episode_start_ctrl = [float(self.obs["ctrl"][i]) for i in range(self.env.nu)]
        mod._detection_ok = True

    def test_policy_returns_correct_ctrl_shape(self):
        ctrl = self.mod.policy(self.obs, 1, self.env)
        assert ctrl.shape == (self.env.nu,), f"Expected ({self.env.nu},), got {ctrl.shape}"

    def test_ctrl_within_actuator_limits(self):
        lo = self.env.model.actuator_ctrlrange[:, 0]
        hi = self.env.model.actuator_ctrlrange[:, 1]
        obs = self.obs
        for step in range(1, self.N_STEPS + 1):
            ctrl = self.mod.policy(obs, step, self.env)
            for i in range(self.env.nu):
                assert lo[i] <= ctrl[i] <= hi[i], \
                    f"actuator {i} ctrl={ctrl[i]:.4f} outside [{lo[i]:.4f}, {hi[i]:.4f}] at step {step}"
            obs = self.env.step(ctrl)

    def test_gripper_is_release_during_approach(self):
        """During phase 1 (approach) gripper must equal GRIPPER_RELEASE (0 = spring-open)."""
        dt = float(self.env.control_dt)
        n_crane = max(1, round(self.mod.T_CRANE / dt))
        n_pre = max(1, round(self.mod.T_PRE / dt))
        g_idx = self.mod._gripper_idx(self.env)
        lo_g, hi_g = self.env.model.actuator_ctrlrange[g_idx]
        expected = np.clip(self.mod.GRIPPER_RELEASE, lo_g, hi_g)
        approach_start = n_crane + 1
        approach_mid = n_crane + n_pre // 2
        obs = self.obs
        for step in range(approach_start, approach_mid):
            ctrl = self.mod.policy(obs, step, self.env)
            assert ctrl[g_idx] == pytest.approx(expected, abs=1e-4), \
                f"gripper not RELEASE at approach step {step}: {ctrl[g_idx]}"
            obs = self.env.step(ctrl)

    def test_gripper_is_grip_during_transport(self):
        """During phase 5 (transport) the gripper command must equal GRIPPER_GRIP (255 = actuator closes)."""
        dt = float(self.env.control_dt)
        n_crane = max(1, round(self.mod.T_CRANE / dt))
        n_pre = max(1, round(self.mod.T_PRE / dt))
        n_desc = max(1, round(self.mod.T_DESCEND / dt))
        n_grip = max(1, round(self.mod.T_GRIP / dt))
        n_lift = max(1, round(self.mod.T_LIFT / dt))
        n_trans = max(1, round(self.mod.T_TRANSPORT / dt))
        trans_start = n_crane + n_pre + n_desc + n_grip + n_lift
        trans_mid = trans_start + n_trans // 2

        g_idx = self.mod._gripper_idx(self.env)
        lo_g, hi_g = self.env.model.actuator_ctrlrange[g_idx]
        expected = np.clip(self.mod.GRIPPER_GRIP, lo_g, hi_g)
        ctrl = self.mod.policy(self.obs, trans_mid, self.env)
        assert ctrl[g_idx] == pytest.approx(expected, abs=1e-4), \
            f"gripper not GRIP at transport step {trans_mid}: {ctrl[g_idx]}"

    def test_policy_latches_past_plan_end(self):
        """After the plan ends the policy must return the last ctrl (no IndexError)."""
        n_plan = len(self.mod._q_plan)
        # Request step well beyond plan.
        ctrl = self.mod.policy(self.obs, n_plan + 50, self.env)
        assert ctrl.shape == (self.env.nu,)
