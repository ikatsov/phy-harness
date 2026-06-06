"""How ``a_gripper`` ctrl (0–255) maps to physical finger geometry in this MJCF.

The Robotiq 2F-85 is driven by a *general* actuator on the ``split`` tendon
(``ur5e_two_finger_scene.xml``). Intuition from gain/bias parameters alone is
easy to get wrong; these tests **measure** settled pad–pad opening in the
``tool0`` frame after dynamics converge.

Method
------
- Hold the default arm pose (same position actuators as ``UR5GripperEnv._home``).
- Set only the gripper actuator command, then step until driver joint speeds are
  small (or a step budget is hit).
- ``finger_opening_width_m`` = magnitude of the dominant axis component of
  ``R_tool0^T (xpos(left_pad) - xpos(right_pad))`` (pads are almost separated
  along one tool axis).

Expected for the bundled scene (``forcerange="-100 100"``)
---------------------------------------------------------
Opening width **decreases** as ``ctrl`` increases from 0 toward 255: low
``ctrl`` ≈ fingers more open, high ``ctrl`` ≈ fingers more closed. The
default env home gripper command should therefore match the low end if policies
assume ``0`` means “release / spread”.
"""

from __future__ import annotations

import numpy as np
import pytest
import mujoco

from robot_manipulation_sim.env import UR5GripperEnv


def _body_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        pytest.fail(f"MJCF missing body {name!r}")
    return bid


def _gripper_actuator_index(model: mujoco.MjModel) -> int:
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name == "a_gripper":
            return i
    pytest.fail("MJCF missing actuator 'a_gripper'")


def _driver_dof_indices(model: mujoco.MjModel) -> tuple[int, int]:
    """Dofs for symmetric driver joints (used for settling / velocity checks)."""
    out: list[int] = []
    for jname in ("left_driver_joint", "right_driver_joint"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            pytest.fail(f"MJCF missing joint {jname!r}")
        out.append(int(model.jnt_dofadr[jid]))
    return out[0], out[1]


def finger_opening_width_m(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Scalar pad–pad opening (m): separation in ``tool0`` along the dominant local axis."""
    bl = _body_id(model, "left_pad")
    br = _body_id(model, "right_pad")
    bt = _body_id(model, "tool0")
    w = data.xpos[bl] - data.xpos[br]
    R = data.xmat[bt].reshape(3, 3)
    local = R.T @ w
    ax = int(np.argmax(np.abs(local)))
    return float(abs(local[ax]))


def settle_at_gripper_ctrl(
    env: UR5GripperEnv,
    gripper_ctrl: float,
    *,
    max_ctrl_steps: int = 5000,
    min_ctrl_steps: int = 200,
    vel_thresh: float = 2.5e-3,
    stable_need: int = 30,
) -> None:
    """Hold arm at ``env._home`` arm targets; command gripper; step until quasistatic."""
    g_idx = _gripper_actuator_index(env.model)
    ctrl = np.asarray(env._home[: env.nu], dtype=np.float64).copy()
    ctrl[g_idx] = float(np.clip(gripper_ctrl, *env.model.actuator_ctrlrange[g_idx]))
    env.set_control(ctrl)
    d0, d1 = _driver_dof_indices(env.model)
    stable = 0
    for k in range(max_ctrl_steps):
        env.step(None)
        if k + 1 < min_ctrl_steps:
            continue
        v = max(abs(float(env.data.qvel[d0])), abs(float(env.data.qvel[d1])))
        if v < vel_thresh:
            stable += 1
            if stable >= stable_need:
                return
        else:
            stable = 0


@pytest.fixture(scope="module")
def env_norgb() -> UR5GripperEnv:
    return UR5GripperEnv(enable_rgb=False, seed=0)


def test_default_home_gripper_matches_max_open_command(env_norgb: UR5GripperEnv) -> None:
    """Policies treat low ``ctrl`` as open; env home must not start partially closed."""
    g_idx = _gripper_actuator_index(env_norgb.model)
    assert float(env_norgb._home[g_idx]) == pytest.approx(0.0, abs=1e-9)


def test_ctrl_zero_more_open_than_ctrl_255(env_norgb: UR5GripperEnv) -> None:
    """End-point geometry: settled ``ctrl=0`` wider pad opening than ``ctrl=255``."""
    m, d = env_norgb.model, env_norgb.data
    env_norgb.reset(box_xy_noise=0.0)
    settle_at_gripper_ctrl(env_norgb, 0.0)
    w0 = finger_opening_width_m(m, d)

    env_norgb.reset(box_xy_noise=0.0)
    settle_at_gripper_ctrl(env_norgb, 255.0)
    w255 = finger_opening_width_m(m, d)

    assert w0 > w255 + 0.002, (
        f"expected ctrl=0 to be strictly more open than ctrl=255; got w0={w0:.5f} w255={w255:.5f}"
    )


@pytest.mark.parametrize(
    "low_c,high_c",
    [(0, 32), (32, 64), (64, 96), (96, 128)],
    ids=["0_ge_32", "32_ge_64", "64_ge_96", "96_ge_128"],
)
def test_opening_nonincreasing_low_ctrl_band(
    env_norgb: UR5GripperEnv, low_c: int, high_c: int
) -> None:
    """Monotone band 0–128: higher ``ctrl`` → narrower opening (no multistable jumps here)."""
    m, d = env_norgb.model, env_norgb.data

    env_norgb.reset(box_xy_noise=0.0)
    settle_at_gripper_ctrl(env_norgb, float(low_c))
    w_lo = finger_opening_width_m(m, d)

    env_norgb.reset(box_xy_noise=0.0)
    settle_at_gripper_ctrl(env_norgb, float(high_c))
    w_hi = finger_opening_width_m(m, d)

    assert w_lo >= w_hi - 5e-4, (
        f"expected opening({low_c}) >= opening({high_c}); got {w_lo:.5f} vs {w_hi:.5f}"
    )


def test_ctrl_220_is_not_maximally_open(env_norgb: UR5GripperEnv) -> None:
    """Regression: ``220`` was historically used as ``RELEASE`` but is not the widest opening."""
    m, d = env_norgb.model, env_norgb.data

    env_norgb.reset(box_xy_noise=0.0)
    settle_at_gripper_ctrl(env_norgb, 0.0)
    w0 = finger_opening_width_m(m, d)

    env_norgb.reset(box_xy_noise=0.0)
    settle_at_gripper_ctrl(env_norgb, 220.0)
    w220 = finger_opening_width_m(m, d)

    assert w0 > w220 + 0.002, (
        f"ctrl=0 should be more open than ctrl=220; got w0={w0:.5f} w220={w220:.5f}"
    )
