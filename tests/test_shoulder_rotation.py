"""Regression: shoulder_rotation — explicit tool0 hinge arc + mink IK; gripper latched."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

from robot_manipulation_sim import UR5GripperEnv

pytest.importorskip("mink")


def _load_policy_module():
    path = Path(__file__).resolve().parents[1] / "policies" / "impl" / "shoulder_rotation" / "shoulder_rotation.py"
    spec = importlib.util.spec_from_file_location("shoulder_rotation", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _actuator_qadr(env: UR5GripperEnv, act_idx: int) -> int:
    jid = int(env.model.actuator_trnid[act_idx, 0])
    return int(env.model.jnt_qposadr[jid])


def test_shoulder_rotation_explicit_arc_ik_and_gripper_latch():
    mod = _load_policy_module()
    policy = mod.policy
    reset_fn = getattr(mod, "reset", None)
    if callable(reset_fn):
        reset_fn()
    env = UR5GripperEnv(enable_rgb=False, seed=0)
    obs = env.reset(box_xy_noise=0.0)

    lift_act = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_shoulder_lift")
    assert lift_act >= 0

    def lift_angle(o):
        jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_lift_joint")
        qadr = int(env.model.jnt_qposadr[jid])
        return float(o["qpos"][qadr])

    q0_arm = {}
    for name in ("shoulder_pan_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"):
        jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr = int(env.model.jnt_qposadr[jid])
        q0_arm[name] = float(obs["qpos"][qadr])

    def latch_targets(o):
        out = []
        for j in range(env.nu):
            if int(env.model.actuator_trntype[j]) == 0:
                out.append(float(o["qpos"][_actuator_qadr(env, j)]))
            else:
                out.append(float(o["ctrl"][j]))
        return out

    q0_ctrl = latch_targets(obs)

    # Nominal hinge-angle advance per step along the explicit arc schedule; IK joint targets can deviate more.
    arc_step = float(mod.SHOULDER_ARC_TOTAL_RAD) * float(env.control_dt) / float(mod.SHOULDER_ARC_DURATION_S)
    slew_tol = 8.0 * arc_step + 0.12

    l0 = lift_angle(obs)
    for i in range(200):
        ctrl = policy(obs, i, env)
        for j in range(env.nu):
            if int(env.model.actuator_trntype[j]) == 0:
                qadr = _actuator_qadr(env, j)
                q = float(obs["qpos"][qadr])
                assert abs(ctrl[j] - q) <= slew_tol + 1e-5, f"actuator {j} joint target slew exceeded at step {i}"
            else:
                assert abs(ctrl[j] - q0_ctrl[j]) < 1e-5, f"actuator {j} (gripper) should hold episode-start setpoint"
        obs = env.step(ctrl)
    l1 = lift_angle(obs)

    assert l1 < l0 - 0.02, "shoulder_lift should move measurably in the corrected (up/away-from-table) direction"

    for name, ref in q0_arm.items():
        jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr = int(env.model.jnt_qposadr[jid])
        assert abs(float(obs["qpos"][qadr]) - ref) < 0.80, f"{name} drifted far from episode-start q (IK coordinates arm)"
