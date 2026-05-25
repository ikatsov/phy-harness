"""Regression: base_rotation task policy should slew pan and keep other joints near episode-start q."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mujoco
import numpy as np

from robot_manipulation_sim import UR5GripperEnv


def _load_policy_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "task" / "base_rotation.py"
    spec = importlib.util.spec_from_file_location("base_rotation", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _actuator_qadr(env: UR5GripperEnv, act_idx: int) -> int:
    jid = int(env.model.actuator_trnid[act_idx, 0])
    return int(env.model.jnt_qposadr[jid])


def test_slow_base_rotation_mostly_pan():
    mod = _load_policy_module()
    policy = mod.policy
    reset_fn = getattr(mod, "reset", None)
    if callable(reset_fn):
        reset_fn()
    env = UR5GripperEnv(enable_rgb=False, seed=0)
    obs = env.reset(box_xy_noise=0.0)

    def pan_angle(o):
        jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pan_joint")
        qadr = int(env.model.jnt_qposadr[jid])
        return float(o["qpos"][qadr])

    q0_arm = {}
    for name in ("shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"):
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

    pan_step = mod.BASE_ROTATION_SPEED * env.control_dt
    p0 = pan_angle(obs)
    for i in range(200):
        ctrl = policy(obs, i, env)
        for j in range(env.nu):
            if j == 0:
                qadr = _actuator_qadr(env, j)
                q = float(obs["qpos"][qadr])
                assert abs(ctrl[j] - q) <= pan_step + 1e-5, f"actuator {j} pan slew exceeded at step {i}"
            else:
                assert abs(ctrl[j] - q0_ctrl[j]) < 1e-5, f"actuator {j} should hold episode-start setpoint"
        obs = env.step(ctrl)
    p1 = pan_angle(obs)

    assert p1 > p0 + 0.05, "shoulder_pan should advance over 200 steps"

    for name, ref in q0_arm.items():
        jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr = int(env.model.jnt_qposadr[jid])
        assert abs(float(obs["qpos"][qadr]) - ref) < 0.12, f"{name} drifted far from episode-start q"
