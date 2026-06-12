"""Validation tests for the cable_routing policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_manipulation_sim.env import UR5GripperEnv

pytest.importorskip("mink")


def _load_policy_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "policies"
        / "impl"
        / "cable_routing"
        / "cable_routing.py"
    )
    spec = importlib.util.spec_from_file_location("cable_routing", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_policy_module()


@pytest.fixture(scope="module")
def env_obs():
    repo_root = Path(__file__).resolve().parents[1]
    spec = yaml.safe_load(
        (repo_root / "policies/impl/cable_routing/cable_routing.yaml").read_text(encoding="utf-8")
    )
    scene_files = tuple((repo_root / rel).resolve() for rel in spec["scene_files"])
    env = UR5GripperEnv(enable_rgb=False, seed=0, scene_files=scene_files)
    obs = env.reset()
    return env, obs


def test_build_plan_returns_expected_shapes(mod, env_obs):
    env, obs = env_obs
    cable_tip = np.array([0.62, -0.20, 0.0], dtype=np.float64)
    pegs = {
        "peg_cyan": np.array([0.40, -0.24, 0.0], dtype=np.float64),
        "peg_magenta": np.array([0.52, -0.08, 0.0], dtype=np.float64),
        "peg_yellow": np.array([0.58, 0.10, 0.0], dtype=np.float64),
    }
    q_plan, g_plan = mod.build_plan(env, obs, cable_tip, pegs)

    dt = float(env.control_dt)
    n_crane = max(1, round(mod.T_CRANE / dt))
    n_app = max(1, round(mod.T_APPROACH / dt))
    n_desc = max(1, round(mod.T_DESCEND / dt))
    n_align = max(1, round(mod.T_WRIST_ALIGN / dt))
    n_recenter = max(1, round(mod.T_WRIST_RECENTER / dt))
    n_grip = max(1, round(mod.T_GRIP / dt))
    n_lift = max(1, round(mod.T_LIFT / dt))
    n_route = max(1, round(mod.T_ROUTE_SEGMENT / dt))
    n_route_segments = 9  # (p_lift + 3*(above,thread,above)) => 10 points => 9 segments
    n_expected = n_crane + n_app + n_align + n_desc + n_recenter + n_grip + n_lift + n_route_segments * n_route

    assert q_plan.shape == (n_expected, env.model.nq)
    assert g_plan.shape == (n_expected,)
    assert g_plan[0] == pytest.approx(mod.GRIPPER_OPEN, abs=1e-8)
    assert g_plan[-1] == pytest.approx(mod.GRIPPER_CLOSED, abs=1e-8)


def test_policy_runs_headless_without_crashing(mod, env_obs):
    env, obs = env_obs
    mod.reset()
    o = obs
    for step in range(20):
        ctrl = mod.policy(o, step, env)
        assert ctrl.shape == (env.nu,)
        o = env.step(ctrl)
