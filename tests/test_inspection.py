"""Validation tests for the ``inspection`` policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

from robot_manipulation_sim.env import UR5GripperEnv
from robot_manipulation_sim.ik.tool_pose import tool0_se3_matrix

pytest.importorskip("mink")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_policy_module():
    path = _repo_root() / "policies" / "impl" / "inspection" / "inspection.py"
    spec = importlib.util.spec_from_file_location("inspection", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_policy_module()


@pytest.fixture(scope="module")
def env_obs():
    root = _repo_root()
    env = UR5GripperEnv(
        enable_rgb=False,
        seed=0,
        scene_files=(
            root / "src" / "robot_manipulation_sim" / "mjcf" / "ur5e_two_finger_inspection_scene.xml",
            root / "src" / "robot_manipulation_sim" / "mjcf" / "scene_objects" / "inspection_part.xml",
        ),
    )
    obs = env.reset(box_xy_noise=0.0)
    return env, obs


def test_inspection_mesh_and_geom_exist(mod, env_obs):
    env, _ = env_obs
    gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, mod.INSPECTION_GEOM_NAME)
    assert gid >= 0, f"missing inspection geom {mod.INSPECTION_GEOM_NAME!r}"
    mid = int(env.model.geom_dataid[gid])
    assert mid >= 0, "inspection geom has no mesh data id"


def test_policy_runs_and_tracks_scan_targets(mod, env_obs):
    reset_fn = getattr(mod, "reset", None)
    if callable(reset_fn):
        reset_fn()

    env, obs = env_obs
    targets = mod.accessible_edge_points_world(env.model, env.data) + np.array(
        [0.0, 0.0, mod.SCAN_STANDOFF_M], dtype=np.float64
    )

    # Run enough steps to cover crane + approach + full edge scan.
    n_scan_pts = int(targets.shape[0])
    n_steps = max(
        1,
        round(mod.CRANE_DURATION_S / float(env.control_dt))
        + round(mod.APPROACH_DURATION_S / float(env.control_dt))
        + max(0, n_scan_pts - 1) * round(mod.SCAN_SEGMENT_DURATION_S / float(env.control_dt))
        + 12,
    )
    tool_positions: list[np.ndarray] = []
    for step in range(n_steps):
        ctrl = mod.policy(obs, step, env)
        assert ctrl.shape == (env.nu,)
        lo = env.model.actuator_ctrlrange[:, 0]
        hi = env.model.actuator_ctrlrange[:, 1]
        assert np.all(ctrl >= lo) and np.all(ctrl <= hi)
        obs = env.step(ctrl)
        d = mujoco.MjData(env.model)
        d.qpos[:] = obs["qpos"]
        mujoco.mj_forward(env.model, d)
        tool_positions.append(tool0_se3_matrix(env.model, d)[:3, 3].copy())

    tool_positions_arr = np.asarray(tool_positions, dtype=np.float64)
    # Ensure TCP actually moves (non-trivial scan/approach motion).
    travel = np.linalg.norm(np.diff(tool_positions_arr, axis=0), axis=1).sum()
    assert travel > 0.10, f"inspection trajectory is too short (travel={travel:.3f} m)"

    # Require the executed trajectory to stay in the neighborhood of all scan targets.
    for i, tgt in enumerate(targets):
        dist = np.min(np.linalg.norm(tool_positions_arr - tgt[np.newaxis], axis=1))
        assert dist < 0.26, f"target {i} not reached closely enough (min dist={dist:.3f} m)"
