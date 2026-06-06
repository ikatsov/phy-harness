"""Joint CSV header/row shape matches MuJoCo model (no GL)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from robot_manipulation_sim import UR5GripperEnv


def _load_simulate():
    path = Path(__file__).resolve().parents[1] / "scripts" / "simulate_policy.py"
    spec = importlib.util.spec_from_file_location("simulate_policy", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_joint_log_row_matches_header():
    sim = _load_simulate()
    env = UR5GripperEnv(enable_rgb=False, seed=0)
    env.reset()
    h = sim.joint_log_header(env.model)
    assert any(c.startswith("target_") for c in h)
    assert h.count("episode") == 1
    tgt = np.array(env.data.ctrl, dtype=np.float64).copy()
    r = sim.joint_log_row(env.model, env.data, episode=1, sim_step=0, target_ctrl=tgt)
    assert len(h) == len(r)
    assert h[0] == "episode" and r[0] == 1
    assert np.isfinite(r[2])  # time_sec
    # Rounded floats (policy target should match applied ctrl at rest)
    assert abs(float(r[h.index("target_a_shoulder_pan")]) - float(r[h.index("ctrl_a_shoulder_pan")])) < 1e-9
