"""Lightweight checks for scripts/robot_manipulation_loop.py (no Gemini)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def loop_mod():
    path = Path(__file__).resolve().parents[1] / "scripts" / "robot_manipulation_loop.py"
    if not path.is_file():
        pytest.skip(f"missing {path} (restore from VCS if you need loop tests)")
    spec = importlib.util.spec_from_file_location("robot_manipulation_loop", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_validate_repaired_source_accepts_policy(loop_mod):
    src = (
        "from __future__ import annotations\n"
        "from typing import Any\n"
        "import numpy as np\n"
        "from robot_manipulation_sim.env import UR5GripperEnv\n\n"
        "def policy(obs: dict[str, Any], step: int, env: UR5GripperEnv) -> np.ndarray:\n"
        "    return np.array(env._home, dtype=np.float64)\n"
    )
    loop_mod._validate_repaired_source(src)


def test_validate_repaired_source_rejects_no_policy(loop_mod):
    with pytest.raises(ValueError, match="policy"):
        loop_mod._validate_repaired_source("x = 1\n")
