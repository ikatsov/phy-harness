"""Tests for ``robot_manipulation_sim.simulate_config``."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_manipulation_sim.simulate_config import builtin_defaults, load_simulate_settings


def test_load_simulate_settings_minimal(tmp_path: Path) -> None:
    (tmp_path / "p.py").write_text("x=1\n", encoding="utf-8")
    cfg = {
        "version": 1,
        "base_dir": ".",
        "policy_file": "p.py",
        "run_dir": "out/run",
        "video": None,
    }
    yml = tmp_path / "sim.yaml"
    yml.write_text(yaml.dump(cfg), encoding="utf-8")
    s = load_simulate_settings(yml)
    assert s.policy_file == (tmp_path / "p.py").resolve()
    assert s.run_dir == (tmp_path / "out" / "run").resolve()
    assert s.video is None


def test_load_simulate_settings_rejects_run_dir_and_video(tmp_path: Path) -> None:
    (tmp_path / "p.py").write_text("x=1\n", encoding="utf-8")
    cfg = {
        "base_dir": ".",
        "policy_file": "p.py",
        "run_dir": "a",
        "video": "b.mp4",
    }
    yml = tmp_path / "sim.yaml"
    yml.write_text(yaml.dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="at most one"):
        load_simulate_settings(yml)


def test_builtin_defaults(tmp_path: Path) -> None:
    p = tmp_path / "z.py"
    p.write_text("x=1\n", encoding="utf-8")
    s = builtin_defaults(policy_file=p)
    assert s.policy_file == p.resolve()
    assert s.steps == 500
