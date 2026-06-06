"""Tests for ``robot_manipulation_sim.simulate_config``."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_manipulation_sim.simulate_config import builtin_defaults, load_simulate_settings


def _write_cfg(tmp_path: Path, cfg: dict) -> Path:
    (tmp_path / "p.py").write_text("x=1\n", encoding="utf-8")
    yml = tmp_path / "sim.yaml"
    yml.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return yml


def test_load_simulate_settings_minimal(tmp_path: Path) -> None:
    cfg = {
        "version": 1,
        "base_dir": ".",
        "policy_file": "p.py",
        "run_dir": "out/run",
        "video": None,
    }
    yml = _write_cfg(tmp_path, cfg)
    s = load_simulate_settings(yml)
    assert s.policy_file == (tmp_path / "p.py").resolve()
    assert s.run_dir == (tmp_path / "out" / "run").resolve()
    assert s.video is None


def test_load_simulate_settings_rejects_run_dir_and_video(tmp_path: Path) -> None:
    cfg = {
        "base_dir": ".",
        "policy_file": "p.py",
        "run_dir": "a",
        "video": "b.mp4",
    }
    yml = _write_cfg(tmp_path, cfg)
    with pytest.raises(ValueError, match="at most one"):
        load_simulate_settings(yml)


def test_builtin_defaults(tmp_path: Path) -> None:
    p = tmp_path / "z.py"
    p.write_text("x=1\n", encoding="utf-8")
    s = builtin_defaults(policy_file=p)
    assert s.policy_file == p.resolve()
    assert s.steps == 500
    assert s.analyzers is not None
    assert len(s.analyzers) == 1
    assert s.analyzers[0].type == "vlm_video_transcriber"
    assert s.analyzers[0].enabled is False


def test_load_simulate_settings_task_derives_policy_and_run_dir(tmp_path: Path) -> None:
    task = "path_tracing"
    p = tmp_path / "policies" / "impl" / task
    p.mkdir(parents=True)
    (p / f"{task}.py").write_text("x=1\n", encoding="utf-8")
    cfg = {
        "version": 1,
        "base_dir": ".",
        "task": task,
    }
    yml = tmp_path / "sim.yaml"
    yml.write_text(yaml.dump(cfg), encoding="utf-8")
    s = load_simulate_settings(yml)
    assert s.policy_file == (tmp_path / "policies" / "impl" / task / f"{task}.py").resolve()
    assert s.run_dir == (tmp_path / "artifacts" / task).resolve()
    assert s.video is None
    assert s.task == task


def test_load_simulate_settings_allows_zero_analyzers(tmp_path: Path) -> None:
    cfg = {
        "base_dir": ".",
        "policy_file": "p.py",
        "analyzers": [],
    }
    yml = _write_cfg(tmp_path, cfg)
    s = load_simulate_settings(yml)
    assert s.analyzers == []


def test_load_simulate_settings_parses_analyzers(tmp_path: Path) -> None:
    cfg = {
        "base_dir": ".",
        "policy_file": "p.py",
        "analyzers": [
            {"type": "vlm_video_transcriber", "enabled": True, "params": {"every_n_frames": 10, "max_frames": 30}},
        ],
    }
    yml = _write_cfg(tmp_path, cfg)
    s = load_simulate_settings(yml)
    assert s.analyzers is not None
    assert len(s.analyzers) == 1
    assert s.analyzers[0].type == "vlm_video_transcriber"
    assert s.analyzers[0].enabled is True
    assert s.analyzers[0].params == {"every_n_frames": 10, "max_frames": 30}


def test_load_simulate_settings_named_options(tmp_path: Path) -> None:
    cfg = {
        "base_dir": ".",
        "policy_file": "p.py",
        "policy_symbol": "policy",
        "include_rgb_observation": True,
        "joint_log_every_steps": 3,
        "video_tile_height": 300,
        "video_tile_width": 500,
        "video_output_fps": 15,
        "disable_kinematic_overlays": True,
    }
    yml = _write_cfg(tmp_path, cfg)
    s = load_simulate_settings(yml)
    assert s.policy_symbol == "policy"
    assert s.include_rgb_observation is True
    assert s.joint_log_every_steps == 3
    assert s.video_tile_height == 300
    assert s.video_tile_width == 500
    assert s.video_output_fps == 15.0
    assert s.disable_kinematic_overlays is True


