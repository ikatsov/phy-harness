"""Tests for task-specific ``joints_csv_base_rotation`` analyzer."""

from __future__ import annotations

import json
from pathlib import Path

from robot_manipulation_sim.validation.analyzers.task_specific.joints_csv_base_rotation import (
    JointsCsvBaseRotationAnalyzer,
)
from robot_manipulation_sim.validation.context import SimulationArtifacts, ValidationContext


def _hdr() -> str:
    return (
        "episode,sim_step,time_sec,shoulder_pan_joint,shoulder_lift_joint,elbow_joint,"
        "wrist_1_joint,wrist_2_joint,wrist_3_joint,gripper_slide,grasp_box_free.pz\n"
    )


def test_base_rotation_passes_pan_only(tmp_path: Path) -> None:
    lines = []
    for i in range(20):
        t = i * 0.05
        pan = 0.02 * i
        lines.append(f"1,{i},{t:.4f},{pan:.5f},0,0,0,0,0,0,0.04\n")
    p = tmp_path / "j.csv"
    p.write_text(_hdr() + "".join(lines), encoding="utf-8")
    mp4 = tmp_path / "rollout.mp4"
    mp4.write_bytes(b"x")
    ctx = ValidationContext(simulation=SimulationArtifacts(video=mp4, joints_csv=p), task_spec="base pan only")
    r = JointsCsvBaseRotationAnalyzer(
        {"no_json_file": True, "min_pan_total_rad": 0.3, "home_max_excursion_rad": 0.2, "home_max_span_per_s": 0.08}
    ).analyze(ctx)
    assert r.ok and r.exit_code == 0
    v = r.artifacts["verdict"]
    assert v["pass"] is True
    assert v["analyzer"] == "joints_csv_base_rotation"


def test_base_rotation_fails_elbow(tmp_path: Path) -> None:
    lines = []
    for i in range(10):
        t = i * 0.1
        e = 0.5 * i
        lines.append(f"1,{i},{t:.2f},0.5,0,{e:.4f},0,0,0,0,0.04\n")
    p = tmp_path / "j.csv"
    p.write_text(_hdr() + "".join(lines), encoding="utf-8")
    mp4 = tmp_path / "rollout.mp4"
    mp4.write_bytes(b"x")
    ctx = ValidationContext(simulation=SimulationArtifacts(video=mp4, joints_csv=p))
    r = JointsCsvBaseRotationAnalyzer(
        {"no_json_file": True, "min_pan_total_rad": 0.01, "home_max_excursion_rad": 0.5, "home_max_span_per_s": 0.04}
    ).analyze(ctx)
    assert not r.ok and r.exit_code == 1
    assert r.artifacts["verdict"]["pass"] is False


def test_writes_json(tmp_path: Path) -> None:
    lines = [f"1,{i},{i*0.1:.2f},{0.1*i:.4f},0,0,0,0,0,0,0.04\n" for i in range(5)]
    p = tmp_path / "j.csv"
    p.write_text(_hdr() + "".join(lines), encoding="utf-8")
    mp4 = tmp_path / "rollout.mp4"
    mp4.write_bytes(b"x")
    ctx = ValidationContext(simulation=SimulationArtifacts(video=mp4, joints_csv=p))
    JointsCsvBaseRotationAnalyzer({"min_pan_total_rad": 0.2, "home_max_excursion_rad": 0.2}).analyze(ctx)
    out = tmp_path / "rollout.joints_base.json"
    assert out.is_file()
    assert "summary_task_agnostic" in json.loads(out.read_text(encoding="utf-8"))
