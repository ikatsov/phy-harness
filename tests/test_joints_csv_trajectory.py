"""Tests for generic ``joints_csv_trajectory`` analyzer."""

from __future__ import annotations

import json
from pathlib import Path

from robot_manipulation_sim.validation.analyzers.generic.joints_csv_trajectory import JointsCsvTrajectoryAnalyzer
from robot_manipulation_sim.validation.context import SimulationArtifacts, ValidationContext


def _ctx(tmp_path: Path, csv_text: str, video_stem: str = "rollout") -> ValidationContext:
    jc = tmp_path / "joints.csv"
    jc.write_text(csv_text, encoding="utf-8")
    mp4 = tmp_path / f"{video_stem}.mp4"
    mp4.write_bytes(b"")
    sim = SimulationArtifacts(
        video=mp4,
        metrics_file=None,
        joints_csv=jc,
    )
    return ValidationContext(simulation=sim, task_spec="test task for task stream")


def test_trajectory_metrics_and_pass_by_default(tmp_path: Path) -> None:
    csv_text = (
        "episode,sim_step,time_sec,shoulder_pan_joint,ctrl_0\n"
        "1,0,0.0,0.0,0.0\n"
        "1,1,0.1,0.1,0.0\n"
        "1,2,0.2,0.2,0.0\n"
        "1,3,0.3,0.25,0.0\n"
        "1,4,0.4,0.3,0.0\n"
    )
    ctx = _ctx(tmp_path, csv_text)
    r = JointsCsvTrajectoryAnalyzer({"no_json_file": True}).analyze(ctx)
    assert r.exit_code == 0
    v = r.artifacts["verdict"]
    assert v["analyzer"] == "joints_csv_trajectory"
    assert v["pass"] is True
    assert v["checks"]["path_length_l2"] > 0
    assert v["checks"]["rms_vel_l2"] >= 0


def test_trajectory_fails_rms_jerk_limit(tmp_path: Path) -> None:
    """Large steps → high finite-difference jerk; should fail when max_rms_jerk is tiny."""
    rows = ["episode,sim_step,time_sec,j0,ctrl_0"]
    t = 0.0
    for i in range(20):
        rows.append(f"1,{i},{t:.3f},{0.0 if i % 2 == 0 else 2.0},0.0")
        t += 0.05
    ctx = _ctx(tmp_path, "\n".join(rows) + "\n")
    r = JointsCsvTrajectoryAnalyzer({"no_json_file": True, "max_rms_jerk": 1e-12}).analyze(ctx)
    assert r.exit_code == 1
    assert r.artifacts["verdict"]["pass"] is False


def test_trajectory_writes_json(tmp_path: Path) -> None:
    csv_text = (
        "episode,sim_step,time_sec,j0\n"
        "1,0,0.0,0.0\n1,1,0.1,0.05\n1,2,0.2,0.1\n1,3,0.3,0.12\n"
    )
    ctx = _ctx(tmp_path, csv_text, video_stem="clip")
    out = tmp_path / "custom.json"
    JointsCsvTrajectoryAnalyzer({"json_out": str(out), "no_json_file": False}).analyze(ctx)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["analyzer"] == "joints_csv_trajectory"


def test_trajectory_writes_default_json_next_to_video(tmp_path: Path) -> None:
    csv_text = (
        "episode,sim_step,time_sec,j0\n"
        "1,0,0.0,0.0\n1,1,0.1,0.05\n1,2,0.2,0.1\n1,3,0.3,0.12\n"
    )
    ctx = _ctx(tmp_path, csv_text, video_stem="rollout")
    JointsCsvTrajectoryAnalyzer({}).analyze(ctx)
    out = tmp_path / "rollout.trajectory.json"
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["analyzer"] == "joints_csv_trajectory"


def test_trajectory_string_false_still_writes_json(tmp_path: Path) -> None:
    """YAML can stringify booleans; ``bool(\"false\")`` must not suppress output."""
    csv_text = (
        "episode,sim_step,time_sec,j0\n"
        "1,0,0.0,0.0\n1,1,0.1,0.05\n1,2,0.2,0.1\n1,3,0.3,0.12\n"
    )
    ctx = _ctx(tmp_path, csv_text, video_stem="r2")
    JointsCsvTrajectoryAnalyzer({"no_json_file": "false"}).analyze(ctx)
    assert (tmp_path / "r2.trajectory.json").is_file()


def test_trajectory_writes_error_json_when_csv_missing(tmp_path: Path) -> None:
    mp4 = tmp_path / "rollout.mp4"
    mp4.write_bytes(b"")
    sim = SimulationArtifacts(
        video=mp4,
        metrics_file=None,
        joints_csv=tmp_path / "nope.csv",
    )
    ctx = ValidationContext(simulation=sim, task_spec=None)
    JointsCsvTrajectoryAnalyzer({}).analyze(ctx)
    out = tmp_path / "rollout.trajectory.json"
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data.get("error") == "joints_csv missing"
