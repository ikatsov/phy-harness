"""Validation task_spec and VLM prompt / verdict helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from robot_manipulation_sim.validation.analyzers.generic.vlm_observer import build_prompt, finalize_verdict


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_validation_example_yaml_has_task_spec_inline():
    """Intent for agents lives in ``examples/task/<task>.yaml`` when using ``task:`` in the main config."""
    path = _repo_root() / "examples" / "task" / "base_rotation.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    ts = raw.get("task_spec") or {}
    inline = ts.get("inline")
    assert isinstance(inline, str) and len(inline.strip()) > 40
    assert "shoulder pan" in inline.lower() or "base" in inline.lower()


def test_build_prompt_dual_stream_and_strict_timeline():
    p = build_prompt(
        metrics_text=None,
        media_hint="(media placeholder)",
        eval_duration_seconds=2.0,
        task_spec=None,
    )
    assert "summary_task_agnostic" in p
    assert "summary_task_evaluation" in p
    assert "second_by_second_neutral" in p
    assert "second_by_second_task" in p
    assert "exactly 2 strings" in p
    assert "task-agnostic" in p.lower() or "Neutral stream" in p


def test_build_prompt_includes_user_task_when_provided():
    p = build_prompt(
        metrics_text=None,
        media_hint="(media)",
        eval_duration_seconds=1.0,
        task_spec="Lift the red box without tipping.",
    )
    assert "Lift the red box without tipping." in p


def test_finalize_verdict_migrates_legacy_keys():
    v = {
        "pass": True,
        "summary": "Legacy summary.",
        "second_by_second": ["s0"],
        "panels_ok": True,
        "motion_controlled": True,
    }
    out = finalize_verdict(v, expected_timeline_entries=1)
    assert out["summary_task_agnostic"] == "Legacy summary."
    assert len(out["second_by_second_neutral"]) == 1
    assert len(out["second_by_second_task"]) == 1


def test_finalize_verdict_timeline_length_enforced():
    v = {
        "pass": True,
        "summary_task_agnostic": "x",
        "summary_task_evaluation": "y",
        "second_by_second_neutral": ["a"],
        "second_by_second_task": ["t"],
        "panels_ok": True,
        "motion_controlled": True,
    }
    out = finalize_verdict(v, expected_timeline_entries=2)
    assert len(out["second_by_second_neutral"]) == 2
    assert len(out["second_by_second_task"]) == 2
    assert out["pass"] is False
