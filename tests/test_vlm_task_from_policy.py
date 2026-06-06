"""Validation task_spec and VLM frame-transcriber helpers."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml

from robot_manipulation_sim.validation.analyzers.vlm_video_transcriber import (
    VlmVideoTranscriberAnalyzer,
    _sample_video_frames,
)
from robot_manipulation_sim.validation.context import SimulationArtifacts, ValidationContext


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_validation_example_yaml_has_task_spec_inline():
    """Intent for agents lives in ``policies/impl/<task>/<task>.yaml`` when using ``task:`` in the main config."""
    path = _repo_root() / "policies" / "impl" / "base_rotation" / "base_rotation.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    ts = raw.get("task_spec") or {}
    inline = ts.get("inline")
    assert isinstance(inline, str) and len(inline.strip()) > 40
    assert "shoulder pan" in inline.lower() or "base" in inline.lower()


def _write_tiny_video(path: Path, n_frames: int = 10) -> None:
    frames = []
    for i in range(n_frames):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 20) % 255
        frames.append(frame)
    imageio.mimsave(path, frames, fps=10)


def test_sample_video_frames_every_n(tmp_path: Path):
    video = tmp_path / "sample.mp4"
    _write_tiny_video(video, n_frames=10)
    idxs, frames, fps = _sample_video_frames(video, every_n_frames=3, max_frames=20)
    assert idxs == [0, 3, 6, 9]
    assert len(frames) == 4
    assert fps and fps > 0.0


def test_transcriber_dry_run_writes_json(tmp_path: Path):
    video = tmp_path / "sample.mp4"
    _write_tiny_video(video, n_frames=9)
    analyzer = VlmVideoTranscriberAnalyzer(
        {
            "dry_run": True,
            "every_n_frames": 4,
            "max_frames": 10,
        }
    )
    ctx = ValidationContext(simulation=SimulationArtifacts(video=video))
    result = analyzer.analyze(ctx)
    assert result.ok is True
    out_path = video.with_suffix(".vlm_transcript.json")
    assert out_path.is_file()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["analyzer"] == "vlm_video_transcriber"
    assert payload["every_n_frames"] == 4
    assert payload["sampled_frames"] == len(payload["entries"])
    assert payload["entries"][0]["description"] == "dry-run: VLM call skipped"
