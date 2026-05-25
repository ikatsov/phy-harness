"""Dry-run tests for vlm_observer via validate_rollout.py (no API key)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_validate_rollout(tmp_path: Path, mp4: Path, *, mode: str) -> subprocess.CompletedProcess[str]:
    cfg = {
        "version": 1,
        "base_dir": str(tmp_path),
        "simulation": {"video": str(mp4.name)},
        "analyzers": [
            {
                "type": "vlm_observer",
                "enabled": True,
                "params": {"dry_run": True, "mode": mode},
            },
        ],
    }
    yml = tmp_path / "val.yaml"
    yml.write_text(yaml.dump(cfg), encoding="utf-8")
    script = _repo_root() / "scripts" / "validate_rollout.py"
    return subprocess.run(
        [sys.executable, str(script), "--config", str(yml)],
        capture_output=True,
        text=True,
    )


def test_validate_rollout_dry_run_video(tmp_path: Path) -> None:
    path = tmp_path / "t.mp4"
    frames = [(np.zeros((64, 96, 3), dtype=np.uint8) + 40) for _ in range(6)]
    imageio.mimsave(str(path), frames, fps=5, codec="libx264", macro_block_size=None)
    r = _run_validate_rollout(tmp_path, path, mode="video")
    assert r.returncode == 0, r.stderr + r.stdout
    assert "dry-run" in r.stdout.lower()
    assert ".vlm.json" in r.stdout


def test_validate_rollout_dry_run_frames(tmp_path: Path) -> None:
    path = tmp_path / "t.mp4"
    frames = [(np.zeros((64, 96, 3), dtype=np.uint8) + 40) for _ in range(6)]
    imageio.mimsave(str(path), frames, fps=5, codec="libx264", macro_block_size=None)
    r = _run_validate_rollout(tmp_path, path, mode="frames")
    assert r.returncode == 0, r.stderr + r.stdout
    assert "extracted" in r.stdout.lower() or "frame" in r.stdout.lower()
