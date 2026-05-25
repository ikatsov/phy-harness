"""YAML validation pipeline (no Gemini)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml

from robot_manipulation_sim.validation import run_validation_from_yaml
from robot_manipulation_sim.validation.config import load_validation_yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_run_validation_manifest_and_vlm_dry(tmp_path: Path) -> None:
    mp4 = tmp_path / "t.mp4"
    frames = [(np.zeros((64, 96, 3), dtype=np.uint8) + 40) for _ in range(6)]
    imageio.mimsave(str(mp4), frames, fps=5, codec="libx264", macro_block_size=None)

    cfg = {
        "version": 1,
        "base_dir": str(tmp_path),
        "simulation": {"video": str(mp4.name)},
        "analyzers": [
            {"type": "artifact_manifest", "enabled": True, "params": {}},
            {
                "type": "vlm_observer",
                "enabled": True,
                "params": {"dry_run": True, "mode": "video", "no_json_file": True},
            },
        ],
    }
    yml = tmp_path / "val.yaml"
    yml.write_text(yaml.dump(cfg), encoding="utf-8")

    summary = run_validation_from_yaml(yml)
    assert summary.exit_code == 0
    assert len(summary.results) == 2
    assert summary.results[0].analyzer_type == "artifact_manifest"
    assert (tmp_path / "validation_manifest.json").is_file()


def test_validate_rollout_script_smoke(tmp_path: Path) -> None:
    mp4 = tmp_path / "t.mp4"
    frames = [(np.zeros((64, 96, 3), dtype=np.uint8) + 40) for _ in range(6)]
    imageio.mimsave(str(mp4), frames, fps=5, codec="libx264", macro_block_size=None)
    cfg = {
        "version": 1,
        "base_dir": str(tmp_path),
        "simulation": {"video": str(mp4.name)},
        "analyzers": [
            {"type": "vlm_observer", "enabled": True, "params": {"dry_run": True, "mode": "video"}},
        ],
    }
    yml = tmp_path / "val.yaml"
    yml.write_text(yaml.dump(cfg), encoding="utf-8")
    script = _repo_root() / "scripts" / "validate_rollout.py"
    r = subprocess.run(
        [sys.executable, str(script), "--config", str(yml)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr + r.stdout


def test_load_validation_yaml_task_mode_merges_analyzers(tmp_path: Path) -> None:
    art = tmp_path / "artifacts" / "demo_task"
    art.mkdir(parents=True)
    mp4 = art / "rollout.mp4"
    frames = [(np.zeros((32, 48, 3), dtype=np.uint8) + 30) for _ in range(4)]
    imageio.mimsave(str(mp4), frames, fps=4, codec="libx264", macro_block_size=None)

    td = tmp_path / "task"
    td.mkdir()
    (td / "demo_task.py").write_text(
        "import numpy as np\n\ndef policy(obs, step, env):\n    return np.array(env.data.ctrl, copy=True)\n",
        encoding="utf-8",
    )
    (td / "demo_task.yaml").write_text(
        yaml.dump(
            {
                "task_spec": {"inline": "Do the demo thing."},
                "task_analyzers": [
                    {"type": "artifact_manifest", "enabled": True, "params": {"output_name": "m2.json"}},
                ],
            }
        ),
        encoding="utf-8",
    )

    cfg_main = {
        "version": 1,
        "base_dir": str(tmp_path),
        "task": "demo_task",
        "analyzers_head": [
            {"type": "artifact_manifest", "enabled": True, "params": {"output_name": "m1.json"}},
        ],
        "analyzers_tail": [
            {"type": "artifact_manifest", "enabled": True, "params": {"output_name": "m3.json"}},
        ],
    }
    yml = tmp_path / "val.yaml"
    yml.write_text(yaml.dump(cfg_main), encoding="utf-8")

    job = load_validation_yaml(yml)
    assert job.task == "demo_task"
    assert [a.type for a in job.analyzers] == ["artifact_manifest"] * 3
    assert job.analyzers[0].params["output_name"] == "m1.json"
    assert job.analyzers[1].params["output_name"] == "m2.json"
    assert job.analyzers[2].params["output_name"] == "m3.json"
    assert job.task_spec_inline == "Do the demo thing."
