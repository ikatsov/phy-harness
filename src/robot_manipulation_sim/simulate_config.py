"""YAML defaults for ``scripts/simulate_policy.py``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AnalyzerSettings:
    """Single analyzer config entry in simulate YAML."""

    type: str
    enabled: bool = True
    params: dict[str, Any] | None = None


@dataclass
class SimulateSettings:
    """Rollout / logging options (from YAML + optional CLI overrides)."""

    policy_file: Path
    task: str | None = None
    policy_symbol: str = "policy"
    steps: int = 500
    include_rgb_observation: bool = False
    run_dir: Path | None = None
    video: Path | None = None
    joint_log_every_steps: int = 1
    video_tile_height: int = 360
    video_tile_width: int = 426
    video_output_fps: float | None = None
    disable_kinematic_overlays: bool = False
    analyzers: list[AnalyzerSettings] | None = None


def _parse_analyzers(raw: Any) -> list[AnalyzerSettings]:
    if raw is None:
        return [AnalyzerSettings(type="vlm_video_transcriber", enabled=False, params={})]
    if not isinstance(raw, list):
        raise ValueError("analyzers must be a list when set")
    out: list[AnalyzerSettings] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"analyzers[{i}] must be a mapping")
        t = item.get("type")
        if not isinstance(t, str) or not t.strip():
            raise ValueError(f"analyzers[{i}].type must be a non-empty string")
        enabled = bool(item.get("enabled", True))
        params_raw = item.get("params", {})
        if params_raw is None:
            params: dict[str, Any] = {}
        elif isinstance(params_raw, dict):
            params = dict(params_raw)
        else:
            raise ValueError(f"analyzers[{i}].params must be a mapping when set")
        out.append(AnalyzerSettings(type=t.strip(), enabled=enabled, params=params))
    return out


def _as_path(base: Path, p: Any, *, what: str) -> Path | None:
    if p is None or p == "":
        return None
    if not isinstance(p, str):
        raise ValueError(f"{what} must be a string or null when set")
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def load_simulate_settings(path: Path, *, yaml_dir: Path | None = None) -> SimulateSettings:
    """Load ``SimulateSettings`` from a YAML file (``version`` optional)."""
    cfg_path = path.resolve()
    if not cfg_path.is_file():
        raise ValueError(f"simulate config not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"simulate config must be a YAML mapping: {cfg_path}")

    parent = yaml_dir if yaml_dir is not None else cfg_path.parent
    base_dir_raw = raw.get("base_dir", ".")
    if not isinstance(base_dir_raw, str):
        raise ValueError("base_dir must be a string when set")
    base = (parent / base_dir_raw).resolve()

    task = raw.get("task")
    if task is not None:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string when set")
        task = task.strip()

    pol = raw.get("policy_file")
    if pol is None and task is not None:
        pol = f"policies/impl/{task}/{task}.py"
    if not pol or not isinstance(pol, str):
        raise ValueError("policy_file is required (or set task)")
    policy_path = _as_path(base, pol, what="policy_file")
    if policy_path is None:
        raise ValueError("policy_file resolved to empty path")

    run_dir_raw = raw.get("run_dir")
    if run_dir_raw is None and task is not None and raw.get("video") in (None, ""):
        run_dir_raw = f"artifacts/{task}"
    run_dir = _as_path(base, run_dir_raw, what="run_dir")
    video = _as_path(base, raw.get("video"), what="video")
    if run_dir is not None and video is not None:
        raise ValueError("set at most one of run_dir and video in simulate config")

    policy_symbol = raw.get("policy_symbol", "policy")
    if not isinstance(policy_symbol, str) or not policy_symbol.strip():
        raise ValueError("policy_symbol must be a non-empty string when set")

    steps = int(raw.get("steps", 500))
    if steps < 1:
        raise ValueError("steps must be >= 1")

    include_rgb_observation = bool(raw.get("include_rgb_observation", False))

    jli = int(raw.get("joint_log_every_steps", 1))

    vch = int(raw.get("video_tile_height", 360))
    vcw = int(raw.get("video_tile_width", 426))
    if vch < 1 or vcw < 1:
        raise ValueError("video_tile_height and video_tile_width must be >= 1")

    vf_raw = raw.get("video_output_fps")
    video_fps: float | None
    if vf_raw is None or vf_raw == "":
        video_fps = None
    else:
        video_fps = float(vf_raw)

    no_tr = bool(raw.get("disable_kinematic_overlays", False))
    analyzers = _parse_analyzers(raw.get("analyzers"))

    return SimulateSettings(
        policy_file=policy_path,
        task=task,
        policy_symbol=policy_symbol.strip(),
        steps=steps,
        include_rgb_observation=include_rgb_observation,
        run_dir=run_dir,
        video=video,
        joint_log_every_steps=jli,
        video_tile_height=vch,
        video_tile_width=vcw,
        video_output_fps=video_fps,
        disable_kinematic_overlays=no_tr,
        analyzers=analyzers,
    )


def builtin_defaults(*, policy_file: Path) -> SimulateSettings:
    """Hard-coded defaults when no YAML file is used (CLI-only)."""
    return SimulateSettings(
        policy_file=policy_file.resolve(),
        task=None,
        run_dir=None,
        video=None,
        analyzers=[AnalyzerSettings(type="vlm_video_transcriber", enabled=False, params={})],
    )
