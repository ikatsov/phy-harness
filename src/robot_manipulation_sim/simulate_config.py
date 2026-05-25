"""YAML defaults for ``scripts/simulate_policy.py``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SimulateSettings:
    """Rollout / logging options (from YAML + optional CLI overrides)."""

    policy_file: Path
    symbol: str = "policy"
    steps: int = 500
    episodes: int = 1
    rgb: bool = False
    lift_z: float = 0.12
    strict: bool = False
    run_dir: Path | None = None
    video: Path | None = None
    joint_log_interval: int = 1
    video_cell_h: int = 360
    video_cell_w: int = 426
    video_separator_frames: int = 12
    video_fps: float | None = None
    no_overview_traces: bool = False


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

    pol = raw.get("policy_file")
    if not pol or not isinstance(pol, str):
        raise ValueError("policy_file is required (string path relative to base_dir or absolute)")
    policy_path = _as_path(base, pol, what="policy_file")
    if policy_path is None:
        raise ValueError("policy_file resolved to empty path")

    run_dir = _as_path(base, raw.get("run_dir"), what="run_dir")
    video = _as_path(base, raw.get("video"), what="video")
    if run_dir is not None and video is not None:
        raise ValueError("set at most one of run_dir and video in simulate config")

    symbol = raw.get("symbol", "policy")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string when set")

    steps = int(raw.get("steps", 500))
    episodes = int(raw.get("episodes", 1))
    if steps < 1 or episodes < 1:
        raise ValueError("steps and episodes must be >= 1")

    rgb = bool(raw.get("rgb", False))
    strict = bool(raw.get("strict", False))

    lift_z = float(raw.get("lift_z", 0.12))

    jli = int(raw.get("joint_log_interval", 1))

    vch = int(raw.get("video_cell_h", 360))
    vcw = int(raw.get("video_cell_w", 426))
    vsf = int(raw.get("video_separator_frames", 12))
    if vch < 1 or vcw < 1:
        raise ValueError("video_cell_h and video_cell_w must be >= 1")
    if vsf < 0:
        raise ValueError("video_separator_frames must be >= 0")

    vf_raw = raw.get("video_fps")
    video_fps: float | None
    if vf_raw is None or vf_raw == "":
        video_fps = None
    else:
        video_fps = float(vf_raw)

    no_tr = bool(raw.get("no_overview_traces", False))

    return SimulateSettings(
        policy_file=policy_path,
        symbol=symbol.strip(),
        steps=steps,
        episodes=episodes,
        rgb=rgb,
        lift_z=lift_z,
        strict=strict,
        run_dir=run_dir,
        video=video,
        joint_log_interval=jli,
        video_cell_h=vch,
        video_cell_w=vcw,
        video_separator_frames=vsf,
        video_fps=video_fps,
        no_overview_traces=no_tr,
    )


def builtin_defaults(*, policy_file: Path) -> SimulateSettings:
    """Hard-coded defaults when no YAML file is used (CLI-only)."""
    return SimulateSettings(
        policy_file=policy_file.resolve(),
        run_dir=None,
        video=None,
    )
