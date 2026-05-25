"""YAML validation config → resolved ``ValidationContext``."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from robot_manipulation_sim.validation.context import SimulationArtifacts, ValidationContext
from robot_manipulation_sim.validation.task_spec import load_task_spec_from_policy
from robot_manipulation_sim.validation.util import coerce_bool


def _resolve_path(base: Path, p: str | Path | None) -> Path | None:
    if p is None or p == "":
        return None
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _safe_task_name(task: str) -> str:
    t = task.strip()
    if not t or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", t):
        raise ValueError(f"invalid task name {task!r} (use letters, digits, _, -, .; no path segments)")
    if ".." in t or "/" in t or "\\" in t:
        raise ValueError(f"invalid task name {task!r}")
    return t


def _load_yaml_mapping(path: Path, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{what} not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{what} must be a YAML mapping: {path}")
    return raw


def _parse_analyzer_block(raw: dict[str, Any], i: str) -> AnalyzerConfig:
    t = raw.get("type")
    if not t or not isinstance(t, str):
        raise ValueError(f"{i}.type is required")
    enabled = coerce_bool(raw.get("enabled"), default=True)
    params = raw.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError(f"{i}.params must be a mapping")
    return AnalyzerConfig(type=t.strip(), enabled=enabled, params=params)


def _parse_analyzer_list(blocks: Any, label: str) -> list[AnalyzerConfig]:
    if not isinstance(blocks, list):
        raise ValueError(f"{label} must be a list")
    out: list[AnalyzerConfig] = []
    for i, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise ValueError(f"{label}[{i}] must be a mapping")
        out.append(_parse_analyzer_block(block, f"{label}[{i}]"))
    return out


def _default_simulation_paths(base: Path, subdir: str) -> dict[str, Path]:
    root = (base / subdir).resolve()
    return {
        "video": root / "rollout.mp4",
        "metrics_file": root / "metrics.txt",
        "joints_csv": root / "joints.csv",
    }


@dataclass
class AnalyzerConfig:
    type: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationJobConfig:
    """In-memory config (from YAML or programmatic)."""

    simulation: SimulationArtifacts
    analyzers: list[AnalyzerConfig] = field(default_factory=list)
    task_spec_inline: str | None = None
    task_spec_policy: Path | None = None
    version: int = 1
    task: str | None = None


def load_validation_yaml(path: Path) -> ValidationJobConfig:
    raw = _load_yaml_mapping(path, "validation config")
    ver = int(raw.get("version", 1))
    cfg_dir = path.parent.resolve()
    base = (cfg_dir / str(raw.get("base_dir", "."))).resolve()

    task_name_raw = raw.get("task")
    task_key: str | None = None
    task_def: dict[str, Any] = {}
    paired_policy_py: Path | None = None
    if task_name_raw is not None:
        if not isinstance(task_name_raw, str):
            raise ValueError("'task' must be a string when set")
        task_key = _safe_task_name(task_name_raw)
        task_dir = cfg_dir / "task"
        td_path = (task_dir / f"{task_key}.yaml").resolve()
        task_def = _load_yaml_mapping(
            td_path,
            f"task YAML for {task_key!r} (expected task/{td_path.name} beside the validation config)",
        )
        paired_policy_py = (task_dir / f"{task_key}.py").resolve()
        if not paired_policy_py.is_file():
            raise ValueError(
                f"task {task_key!r}: paired policy module must exist at {paired_policy_py} "
                f"(same directory and stem as the task YAML {td_path.name!r}; "
                f"the ``task`` field must match those filenames exactly, without extension)."
            )

    # --- simulation ---
    sim = raw.get("simulation") or {}
    if not isinstance(sim, dict):
        raise ValueError("'simulation' must be a mapping when set")

    if task_key is not None:
        sub = f"artifacts/{task_key}"
        defaults = _default_simulation_paths(base, sub)
        video_path = sim.get("video", defaults["video"])
        metrics_path = sim.get("metrics_file", defaults["metrics_file"])
        joints_path = sim.get("joints_csv", defaults["joints_csv"])
    else:
        video_path = sim.get("video")
        metrics_path = sim.get("metrics_file")
        joints_path = sim.get("joints_csv")

    video = _resolve_path(base, video_path)
    if video is None or not video.is_file():
        raise ValueError(f"simulation.video must exist relative to base_dir: {video_path!r}")

    artifacts = SimulationArtifacts(
        video=video,
        metrics_file=_resolve_path(base, metrics_path),
        joints_csv=_resolve_path(base, joints_path),
    )

    # --- task_spec ---
    ts_main = raw.get("task_spec") or {}
    ts_task = task_def.get("task_spec") or {}
    if not isinstance(ts_main, dict):
        raise ValueError("task_spec must be a mapping when set in main config")
    if not isinstance(ts_task, dict):
        raise ValueError("task_spec must be a mapping in task definition YAML")

    inline = ts_main.get("inline")
    policy_mod = ts_main.get("policy_module")
    if task_key is not None:
        if inline is None:
            inline = ts_task.get("inline")
        if policy_mod is None:
            policy_mod = ts_task.get("policy_module")

    if inline is not None and not isinstance(inline, str):
        raise ValueError("task_spec.inline must be a string when set")
    policy_path = _resolve_path(base, policy_mod) if policy_mod else None
    if task_key is not None and policy_path is None and paired_policy_py is not None:
        policy_path = paired_policy_py
    analyzers: list[AnalyzerConfig] = []
    if task_key is not None:
        head = raw.get("analyzers_head")
        tail = raw.get("analyzers_tail")
        if head is None and raw.get("analyzers") is not None:
            raise ValueError(
                "when 'task' is set, use 'analyzers_head' and 'analyzers_tail' "
                "(not a single 'analyzers' list); task-specific entries come from the task definition file"
            )
        if head is None or tail is None:
            raise ValueError("when 'task' is set, both 'analyzers_head' and 'analyzers_tail' are required")
        analyzers.extend(_parse_analyzer_list(head, "analyzers_head"))
        task_blocks = task_def.get("task_analyzers")
        if task_blocks is None:
            task_blocks = []
        analyzers.extend(_parse_analyzer_list(task_blocks, "task_analyzers (task definition)"))
        analyzers.extend(_parse_analyzer_list(tail, "analyzers_tail"))
    else:
        analyzers_raw = raw.get("analyzers")
        if not isinstance(analyzers_raw, list) or not analyzers_raw:
            raise ValueError("'analyzers' must be a non-empty list when 'task' is not set")
        for i, block in enumerate(analyzers_raw):
            if not isinstance(block, dict):
                raise ValueError(f"analyzers[{i}] must be a mapping")
            analyzers.append(_parse_analyzer_block(block, f"analyzers[{i}]"))

    if not analyzers:
        raise ValueError("merged analyzers list is empty")

    return ValidationJobConfig(
        version=ver,
        simulation=artifacts,
        analyzers=analyzers,
        task_spec_inline=inline.strip() if isinstance(inline, str) and inline.strip() else None,
        task_spec_policy=policy_path,
        task=task_key,
    )


def build_context(job: ValidationJobConfig, *, config_path: Path | None = None) -> ValidationContext:
    metrics_text = None
    mf = job.simulation.metrics_file
    if mf is not None and mf.is_file():
        metrics_text = mf.read_text(encoding="utf-8")

    task_spec = job.task_spec_inline
    if task_spec is None and job.task_spec_policy is not None:
        task_spec = load_task_spec_from_policy(job.task_spec_policy)

    return ValidationContext(
        simulation=job.simulation,
        metrics_text=metrics_text,
        task_spec=task_spec,
        config_path=config_path,
    )
