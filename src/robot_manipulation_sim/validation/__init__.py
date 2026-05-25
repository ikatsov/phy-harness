"""Rollout validation pipeline (YAML-driven, pluggable analyzers)."""

from __future__ import annotations

from pathlib import Path

from robot_manipulation_sim.validation.config import AnalyzerConfig, ValidationJobConfig, build_context, load_validation_yaml
from robot_manipulation_sim.validation.context import SimulationArtifacts, ValidationContext
from robot_manipulation_sim.validation.runner import ValidationRunSummary, run_validation
from robot_manipulation_sim.validation.util import load_dotenv_repo, repo_root

__all__ = [
    "AnalyzerConfig",
    "SimulationArtifacts",
    "ValidationContext",
    "ValidationJobConfig",
    "ValidationRunSummary",
    "build_context",
    "load_validation_yaml",
    "load_dotenv_repo",
    "repo_root",
    "run_validation",
    "run_validation_from_yaml",
]


def run_validation_from_yaml(path: Path | str) -> ValidationRunSummary:
    p = Path(path)
    job = load_validation_yaml(p)
    return run_validation(job, config_path=str(p))
