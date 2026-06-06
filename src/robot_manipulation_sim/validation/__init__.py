"""Minimal validation surface: VLM transcriber + shared analyzer types."""

from __future__ import annotations

from robot_manipulation_sim.validation.context import SimulationArtifacts, ValidationContext
from robot_manipulation_sim.validation.analyzers.base import AnalyzerResult, RolloutAnalyzer
from robot_manipulation_sim.validation.analyzers.vlm_video_transcriber import VlmVideoTranscriberAnalyzer
from robot_manipulation_sim.validation.util import load_dotenv_repo, repo_root

__all__ = [
    "AnalyzerResult",
    "RolloutAnalyzer",
    "SimulationArtifacts",
    "ValidationContext",
    "VlmVideoTranscriberAnalyzer",
    "load_dotenv_repo",
    "repo_root",
]
