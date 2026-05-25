"""Built-in rollout analyzers.

**Layout**

- ``generic/`` — default, task-agnostic analyzers (``artifact_manifest``, ``vlm_observer``, ``joints_csv_trajectory``).
  Do **not** add or modify these from the robot-manipulation-task-loop skill; only product/harness
  changes.
- ``task_specific/`` — rubric analyzers tied to a task description; the skill may **add** modules
  here and register their ``type`` in ``TASK_SPECIFIC_REGISTRY`` (merged into ``REGISTRY``).
"""

from __future__ import annotations

from typing import Any, Callable

from robot_manipulation_sim.validation.analyzers.base import RolloutAnalyzer
from robot_manipulation_sim.validation.analyzers.generic.artifact_manifest import ArtifactManifestAnalyzer
from robot_manipulation_sim.validation.analyzers.generic.joints_csv_trajectory import JointsCsvTrajectoryAnalyzer
from robot_manipulation_sim.validation.analyzers.generic.vlm_observer import VlmObserverAnalyzer
from robot_manipulation_sim.validation.analyzers.task_specific.joints_csv_base_rotation import (
    JointsCsvBaseRotationAnalyzer,
)

AnalyzerFactory = Callable[[dict[str, Any]], RolloutAnalyzer]

GENERIC_REGISTRY: dict[str, AnalyzerFactory] = {
    "artifact_manifest": lambda p: ArtifactManifestAnalyzer(p),
    "vlm_observer": lambda p: VlmObserverAnalyzer(p),
    "joints_csv_trajectory": lambda p: JointsCsvTrajectoryAnalyzer(p),
}

TASK_SPECIFIC_REGISTRY: dict[str, AnalyzerFactory] = {
    "joints_csv_base_rotation": lambda p: JointsCsvBaseRotationAnalyzer(p),
}

REGISTRY: dict[str, AnalyzerFactory] = {**GENERIC_REGISTRY, **TASK_SPECIFIC_REGISTRY}


def make_analyzer(analyzer_type: str, params: dict[str, Any] | None = None) -> RolloutAnalyzer:
    key = analyzer_type.strip()
    if key not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown analyzer type {analyzer_type!r}. Known: {known}")
    return REGISTRY[key](params or {})
