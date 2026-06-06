"""Runtime context passed to each rollout analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SimulationArtifacts:
    """Paths produced by ``simulate_policy.py`` (or equivalent)."""

    video: Path
    metrics_file: Path | None = None
    joints_csv: Path | None = None


@dataclass
class ValidationContext:
    """Resolved inputs for analyzers."""

    simulation: SimulationArtifacts
    metrics_text: str | None = None
    """Plain-text ``metrics.txt`` body, if present."""
    task_spec: str | None = None
    """Optional task text forwarded to analyzers."""
    config_path: Path | None = None
    """Path to the YAML file that produced this context, if any."""
