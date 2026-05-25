"""Run enabled analyzers in order."""

from __future__ import annotations

from dataclasses import dataclass, field

from robot_manipulation_sim.validation.analyzers import make_analyzer
from robot_manipulation_sim.validation.analyzers.base import AnalyzerResult
from robot_manipulation_sim.validation.config import ValidationJobConfig, build_context


@dataclass
class ValidationRunSummary:
    """Aggregate result."""

    results: list[AnalyzerResult] = field(default_factory=list)
    exit_code: int = 0


def run_validation(job: ValidationJobConfig, *, config_path: str | None = None) -> ValidationRunSummary:
    from pathlib import Path

    ctx = build_context(job, config_path=Path(config_path).resolve() if config_path else None)
    summary = ValidationRunSummary()
    for spec in job.analyzers:
        if not spec.enabled:
            continue
        analyzer = make_analyzer(spec.type, spec.params)
        result = analyzer.analyze(ctx)
        summary.results.append(result)
        summary.exit_code = max(summary.exit_code, result.exit_code)
    return summary
