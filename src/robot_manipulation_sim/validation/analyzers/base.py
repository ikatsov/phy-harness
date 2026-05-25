"""Analyzer protocol and result type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from robot_manipulation_sim.validation.context import ValidationContext


@dataclass
class AnalyzerResult:
    """Outcome of a single analyzer run."""

    analyzer_type: str
    ok: bool
    exit_code: int = 0
    """0 = success; non-zero fails the overall validation run."""
    messages: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    """Optional paths or structured outputs (e.g. verdict dict)."""


class RolloutAnalyzer(Protocol):
    """Pluggable analyzer over simulation artifacts."""

    def analyze(self, ctx: ValidationContext) -> AnalyzerResult: ...
