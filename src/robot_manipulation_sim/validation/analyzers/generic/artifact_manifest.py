"""Write a small JSON manifest of resolved simulation paths (and optional task spec presence)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robot_manipulation_sim.validation.analyzers.base import AnalyzerResult
from robot_manipulation_sim.validation.context import ValidationContext


class ArtifactManifestAnalyzer:
    """Records which artifact paths were configured (for CI / debugging / future analyzers)."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(params or {})

    def analyze(self, ctx: ValidationContext) -> AnalyzerResult:
        sim = ctx.simulation
        out_name = str(self.params.get("output_name", "validation_manifest.json"))
        out_path = Path(self.params.get("output_path") or (sim.video.parent / out_name))
        out_path = out_path.resolve()

        manifest: dict[str, Any] = {
            "video": str(sim.video.resolve()),
            "metrics_file": str(sim.metrics_file.resolve()) if sim.metrics_file else None,
            "joints_csv": str(sim.joints_csv.resolve()) if sim.joints_csv else None,
            "task_spec_configured": ctx.task_spec is not None,
            "task_spec_chars": len(ctx.task_spec) if ctx.task_spec else 0,
        }
        for key in ("metrics_file", "joints_csv"):
            p = manifest.get(key)
            if p:
                manifest[f"{key}_exists"] = Path(p).is_file()
            else:
                manifest[f"{key}_exists"] = False

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"artifact_manifest: wrote {out_path}", flush=True)
        return AnalyzerResult(
            "artifact_manifest",
            ok=True,
            exit_code=0,
            messages=[str(out_path)],
            artifacts={"manifest_path": str(out_path), "manifest": manifest},
        )
