#!/usr/bin/env python3
"""Run rollout validation from a YAML config (multiple analyzers over simulation artifacts)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/validate_rollout.py` without editable install (repo root on path).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from robot_manipulation_sim.validation import load_dotenv_repo, run_validation_from_yaml


def main() -> None:
    load_dotenv_repo()
    p = argparse.ArgumentParser(
        description=(
            "Validate simulation outputs using analyzers declared in a YAML file "
            "(see examples/validation.example.yaml)."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to validation YAML (simulation paths, task / analyzers; see examples/validation.example.yaml).",
    )
    args = p.parse_args()
    cfg = args.config.resolve()
    if not cfg.is_file():
        raise SystemExit(f"config not found: {cfg}")
    summary = run_validation_from_yaml(cfg)
    if summary.exit_code != 0:
        for r in summary.results:
            if r.exit_code != 0:
                print(f"[{r.analyzer_type}] exit={r.exit_code} ok={r.ok}", file=sys.stderr)
                for m in r.messages:
                    print(f"  {m}", file=sys.stderr)
    raise SystemExit(summary.exit_code)


if __name__ == "__main__":
    main()
