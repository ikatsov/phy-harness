"""Load ``TASK_SPEC`` / ``VLM_TASK`` from a policy module (when YAML uses ``task_spec.policy_module``)."""

from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path


def load_task_spec_from_policy(policy_path: Path) -> str:
    policy_path = policy_path.resolve()
    if not policy_path.is_file():
        raise ValueError(f"policy module not found: {policy_path}")
    spec = importlib.util.spec_from_file_location("validation_task_spec_policy", policy_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load policy: {policy_path}")
    mod = importlib.util.module_from_spec(spec)
    mod_name = f"validation_task_spec_{uuid.uuid4().hex}"
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    for attr in ("TASK_SPEC", "VLM_TASK"):
        task = getattr(mod, attr, None)
        if isinstance(task, str) and task.strip():
            return task.strip()
    raise ValueError(
        f"{policy_path} must define module-level string TASK_SPEC or VLM_TASK "
        "(intent description for agents; not sent to observer VLM)."
    )
