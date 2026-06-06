# Task bundles (`impl/<task>/<task>.yaml` + `<task>.py` + analyzers)

Each **task name** is a single identifier (e.g. `base_rotation`) used consistently:

1. **`policies/simulate_policy.example.yaml`** (or your copy) — top-level **`task: <name>`** (must match directory and filenames below).
2. **`policies/impl/<name>/<name>.yaml`** — `task_spec` (+ optional policy metadata).
3. **`policies/impl/<name>/<name>.py`** — policy passed to **`simulate_policy.py`**.
4. **`policies/impl/<name>/<analyzer_type>.py`** — optional policy-specific analyzers; enable via `analyzers:` in simulate YAML. Module contract: **`build(params) -> analyzer`**.

The harness requires the paired policy file **`policies/impl/<name>/<name>.py`** to exist whenever **`task:`** is set (see `load_simulate_settings` in `src/robot_manipulation_sim/simulate_config.py`).

Paired unit tests live under **`tests/test_<name>.py`** (e.g. `tests/test_base_rotation.py`).
