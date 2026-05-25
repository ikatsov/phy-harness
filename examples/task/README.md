# Task bundles (`<task>.yaml` + `<task>.py`)

Each **task name** is a single identifier (e.g. `base_rotation`) used in three places with the **same stem**:

1. **`examples/validation.example.yaml`** (or your copy) — top-level **`task: <name>`** (must match the filenames below).
2. **`examples/task/<name>.yaml`** — `task_spec`, `task_analyzers`; rollout defaults live under **`artifacts/<name>/`** (relative to **`base_dir`** in the main validation YAML).
3. **`examples/task/<name>.py`** — policy passed to **`simulate_policy.py`** (use **`--run-dir artifacts/<name>`** so validation’s default **`artifacts/<task>`** matches).

The harness requires the paired policy file **`examples/task/<name>.py`** to exist whenever **`task:`** is set (see `load_validation_yaml` in `src/robot_manipulation_sim/validation/config.py`).

Paired unit tests live under **`tests/test_<name>.py`** (e.g. `tests/test_base_rotation.py`).
