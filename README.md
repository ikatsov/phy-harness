# Robotic manipulation dev harness

Python harness for **Universal Robots UR5e** simulation with a **Robotiq 2F-85** adaptive gripper (tendon drive, **0–255** ctrl), multi-view rollout video with overlays, joints logging, and analyzer support for task policy development.

The current development workflow is:

```
  ┌──────────────────────────────────────────────────────────────────┐
  │ INPUTS                                                           │
  │  Start with policies/impl/<task>/<task>.yaml (task spec)         │
  └───────────────────────────────┬──────────────────────────────────┘
                                  ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ DESIGN & IMPLEMENT                                               │
  │  Coding model implements/updates policy, analyzers, and tests    │
  └───────────────────────────────┬──────────────────────────────────┘
                                  ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ SIMULATE                                                         │
  │  simulate_policy.py generates artifacts/<task>/ with:            │
  │  - augmented rollout video (overlays)                            │
  │  - joints.csv                                                    │
  │  - VLM transcript JSON (if enabled)                              │
  │  - custom analyzer outputs                                       │
  └───────────────────────────────┬──────────────────────────────────┘
                                  ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ CODING FEEDBACK LOOP                                             │
  │  Coding model analyzes simulation outputs (video frames, logs,   │
  │  transcripts, analyzer results), adds focused analyzers/tests,   │
  │  and loops back to DESIGN & IMPLEMENT.                           │
  └──────────────────────────────────────────────────────────────────┘
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For rollout validation with Gemini / VLM analyzers (and `.env` loading):

```bash
pip install -e ".[vlm]"
cp .env.example .env   # set GEMINI_API_KEY; do not commit .env
```

MuJoCo uses an OpenGL backend for offscreen RGB (`mujoco.Renderer`). On a normal desktop, the default platform GL is fine. In headless CI, use `UR5GripperEnv(enable_rgb=False)` or configure OSMesa/EGL per the [MuJoCo rendering docs](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer). Run full RGB rollouts (`simulate_policy.py`) in a terminal where rendering already works.

### Optional: inverse kinematics (``mink``)

Differential IK on the **same** simulator MJCF uses [mink](https://github.com/kevinzakka/mink) (MuJoCo-based) plus a QP backend:

```bash
pip install -e ".[ik]"
```

## Task workflow

### Inputs

Create a **task bundle** at `policies/impl/<task>/`:

| File | Purpose |
|------|---------|
| `<task>.yaml` | **`task_spec.inline`** (intent), optional **`policy_module`** |
| `<task>.py` | `policy(obs, step, env) -> ctrl` (and optional `reset`) |
| `<analyzer_type>.py` | Optional policy-specific analyzer (`build(params) -> analyzer`) |
| `tests/test_<task>.py` | Headless unit tests (paired with the task) |

### Canonical loop (per task)

From the repo root with the venv active:

```bash
# 1 — Headless unit tests
pytest -q tests/test_<task>.py

# 2 — Simulation rollout (writes artifacts/<task>/)
python scripts/simulate_policy.py --config policies/simulate_policy.example.yaml \
  policies/impl/<task>/<task>.py --run-dir artifacts/<task>

# 3 — Re-run after code/analyzer/test updates
# python scripts/simulate_policy.py --config policies/simulate_policy.example.yaml \
#   policies/impl/<task>/<task>.py --run-dir artifacts/<task>
```

**Design & implement:** done in policy/analyzer/test files for the task.

**Simulate:** produces augmented rollout video, joints log, and analyzer artifacts.

**Coding feedback loop:** analyze artifacts (including extracted video frames and analyzer JSON), implement fixes, add focused tests/analyzers, and repeat.

**Outputs** under `artifacts/<task>/`: `rollout.mp4`, `metrics.txt`, `joints.csv`, `rollout.vlm_transcript.json` (when VLM transcriber is enabled), and optional custom analyzer JSON files.

`joints.csv` logs joint `qpos` (and free-joint pose columns), then for each actuator a **`target_*`** column (the vector returned by the policy for that step) and a **`ctrl_*`** column (`data.ctrl` after physics, i.e. applied command). Floats are rounded to five decimal places to keep files smaller.

## Third-party assets

The robot uses **vendored** [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `universal_robots_ur5e` OBJ meshes under `src/robot_manipulation_sim/mjcf/menagerie_ur5e/` (see `NOTICE.txt` and `MENAGERIE_LICENSE` there). Default scene: `ur5e_two_finger_scene.xml`.
