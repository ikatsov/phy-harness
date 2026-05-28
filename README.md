# Robotic manipulation dev harness

Python harness for **Universal Robots UR5e** simulation with a **Robotiq 2F-85** adaptive gripper (tendon drive, **0–255** ctrl), **RGB cameras**, and a validation loop for task policies (e.g. grasp, touch, lift). This harness is then used by the coding skill to implemnet the following development loop:

```
  ┌─────────────────────────────────────────────────────────────────┐
  │ INPUTS                                                          │
  │  You: task stem + policies/impl/<task>/  →  <task>.yaml (spec)  │
  └───────────────────────────────┬─────────────────────────────────┘
                                  ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ DESIGN & IMPLEMENT                                              │
  │  Module plan (IK, vision, phases)  →  <task>.py, analyzers,     │
  │                                        tests/test_<task>.py     │
  └───────────────────────────────┬─────────────────────────────────┘
                                  ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ VERIFY                                                          │
  │  pytest  →  simulate_policy.py  →  validate_rollout.py          │
  │              artifacts/<task>/       rollout.*.json, VLM        │
  └───────────────────────────────┬─────────────────────────────────┘
                                  ▼
                    Pass task_spec + analyzers?
                         │              │
                        yes             no
                         │              │
                         ▼              ▼
                   Task solved    You pick next step:
                                  ┌────┬────┬────┐
                                  │4.0 │4.1 │4.2 │
                                  └─┬──┴─┬──┴─┬──┘
                      revise arch ──┘    │    └── update policy
                         │         refine analyzers
                         └──────────────┴──────────────► (loop back to DESIGN)
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

## Robotics devagent (Cursor skill)

The **robotics-devagent** skill (`.cursor/skills/robotics-devagent/SKILL.md`) guides iterative policy development in this repo: read a task spec, implement in small modules (IK, vision, phases), validate with simulation artifacts, and iterate with you in the loop until metrics and analyzers support “task solved.”

**Invoke it in Cursor** with `/robotics-devagent` (or attach the skill) and name the **exact task stem** (e.g. `box_touch`, `base_rotation`).

### What you provide

Create a **task bundle** at `policies/impl/<task>/`:

| File | Purpose |
|------|---------|
| `<task>.yaml` | **`task_spec.inline`** (intent), **`task_analyzers`**, optional **`policy_module`** |
| `<task>.py` | `policy(obs, step, env) -> ctrl` (and optional `reset`) |
| `<analyzer_type>.py` | Optional custom analyzers (`build(params) -> RolloutAnalyzer`) |
| `tests/test_<task>.py` | Headless unit tests (paired with the task) |

The agent edits **only** that bundle and its test—not `src/`, `scripts/`, shared example YAMLs (except setting **`task:`** in `policies/validation.example.yaml`), or other tasks’ bundles.

### Canonical loop (per task)

From the repo root with the venv active:

```bash
# 1 — Headless unit tests
pytest -q tests/test_<task>.py

# 2 — Rollout (video, metrics, joints.csv under artifacts/<task>/)
python scripts/simulate_policy.py --config policies/simulate_policy.example.yaml \
  policies/impl/<task>/<task>.py --run-dir artifacts/<task>

# 3 — Set task: <task> in policies/validation.example.yaml, then validate
python scripts/validate_rollout.py --config policies/validation.example.yaml
```

**Simulate:** only override the policy path and `--run-dir` (steps, `rgb`, video layout stay in `policies/simulate_policy.example.yaml`).

**Validate:** only change **`task:`** in `policies/validation.example.yaml`; do not fork per-task validation YAMLs.

**Outputs** under `artifacts/<task>/`: `rollout.mp4`, `metrics.txt`, `joints.csv`, `rollout.*.json` (task analyzers), `rollout.vlm.json` (when VLM is enabled).

## Third-party assets

The robot uses **vendored** [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `universal_robots_ur5e` OBJ meshes under `src/robot_manipulation_sim/mjcf/menagerie_ur5e/` (see `NOTICE.txt` and `MENAGERIE_LICENSE` there). Default scene: `ur5e_two_finger_scene.xml`.
