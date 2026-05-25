# Robotic manipulation dev harness (MuJoCo)

Python harness for **Universal Robots UR5e** simulation (Menagerie meshes) with a **Robotiq 2F-85** adaptive gripper (tendon drive, **0–255** ctrl), **RGB cameras**, and a validation loop for task policies (e.g. grasp / lift).

## Third-party assets

The robot uses **vendored** [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `universal_robots_ur5e` OBJ meshes under `src/robot_manipulation_sim/mjcf/menagerie_ur5e/` (see `NOTICE.txt` and `MENAGERIE_LICENSE` there). The default scene is `ur5e_two_finger_scene.xml`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Copy **`.env.example`** → **`.env`** and set **`GEMINI_API_KEY`** (used by `scripts/validate_rollout.py` and `robot_manipulation_loop.py` when present). Do not commit `.env`.

MuJoCo uses an OpenGL backend for offscreen RGB (`mujoco.Renderer`). On a normal desktop, the default (`glfw` / platform GL) is fine. In headless environments, install/configure OSMesa/EGL as described in the [MuJoCo rendering docs](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer) or run with `UR5GripperEnv(enable_rgb=False)` for **state-only** observations.

## Quick use

```python
from robot_manipulation_sim import UR5GripperEnv

env = UR5GripperEnv(enable_rgb=True)  # False in CI / no GL
obs = env.reset()
obs = env.step(env._home)  # 6 arm position actuators + Robotiq tendon command (ctrl 0–255)
```

Observations include `images` (when `enable_rgb=True`), `qpos`, `qvel`, `ctrl`, `box_height`, `time`.

## Examples

- **Automated VLM loop** (rollout → judge → optional Gemini policy rewrite until pass): `python scripts/robot_manipulation_loop.py examples/task/base_rotation.py --run-dir artifacts/base_rotation` (if the script omits **`--run-dir`**, check its **`--help`** for the default; needs `pip install -e ".[vlm]"` and **`GEMINI_API_KEY`**; put your **intent / success criteria** in **`task_spec.inline`** in the task YAML—or keep **`VLM_TASK`** in the policy if the loop still expects it—and compare against **`rollout.vlm.json`**; use **`--no-auto-fix`** for a single VLM pass without editing the policy file)
- **Simulate a policy** (writes **`rollout.mp4`** with a 2×2 camera grid—**overview RGB/depth** include thin traced link COM paths unless **`--no-overview-traces`**—plus **`metrics.txt`**, **`joints.csv`** under **`run_dir`**, or legacy **`video`** single-file layout): defaults live in **`examples/simulate_policy.example.yaml`** (auto-loaded when that file exists). Run e.g. `python scripts/simulate_policy.py --config examples/simulate_policy.example.yaml`, or override the policy path / any flag on the CLI (`python scripts/simulate_policy.py examples/task/base_rotation.py --steps 400`). With no default YAML, pass a **`policy_file`** positional plus any **`--run-dir`** / **`--video`** you need.
- **Rollout validation (YAML)** (needs `pip install -e ".[vlm]"` for Gemini analyzers): after `simulate_policy.py … --run-dir DIR`, run `python scripts/validate_rollout.py --config examples/validation.example.yaml`. With **`task:`** set to a stem **`<task>`**, the loader reads **`examples/task/<task>.yaml`** for **`task_spec`** and **`task_analyzers`**, requires **`examples/task/<task>.py`**, merges **`analyzers_head`** + task analyzers + **`analyzers_tail`**, and resolves rollout paths under **`artifacts/<task>/`** (relative to **`base_dir`**) unless **`simulation.*`** overrides. Configs **without** **`task`** keep the legacy shape: top-level **`task_spec`**, single **`analyzers`** list, explicit **`simulation`** paths.
- **Example task** (`task: base_rotation`): policy at **`examples/task/base_rotation.py`** — slow **base-axis** rotation via small closed-loop steps on `shoulder_pan_joint`; other joints held at episode-start setpoints. **Intent** for validation lives in **`examples/task/base_rotation.yaml`**.

Policies are plain Python files defining `policy(obs, step, env)` returning a length-`env.nu` control vector (joint / gripper **targets** in actuator units). For iteration workflows, keep **success criteria** in **`examples/task/<task>.yaml`** (**`task_spec.inline`**) or override from the main config; you can also use **`policy_module`** with **`TASK_SPEC`** / **`VLM_TASK`** so you (or an agent) can compare them to **`validate_rollout.py`** / **`vlm_observer`** output in **`rollout.vlm.json`**.

## Cursor skill

See `.cursor/skills/robot-manipulation-task-loop/SKILL.md` for the iterative **spec → implement → validate in sim** workflow tied to this repo.
