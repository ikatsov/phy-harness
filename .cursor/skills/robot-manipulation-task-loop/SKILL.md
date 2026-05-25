---
name: robot-manipulation-task-loop
description: >-
  Develops and validates robot manipulation task code (grasp, lift, place) against
  the MuJoCo UR5e + gripper harness in this repository. Prefer a Cursor-agent loop: use
  scripts/simulate_policy.py and scripts/validate_rollout.py (YAML validation: generic analyzers
  plus task-specific ones) as tools (plus pytest),
  read simulator artifacts and VLM JSON, compare ``rollout.vlm.json`` streams to ``task_spec`` in the validation YAML (e.g. ``task_spec.inline``), edit the **policy** and its **paired unit test**, and when the **task** changes author **task-specific** analyzers under ``validation/analyzers/task_specific/`` (VLM-based, joint-log-based, or other—see **Edit boundary**). The user must name the **exact task stem** (e.g. ``base_rotation``) matching ``<cfg_dir>/task/<task>.yaml`` and ``<cfg_dir>/task/<task>.py``; work only on that bundle. Do not edit ``validation/analyzers/generic/`` from this skill. Do not rely on Gemini to overwrite the whole policy unless the user explicitly asks.
  Optional: scripts/robot_manipulation_loop.py with --no-auto-fix can chain rollout+VLM without
  auto-repair. Use when the user asks for manipulation tasks, sim validation, policy iteration,
  rollout videos, VLM validation, MuJoCo UR5e scenes, gripper control, or camera-based
  observation loops in robotic-dev-agent.
---

# Robot manipulation task loop (MuJoCo harness)

## Scope

This skill describes a **Cursor agent + harness** loop: the agent **runs repo scripts as tools**, **reads simulator and VLM artifacts** (not only **`rollout.vlm.json`**), and **updates the user’s policy and its paired regression tests**—not an automated “Gemini rewrites the entire policy” path unless the user explicitly wants that.

## Task name contract (mandatory)

Before editing anything, **confirm the task stem** with the user (or from their message). It must be the **exact** identifier used in all of the following—**no path, no extension, no alias**:

- **`task:`** in the main validation YAML (string next to the config file, e.g. **`base_rotation`**).
- **`<cfg_dir>/task/<task>.yaml`** — with **`examples/validation.example.yaml`**, **`cfg_dir`** is **`examples/`**, so paths are **`examples/task/<task>.yaml`** (e.g. **`examples/task/base_rotation.yaml`**).
- **`<cfg_dir>/task/<task>.py`** — the policy passed to **`simulate_policy.py`** (e.g. **`examples/task/base_rotation.py`**).
- **Rollout bundle** — **`artifacts/<task>/`** under **`base_dir`** (e.g. repo **`artifacts/base_rotation/`** when **`base_dir: ..`** from **`examples/`**), unless **`simulation.*`** paths override.

**`load_validation_yaml`** rejects invalid names and **requires** the paired **`.py`** to exist beside the task YAML. If the user does not state this stem explicitly, **ask** which task they mean; **do not** assume a different file (e.g. do not switch from **`base_rotation`** to another stem without confirmation).

**Edit boundary (mandatory):** You may edit **only**:

1. **The policy file** — **`examples/task/<task>.py`** (same layout for any validation config: **`<cfg_dir>/task/<task>.py`**), where **`<task>`** is exactly the **`task:`** value. It defines **`policy`** and private helpers in that module.

2. **The corresponding unit test file** — **one** module under **`tests/`** devoted solely to that task stem: **`tests/test_<task>.py`** (e.g. **`task: base_rotation`** → **`tests/test_base_rotation.py`**). Use it to encode and update behavioral expectations alongside **`policy`** changes. You may **create** that file only if the user agrees or it already exists as the paired test for this task.

3. **The validation YAML** — the file passed as **`--config`** to **`validate_rollout.py`** (typically **`examples/validation.example.yaml`**). Set **`task:`** to the stem above so the loader reads **`examples/task/<task>.yaml`** for **intent** (**`task_spec`**) and **task-specific analyzers**; keep generic analyzers in **`analyzers_head`** / **`analyzers_tail`**.

4. **Task-specific analyzers (narrow `src/` exception)** — when the user gives a **new** task (new success criteria / motion contract), **do not** extend generic analyzers under **`validation/analyzers/generic/`** (those are **`vlm_observer`**, **`joints_csv_trajectory`**, etc.—harness-owned). Instead **add a new compact module** under **`src/robot_manipulation_sim/validation/analyzers/task_specific/`** (any clear filename; **`joints_csv_<short_task_slug>.py`** is a good convention for CSV rubrics). Implement one analyzer class. For analyzers that write JSON verdict files, use the same dual-stream keys as **`vlm_observer`**: **`summary_task_agnostic`**, **`summary_task_evaluation`**, **`second_by_second_neutral`**, **`second_by_second_task`**, plus **`pass`**, unless a lighter contract is enough. Register **`"<type_name>": lambda p: YourClass(p)`** in **`analyzers/__init__.py`** inside **`TASK_SPECIFIC_REGISTRY`** only (do **not** add to **`GENERIC_REGISTRY`**). Add that **`type`** to the validation YAML. **Disable or remove** the previous task-specific analyzer from YAML when it no longer applies; you may delete obsolete modules under **`task_specific/`** only if nothing references them. Use **`task_specific/joints_csv_base_rotation.py`** as the template for the current **base pan + other joints at home** example. Keep each file **small and explicit** (known CSV columns, thresholds via YAML **`params`** only).

You **must not** edit harness or shared code, including: **`scripts/**`**, **`mjcf/**`**, **`src/robot_manipulation_sim/`** **outside** **`validation/analyzers/task_specific/*.py`** and the **`TASK_SPECIFIC_REGISTRY`** block in **`validation/analyzers/__init__.py`**, **`validation/analyzers/generic/**`**, **`GENERIC_REGISTRY`**, **any other `tests/**` file** (not the paired **`tests/test_<task>.py`** for the active **`task:`** stem), **`.cursor/**`**, **`pyproject.toml`**, **`README.md`**, etc.—**even if a fix seems obvious**. For anything else, **stop**, explain what is needed, and **request the user** to implement it. You may still **read** harness files and docs.

Harness code (read-only context unless the user changes it): `src/robot_manipulation_sim/` **except** new/edited **`validation/analyzers/task_specific/*.py`** + **`TASK_SPECIFIC_REGISTRY`** lines in **`analyzers/__init__.py`** (`UR5GripperEnv`, MJCF under `mjcf/`, **`generic/`** analyzers).

## Tools (what to run)

| Tool | Role | API / GL |
|------|------|----------|
| **`pytest`** (e.g. `-q`) | Fast regression (run after edits to the policy and/or its **paired** `tests/test_<task>.py`) | No |
| **`scripts/simulate_policy.py`** | Rollout; writes **`rollout.mp4`**, **`metrics.txt`**, **`joints.csv`** under **`--run-dir`** | MuJoCo **GL** for video |
| **`scripts/validate_rollout.py --config …`** | Rollout **validation pipeline**: YAML + **analyzers** (e.g. **`artifact_manifest`**, generic **`vlm_observer`** / **`joints_csv_trajectory`**, task-specific **`joints_csv_*`** under **`task_specific/`**); see **`examples/validation.example.yaml`** | **Gemini** for **`vlm_observer`**; CSV analyzers are CPU-only |

Typical commands:

```bash
source .venv/bin/activate
pip install -e ".[vlm]"
export GEMINI_API_KEY=...   # or GOOGLE_API_KEY — only for validate_rollout / vlm_observer (not simulate_policy)

pytest -q
python scripts/simulate_policy.py --config examples/simulate_policy.example.yaml --episodes 1 --steps 400
python scripts/validate_rollout.py --config examples/validation.example.yaml
```

Edit **`examples/validation.example.yaml`** (or a copy): set **`task:`** to the **exact** stem, split **`analyzers_head`** / **`analyzers_tail`**, and edit **`examples/task/<task>.yaml`** for rubric + **`task_spec.inline`**.

Keep **intent / success criteria** in **`task_spec.inline`** inside **`examples/task/<task>.yaml`** (or override with top-level **`task_spec`** in the main config). The **`vlm_observer`** model receives that text for the **task-aligned** fields only (**`summary_task_evaluation`**, **`second_by_second_task`**); **`summary_task_agnostic`** and **`second_by_second_neutral`** stay pixel/metrics-only. After the run, read **`rollout.vlm.json`** (both streams). Edit the task YAML when criteria change.

**`vlm_observer`** (defaults): **`mode: video`** sends the clip **inline** via **`google.genai`** when under **~19 MiB** (same key as the Gemini UI). If the clip is larger, the analyzer **fails** with guidance to shorten **`max_duration_seconds`** or use **`mode: frames`**. JSON **`pass`** reflects **panels + controlled motion + no obvious catastrophic collision**, not unstated task goals. Exit **non-zero** if any analyzer fails. Env: [reference.md](reference.md#vlm_observer-gemini).

- **Manual / second-opinion**: attach **`rollout.mp4`** in chat and use the rubric in [reference.md](reference.md#vlm-rollout-rubric).
- **CI / no API key**: **`validate_rollout.py --config …`** with **`vlm_observer.params.dry_run: true`**, or skip VLM and only run **`simulate_policy.py`**.

## Simulator artifacts (`--run-dir`, same folder)

After **`simulate_policy.py … --run-dir DIR`**, treat **`DIR/`** as the primary evidence bundle. **Always** use it alongside **`rollout.vlm.json`** when debugging or iterating—vision verdicts and logs can disagree.

| Artifact | What it is | Why read it |
|----------|------------|---------------|
| **`joints.csv`** | Per–sim-step log: episode, **`sim_step`**, **`time_sec`**, joint **qpos** (incl. free-joint box pose), **`ctrl_*`** actuators | Spot oscillation, saturation, drift, gripper chatter, box pose vs. lift; correlate timesteps with **`rollout.mp4`**. Sampling interval: **`--joint-log-interval`** (default every step). |
| **`metrics.txt`** | Aggregate run summary (success rate, **`episode_*_final_box_z`**, thresholds) | Numeric ground truth; optional context appended to the VLM prompt—compare to **`summary_task_agnostic`** / **`second_by_second_neutral`** and to **`summary_task_evaluation`** / **`task_spec`**. |
| **`rollout.mp4`** | 2×2 **RGB + depth** grid (perspective + wrist, then matching depth tiles) | Motion quality, contacts, panel visibility; pairs with **`joints.csv`** rows at the same **`sim_step`**. |

If **`joints.csv`** is missing or sparse, you may ask the user to re-run with a different **`--joint-log-interval`** (or to change **`simulate_policy.py`**); **do not** edit the script yourself under this skill.

## When to stop and ask the user (harness / evaluation)

Escalate with a **concrete request** (file(s), change, rationale) if any of the following apply:

- **Simulation harness**: e.g. **`simulate_policy.py`** logging columns, default **`--run-dir`** layout, video tiling, episode boundaries, or MJCF / **`UR5GripperEnv`** behavior.
- **Evaluation / VLM harness**: e.g. **`scripts/validate_rollout.py`**, **`generic/vlm_observer.py`** prompts, model env vars. **Adding** a **new** task-specific analyzer under **`task_specific/`** when the **task changed** is **in scope** (see **Edit boundary** item **4**). Broad refactors of **`validation/`** or edits to **`generic/`** or **`vlm_observer.py`** without user scope: **stop** and ask.
- **Scene / physics**: MJCF bodies, cameras, actuators, or anything outside the **policy + paired test** pair.
- **Other tests or packaging**: any **`tests/**`** file **other than** the paired **`tests/test_<task>.py`** or **small tests colocated with a new `task_specific` analyzer** you added; dependency or CI edits; broad **`src/`** refactors outside **`validation/analyzers/task_specific/*.py`** + **`TASK_SPECIFIC_REGISTRY`**.

Until the user makes those changes (or explicitly widens scope), **limit edits** to the **policy file**, **paired test**, **validation YAML**, and **task-specific analyzers + `TASK_SPECIFIC_REGISTRY` registration** as in **Edit boundary**.

## Preferred agent loop (policy + paired tests in Cursor)

1. **Clarify the task stem** (see **Task name contract**) and encode intent in **`task_spec.inline`** in **`examples/task/<task>.yaml`** (or top-level **`task_spec`** overrides in the main config / legacy **`policy_module`**). If the task is **new** vs the last iteration, **add or replace** a **compact analyzer under `task_specific/`** (see **Edit boundary** item **4**); rely on generic **`joints_csv_trajectory`** / **`vlm_observer`** for broad signals—do not overload them with task rubrics.
2. **Implement or edit** `policy(...)` (and helpers **in the policy file**). Update **`tests/test_<task>.py`** when expectations change (same **`<task>`** as **`task:`** / **`examples/task/<task>.py`**).
3. Run **`pytest -q`** (focus on the paired test + policy; if failures need **other** tests or **`src/`** fixes, stop and ask the user).
4. Run **`simulate_policy.py … --run-dir DIR`** (suggest CLI flags to the user as needed) → read **`metrics.txt`**, **`joints.csv`**, **`rollout.mp4`** as needed; progress lines print to the terminal only.
5. Run **`python scripts/validate_rollout.py --config …`** → read **`rollout.<video_stem>.*.json`** from task-specific analyzers (if enabled), **`rollout.trajectory.json`** from **`joints_csv_trajectory`** (if enabled), and **`rollout.vlm.json`** (**`summary_task_agnostic`** / **`second_by_second_neutral`** vs **`summary_task_evaluation`** / **`second_by_second_task`**) against **`task_spec`** from the YAML.
6. If something is wrong: **cross-check** **joints JSON** (numeric) and VLM **neutral** lines against **`joints.csv`** / **`metrics.txt`**; check **task** lines against **`task_spec`**. If the fix belongs in the **policy, paired test, validation YAML, or task-specific analyzer**, edit and repeat from step 3. If it belongs in the **rest of the harness**, **stop** and ask the user.

The **harness** is: repo + venv + scripts + **your** policy + **your** paired test file + **your** validation YAML + **your** registered **`task_specific/*`** modules for the current task. Everything else stays read-only unless the user expands scope.

## Rollout video only (no Gemini)

```bash
python scripts/simulate_policy.py --config examples/simulate_policy.example.yaml
```

Use **`artifacts/<task>`** (same stem as **`task:`** in your validation YAML) so **`validate_rollout.py`** reads the latest rollout next to **`base_dir`**.

## Optional: `robot_manipulation_loop.py` (shell orchestrator)

`scripts/robot_manipulation_loop.py` chains **`pytest`** → **`simulate_policy.py`** → **`validate_rollout.py`**. With **`--no-auto-fix`** (recommended alignment with this skill), it **does not** ask Gemini to replace the policy file—it only runs the same tools you would run by hand. With **`--auto-fix`** (default in that script), it **does** call Gemini for a full-file policy rewrite and backups **`.bak.iterN.py`**; **use that only when the user explicitly wants automated policy replacement**, not as the default interpretation of this skill.

```bash
python scripts/robot_manipulation_loop.py examples/task/base_rotation.py \
  --run-dir artifacts/base_rotation --no-auto-fix
```

Tuning flags on the loop still forward VLM options (**`--vlm-mode`**, **`--gemini-model`**, etc.) when present; see **`--help`**.

## Contract for task solutions

- **Editable surface:** **`examples/task/<task>.py`**, **exactly one** paired **`tests/test_<task>.py`**, the **validation YAML**, and **task-specific analyzers under `task_specific/` + `TASK_SPECIFIC_REGISTRY`** (see **Edit boundary**). All other paths are **read-only** unless the user does them (or widens scope).
- Prefer **Python** in the policy file; keep the paired test **narrow** (imports **`policy`** / env, no duplicate harness of **`simulate_policy.py`**).
- **VLM**: put **intent / success-criteria** in **`task_spec.inline`** (or **`policy_module`**). **`vlm_observer`** uses it for **task-aligned** JSON fields; the **neutral** VLM fields stay pixel/metrics-only.
- **Control**: actuator **targets** length `env.nu` (7). Clip to `env.model.actuator_ctrlrange` when needed.
- **Observations**: `obs["qpos"]`, `obs["qvel"]`, `obs["ctrl"]`, `obs["box_height"]`, optional `obs["images"]` if `enable_rgb=True` (requires GL). Need **`enable_rgb`** or env construction changes? **Ask the user**, do not edit **`UR5GripperEnv`** here.

## Other CLI flags (suggest to user; do not change script defaults in repo)

- **`--strict`**, **`--rgb`**, **`--steps`**, **`--episodes`**, **`--lift-z`**, **`--joint-log-interval`**, **`--video-cell-*`**, etc.: document what you need; the user runs **`simulate_policy.py`** with them or edits the harness.
- **`validate_rollout.py`**, **`examples/validation.example.yaml`** (template), **`generic/vlm_observer.py`**: **request**, do not edit under this skill—**except** you may add/replace **`validation/analyzers/task_specific/*.py`** and **`TASK_SPECIFIC_REGISTRY`** lines per **Edit boundary** item **4**.

## Iteration checklist

```
Task loop (Cursor agent + harness):
- [ ] Confirm **`task:`** stem matches **`examples/task/<task>.yaml`** and **`examples/task/<task>.py`** exactly; confirm paired **`tests/test_<task>.py`** and **validation YAML** path for **`--config`** (create paired test only with user OK)
- [ ] Update **`task_spec.inline`** in **`examples/task/<task>.yaml`** (or the task YAML your config uses); if the **task** is **new** (not just tuning thresholds), add a **`task_specific/`** analyzer + **`TASK_SPECIFIC_REGISTRY`** + **`task_analyzers`** in that YAML (or disable/remove the old task-specific analyzer)
- [ ] Use MJCF / API names from docs or **read-only** harness (do not edit scene/package)
- [ ] Implement `policy(...)` in the **policy file**; update **paired test** when expectations shift
- [ ] pytest -q
- [ ] simulate_policy.py (YAML defaults or ``--config``) → read **`metrics.txt`**, **`joints.csv`**, **`rollout.mp4`**
- [ ] ``python scripts/validate_rollout.py --config …``  → read **joints JSON** (if any) + **`rollout.vlm.json`**; compare streams to **`task_spec`**
- [ ] Read **`rollout.vlm.json`**; reconcile neutral stream with logs/metrics
- [ ] If fix fits **policy + paired test + validation YAML + task-specific analyzer**: edit and repeat. Else: **stop** and request user harness / eval / other-test changes
```

## Notes

- **MuJoCo GL** is required to **encode** video under **`--run-dir`** or **`--video`**. **`GEMINI_API_KEY`** / **`GOOGLE_API_KEY`** is only needed for **`validate_rollout.py`** / **`vlm_observer`** (not **`simulate_policy.py`** or **`dry_run: true`**).
- **Menagerie assets**: `mjcf/menagerie_ur5e/`.

## Additional resources

- [reference.md](reference.md) — file map, rubric, **VLM env vars**
