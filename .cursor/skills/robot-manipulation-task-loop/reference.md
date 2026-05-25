# Reference — robot_manipulation_sim

## Layout

| Path | Role |
|------|------|
| `src/robot_manipulation_sim/env.py` | `UR5GripperEnv`, `map_normalized_actions` |
| `src/robot_manipulation_sim/cameras.py` | `render_rgb`, `render_cameras`, `CameraSpec`, `render_multiview_strip`, `render_multiview_grid` |
| `src/robot_manipulation_sim/mjcf/ur5e_two_finger_scene.xml` | Default scene: Menagerie **UR5e** meshes + **Robotiq 2F-85** gripper + cameras + `grasp_box` |
| `src/robot_manipulation_sim/mjcf/menagerie_ur5e/` | Vendored Menagerie `universal_robots_ur5e` assets + `ur5e_robot.xml` fragment |
| `examples/simulate_policy.example.yaml` | Defaults for **`scripts/simulate_policy.py`** (policy path, **`run_dir`**, steps, video grid, joint log interval, …); auto-used when present unless you pass **`--config`** elsewhere |
| `scripts/simulate_policy.py` | **Simulation only**: loads **`examples/simulate_policy.example.yaml`** by default (or **`--config`**); writes **`rollout.mp4`** (2×2 camera grid), **`metrics.txt`**, **`joints.csv`** under **`run_dir`**, or legacy **`--video`** layout; CLI overrides YAML (no Gemini / VLM) |
| `scripts/validate_rollout.py` | **Rollout validation**: **`--config path.yaml`** runs ordered **analyzers** over **`simulation.*`** paths (video, metrics, joints); see **`examples/validation.example.yaml`** and **`src/robot_manipulation_sim/validation/`** |
| `scripts/robot_manipulation_loop.py` | **Automated harness**: `pytest` → `simulate_policy.py` → VLM → optional Gemini **full-file policy repair** until **`pass`** or **`--max-iterations`** |
| `src/robot_manipulation_sim/validation/analyzers/generic/` | **Generic** (default) analyzers: **`vlm_observer`**, **`joints_csv_trajectory`**. Maintained with the harness; **do not** add or change these from the robot-manipulation skill. |
| `src/robot_manipulation_sim/validation/analyzers/task_specific/joints_csv_base_rotation.py` | Bundled **task-specific** joints analyzer: **`<stem>.joints_base.json`** from **`joints.csv`** (base pan + home hold). **New task → new module under `task_specific/` + `TASK_SPECIFIC_REGISTRY`** (see skill), not under `generic/`. |
| `examples/task/<task>.yaml` (+ **`<task>.py`**) | Per-task **`task_spec`** + **`task_analyzers`**; **`task:`** in the main config must equal **`<task>`** exactly. Task files always live in a **`task/`** directory next to the validation YAML. |

## MJCF identifiers

- **Manipuland body**: `grasp_box` (free joint); geom `box_geom`.
- **Arm joints**: `shoulder_pan_joint`, `shoulder_lift_joint`, `elbow_joint`, `wrist_1_joint`, `wrist_2_joint`, `wrist_3_joint`.
- **Gripper**: **Robotiq 2F-85** adaptive gripper (Menagerie kinematics + tendon ``split``; actuator ``a_gripper`` ctrl **0–255**). Legacy logs may still show ``gripper_slide`` if you use an older MJCF.
- **Cameras in MJCF** (not all used in every build): `overview` (perspective), `topdown`, `side_rgb`, `wrist_rgb` (tool-mounted). Default **`DEFAULT_CAMERAS`** for observations / rollout is **`overview`** + **`wrist_rgb`**. Rollout MP4 is a **2×2** grid: **row0** = perspective RGB | wrist RGB; **row1** = matching **grayscale depth** maps (``depth_to_grayscale_rgb``: scene **extent**, z-near/far × extent, percentiles, then **exponential decay** in normalized depth for near-field contrast — see ``render_rollout_rgb_depth_grid`` in **`cameras.py`**, called from **`simulate_policy.py`**). When recording video, **thin polylines** trace each link’s COM path on the **overview RGB and overview depth** tiles only (``--no-overview-traces`` to disable).

## Success helper

`UR5GripperEnv.lift_success(min_height=0.12)` uses body `grasp_box` world-frame \(z\).

## Rollout validation (YAML + analyzers)

**`scripts/validate_rollout.py --config …`** loads **`version`**, optional **`base_dir`**, optional **`task`** (loads **`task/<task>.yaml`** next to the validation config for **`task_spec`** + **`task_analyzers`**; requires **`task/<task>.py`**), resolves rollout paths under **`artifacts/<task>/`** (unless **`simulation.*`** overrides), optional top-level **`task_spec`** overrides, and either legacy **`analyzers`** or **`analyzers_head`** / **`analyzers_tail`** (when **`task`** is set). Analyzers run **in order**; the process exits with the **worst** exit code among enabled analyzers.

Shipped analyzers (see **`robot_manipulation_sim.validation.analyzers`**; **`REGISTRY`** merges **`GENERIC_REGISTRY`** and **`TASK_SPECIFIC_REGISTRY`**):

- **`artifact_manifest`** (**`generic/`**): writes **`validation_manifest.json`** next to the video (resolved paths + whether **`task_spec`** was configured). Result **`artifacts`** include **`manifest_path`** and **`manifest`**.
- **`joints_csv_trajectory`** (**`generic/`**): Task-agnostic **`joints.csv`** metrics — configuration-space path length (L2 along samples), velocity / acceleration / jerk L2 norms (RMS and peaks), optional per-second summaries. Optional YAML **`params`**: **`max_rms_jerk`**, **`max_peak_jerk`** (if set, **`pass`** / exit code reflect them; otherwise defaults **`pass`**). Output **`<video_stem>.trajectory.json`** (see **`output_stem`**, **`json_out`**, **`no_json_file`**).
- **`joints_csv_base_rotation`** (**`task_specific/`**): Reads **`joints.csv`** for the **slow shoulder_pan + other joints at episode start** rubric; writes **`<video_stem>.joints_base.json`**. When the **task changes**, add a **new** module under **`task_specific/`** and a **`TASK_SPECIFIC_REGISTRY`** entry (see robot-manipulation-task-loop skill)—do not grow one omnibus joint analyzer and do not add rubrics under **`generic/`**.
- **`vlm_observer`** (**`generic/`**): Gemini **observer + task streams** (see that module); parameters are set under **`analyzers[].params`** in YAML (**`mode`**, **`model`**, **`max_duration_seconds`**, **`max_frames`**, **`inline_video_max_mb`**, **`dry_run`**, **`json_out`**, **`no_json_file`**).

### `vlm_observer` (Gemini)

Uses **`google.genai`** for **`mode: video`** (inline MP4/MOV bytes under a size cap) and **`google.generativeai`** for **`mode: frames`** (PNG samples via the legacy client):

- **Default `mode: video`**: the evaluated clip is sent **inline** when its size is ≤ **`inline_video_max_mb`** (default **19**), matching typical small-video **`generateContent`**. If the clip exceeds that cap, the analyzer **fails** with guidance to shorten **`max_duration_seconds`**, raise the cap slightly if your quota allows, or use **`mode: frames`**. Auth: **`GEMINI_API_KEY`**. For Files API ingest issues on other tooling, see **`python scripts/diagnose_gemini_file_api.py`**.
- **Length cap**: **`max_duration_seconds`** (default **45**) — only the **first N seconds** are clipped with **ffmpeg** (`-c copy`) when the rollout is longer.
- **`mode: frames`**: **ffmpeg** uniformly samples **`max_frames`** PNGs (default **10**, clamped **4–16**) at a bounded fps (**0.12–2.5** Hz).
- **Model**: **`GEMINI_MODEL`** or **`params.model`** (default **`gemini-3.5-flash`**).
- **Auth**: **`GEMINI_API_KEY`** or **`GOOGLE_API_KEY`** (unless **`dry_run: true`**).
- **Input**: paths come from **`simulation`** in the YAML; **`metrics_file`** text is appended to the prompt. **`task_spec`** from the YAML (when non-empty) is embedded for the **task-aligned** fields only; the **neutral** stream must ignore unstated goals. If there is no **`task_spec`**, the prompt instructs fixed **N/A** placeholders for the task stream.
- **Output**: prints JSON to stdout; by default writes **`<video_stem>.vlm.json`** next to the MP4 (**`json_out`** / **`no_json_file`** in params). Verdict includes **`motion_controlled`**, **`summary_task_agnostic`**, **`summary_task_evaluation`**, **`second_by_second_neutral`**, **`second_by_second_task`** (same timeline length); **exit non-zero** if **`pass`** is not true (`pass` = panels + controlled motion + no obvious catastrophic collision, not task completion).

**Typical flow**: run **`simulate_policy.py`** to produce **`rollout.mp4`** + **`metrics.txt`** (+ **`joints.csv`**), copy or edit **`examples/validation.example.yaml`** (set **`task_spec.inline`** and paths), then **`python scripts/validate_rollout.py --config …`**. Reconcile **`task_spec`** with the JSON **after** validation.

**Batch loop** (if present in your tree): **`scripts/robot_manipulation_loop.py`** runs **`simulate_policy.py`** then **`validate_rollout.py`** each iteration, and optionally rewrites the policy file via a separate Gemini JSON call until **`pass`** or **`--max-iterations`**.

## VLM rollout rubric

Use this structure when doing a **manual** second opinion in chat (attach **`rollout.mp4`**). The **`vlm_observer`** analyzer asks the model for JSON with a **neutral** stream (**`summary_task_agnostic`**, **`second_by_second_neutral`**) plus a **task-aligned** stream (**`summary_task_evaluation`**, **`second_by_second_task`**) using **`task_spec`** from your validation YAML when present.

**Context to paste with the video**

- One MP4: **2×2 grid** — top row **perspective RGB** (`overview`) | **wrist RGB** (`wrist_rgb`); bottom row **grayscale depth** for the same two views.
- Optional: your **intent / success criteria** (e.g. from **`task_spec.inline`** in the validation YAML)—the automated **`vlm_observer`** uses it for **`summary_task_evaluation`** / **`second_by_second_task`** only.
- Paste **terminal summary**: `SUCCESS_RATE`, each line `episode … final_box_z=…`, and any tracebacks.

**Questions for the VLM (manual review)**

1. **Layout**: Confirm you see four distinct tiles in a 2×2 grid; briefly say what each shows.
2. **Observed behavior**: Describe what the robot and box are doing over time (reach, grasp, lift, hold, idle, etc.) with rough phase (early / mid / late in the clip).
3. **Evidence**: Point to **which tile** best supports your description (e.g. perspective RGB vs wrist RGB vs either depth panel).
4. **Defects**: Note **slip**, **collision** (self, floor, box ejected), **jitter**, **idle** behavior, or **black/empty** panel if any.
5. **Vs metrics**: If terminal `final_box_z` suggests success but video disagrees (or the reverse), call that out explicitly.
6. **Timeline**: For scripted JSON from **`vlm_observer`**, expect **`second_by_second_neutral`** and **`second_by_second_task`**: paired strings per second from clip start (see **`generic/vlm_observer.py`**). The generic **`joints_csv_trajectory`** analyzer emits similar per-second lines from **`joints.csv`** (metrics-only neutral stream; task stream echoes **`task_spec`** excerpt).

**Agent follow-up**

- Turn findings into **concrete code or MJCF edits**, then re-run **`pytest`**, **`simulate_policy.py`**, **`validate_rollout.py --config …`**, and **compare** neutral vs task streams to **`task_spec`** until intent and observation align (or document a blocker).

When using **only** the automated script, read **`rollout.vlm.json`** and reconcile **`summary_task_evaluation`** / **`second_by_second_task`** with **`task_spec`**, plus **`metrics.txt`** and **`joints.csv`**, instead of a chat transcript.
