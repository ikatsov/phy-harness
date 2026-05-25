"""Policy for ``task: base_rotation`` — slow rotation about the robot base (vertical axis).

Paired files (same stem ``base_rotation``):
  ``examples/task/base_rotation.yaml`` — ``task_spec`` + ``task_analyzers`` (validation loads it when the main config sets ``task: base_rotation``).
  ``examples/task/base_rotation.py`` — this module (pass to ``simulate_policy.py``).

Uses **closed-loop** targets on ``shoulder_pan_joint`` (small steps from measured ``qpos``).

Non-pan **joint** actuators keep a **fixed position setpoint** equal to ``qpos`` at episode start (no per-step
slew from measured ``q`` toward that setpoint). The Robotiq gripper (tendon actuator) keeps the initial **ctrl**
(0–255). Slew-only holding on the arm fought the high-gain position actuators and **blocked base rotation** in practice.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robot_manipulation_sim.env import UR5GripperEnv

# Compare ``rollout.vlm.json`` / ``rollout.joints_base.json`` to ``task_spec`` in ``base_rotation.yaml`` after ``validate_rollout.py``.
#
# Ground truth for *this* task: ``joints_csv_base_rotation`` on ``joints.csv`` (``rollout.joints_base.json``);
# generic smoothness: ``joints_csv_trajectory`` → ``rollout.trajectory.json``.
# ``vlm_observer`` neutral text may still say “stationary” because base-only motion is slow and subtle
# in the 2x2 rollout video—do not treat that narrative as contradicting the joint log.

# rad/s about base (shoulder_pan_joint, actuator index 0). With non-pan held at q(t0), pan tracking
# is sub-linear below ~0.11 rad/s on the bundled harness. 0.111 rad/s keeps margin on ``min_pan_total_rad``
# for a 500-step / 10 s rollout without tripping ``home_max_span_per_s`` / excursion limits.
BASE_ROTATION_SPEED = 0.7

# Latched at ``simulate_policy`` step ``k == 0`` each episode: non-pan **ctrl** (incl. Robotiq 0–255).
_episode_start_ctrl: list[float] | None = None


def reset() -> None:
    """Optional hook: ``simulate_policy`` calls ``reset()`` after ``env.reset()`` each episode."""
    global _episode_start_ctrl
    _episode_start_ctrl = None


def _actuator_qadr(env: UR5GripperEnv, act_idx: int) -> int:
    """qpos address for joint actuators; tendon / non-joint actuators are not supported here."""
    jid = int(env.model.actuator_trnid[act_idx, 0])
    return int(env.model.jnt_qposadr[jid])


def policy(obs: dict[str, Any], step: int, env: UR5GripperEnv) -> np.ndarray:
    """Advance pan with small closed-loop steps; hold other actuators at latched episode-start targets.

    Arm joints (excluding pan) use the measured **qpos** at the first step (same idea as the old
    parallel-gripper policy). The Robotiq **tendon** actuator keeps the initial **ctrl** (0–255).
    """
    global _episode_start_ctrl

    lo = env.model.actuator_ctrlrange[:, 0]
    hi = env.model.actuator_ctrlrange[:, 1]
    pan_step = BASE_ROTATION_SPEED * env.control_dt
    ctrl = np.empty(env.nu, dtype=np.float64)

    if step == 0 or _episode_start_ctrl is None:
        _episode_start_ctrl = []
        for i in range(env.nu):
            if int(env.model.actuator_trntype[i]) == 0:  # mjTRN_JOINT
                qadr = _actuator_qadr(env, i)
                _episode_start_ctrl.append(float(obs["qpos"][qadr]))
            else:
                _episode_start_ctrl.append(float(obs["ctrl"][i]))

    for i in range(env.nu):
        if i == 0:
            qadr = _actuator_qadr(env, i)
            q = float(obs["qpos"][qadr])
            target = q + pan_step
            ctrl[i] = float(np.clip(target, lo[i], hi[i]))
        else:
            ctrl[i] = float(np.clip(_episode_start_ctrl[i], lo[i], hi[i]))
    return ctrl
