#!/usr/bin/env python3
"""Simulate a policy in the UR5 MuJoCo harness: rollout, optional multi-view MP4, metrics, joint log."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Protocol, TextIO

import imageio.v2 as imageio
import mujoco
import numpy as np

from robot_manipulation_sim.cameras import (
    project_world_positions_to_camera_pixels,
    render_rollout_rgb_depth_grid,
)
from robot_manipulation_sim.env import UR5GripperEnv
from robot_manipulation_sim.simulate_config import SimulateSettings, builtin_defaults, load_simulate_settings


def _load_dotenv() -> None:
    """Load repo-root `.env` regardless of shell cwd (default load_dotenv() only checks cwd)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    load_dotenv(root / ".env.local")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_simulate_config_path() -> Path:
    return _repo_root() / "examples" / "simulate_policy.example.yaml"


class PolicyFn(Protocol):
    def __call__(self, obs: dict[str, Any], step: int, env: UR5GripperEnv) -> np.ndarray: ...


def load_policy(path: Path, symbol: str = "policy") -> PolicyFn:
    spec = importlib.util.spec_from_file_location("user_policy_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_policy_module"] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, symbol, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"{path} must define callable {symbol}(obs, step, env) -> ctrl")
    return fn  # type: ignore[return-value]


# Bodies whose world-frame COM paths are projected onto the ``overview`` tile (RGB + depth).
_OVERVIEW_TRACE_BODIES: tuple[str, ...] = (
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "tool0",
)
_OVERVIEW_TRACE_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 80, 80),
    (255, 200, 80),
    (120, 255, 120),
    (80, 200, 255),
    (180, 120, 255),
    (255, 120, 200),
    (240, 240, 255),
)


def _new_overview_traces() -> dict[str, list[tuple[float, float]]]:
    return {n: [] for n in _OVERVIEW_TRACE_BODIES}


def _append_overview_trace_samples(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    traces: dict[str, list[tuple[float, float]]],
    cell_w: int,
    cell_h: int,
) -> None:
    for name in _OVERVIEW_TRACE_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        xyz = np.asarray(data.xpos[bid], dtype=np.float64).reshape(1, 3)
        uv = project_world_positions_to_camera_pixels(
            model, data, "overview", int(cell_w), int(cell_h), xyz
        )[0]
        if np.all(np.isfinite(uv)):
            traces.setdefault(name, []).append((float(uv[0]), float(uv[1])))


def _overview_traces_to_draw_args(
    traces: dict[str, list[tuple[float, float]]],
) -> tuple[tuple[np.ndarray, tuple[int, int, int]], ...] | None:
    out: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    for name, color in zip(_OVERVIEW_TRACE_BODIES, _OVERVIEW_TRACE_COLORS, strict=True):
        seq = traces.get(name, [])
        if len(seq) < 2:
            continue
        out.append((np.asarray(seq, dtype=np.float64), color))
    return tuple(out) if out else None


def _multiview_frame(
    env: UR5GripperEnv,
    cell_h: int,
    cell_w: int,
    *,
    overview_traces: dict[str, list[tuple[float, float]]] | None = None,
) -> np.ndarray:
    """2×2 grid: row0 = perspective (``overview``) RGB | wrist RGB; row1 = matching depth maps."""
    pers = _overview_traces_to_draw_args(overview_traces) if overview_traces is not None else None
    return render_rollout_rgb_depth_grid(
        env.model,
        env.data,
        cell_h,
        cell_w,
        perspective_camera="overview",
        wrist_camera="wrist_rgb",
        perspective_traces=pers,
    )


def _open_video_writer(path: Path, fps: float) -> Any:
    return imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=None,
    )


def joint_log_header(model: mujoco.MjModel) -> list[str]:
    cols = ["episode", "sim_step", "time_sec"]
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        jt = int(model.jnt_type[j])
        if jt == mujoco.mjtJoint.mjJNT_FREE:
            for lab in ("px", "py", "pz", "qw", "qx", "qy", "qz"):
                cols.append(f"{name}.{lab}")
        elif jt == mujoco.mjtJoint.mjJNT_BALL:
            for k in range(4):
                cols.append(f"{name}.q{k}")
        else:
            cols.append(name)
    for a in range(model.nu):
        an = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"actuator_{a}"
        cols.append(f"ctrl_{an}")
    return cols


def joint_log_row(model: mujoco.MjModel, data: mujoco.MjData, episode: int, sim_step: int) -> list[Any]:
    row: list[Any] = [episode, sim_step, float(data.time)]
    for j in range(model.njnt):
        adr = int(model.jnt_qposadr[j])
        jt = int(model.jnt_type[j])
        if jt == mujoco.mjtJoint.mjJNT_FREE:
            row.extend(float(x) for x in data.qpos[adr : adr + 7])
        elif jt == mujoco.mjtJoint.mjJNT_BALL:
            row.extend(float(x) for x in data.qpos[adr : adr + 4])
        else:
            row.append(float(data.qpos[adr]))
    for a in range(model.nu):
        row.append(float(data.ctrl[a]))
    return row


def _resolve_outputs(
    *,
    run_dir: Path | None,
    video: Path | None,
) -> tuple[Path | None, Path | None, Path | None]:
    """Return (video_path, metrics_path, joints_csv_path)."""
    if run_dir is not None and video is not None:
        raise SystemExit("Use either --run-dir or --video, not both.")
    if run_dir is not None:
        rd = run_dir.resolve()
        rd.mkdir(parents=True, exist_ok=True)
        return rd / "rollout.mp4", rd / "metrics.txt", rd / "joints.csv"
    if video is not None:
        v = video.resolve()
        v.parent.mkdir(parents=True, exist_ok=True)
        return (
            v,
            v.parent / f"{v.stem}.metrics.txt",
            v.parent / f"{v.stem}_joints.csv",
        )
    return None, None, None


def _merge_cli_overrides(settings: SimulateSettings, args: argparse.Namespace) -> SimulateSettings:
    """Apply argparse overrides (``default=argparse.SUPPRESS`` → only set keys appear on ``args``)."""
    names = {f.name for f in fields(SimulateSettings)}
    skip = {"policy_file"}
    changes: dict[str, Any] = {}
    for k, v in vars(args).items():
        if k in skip or k not in names:
            continue
        changes[k] = v
    return replace(settings, **changes) if changes else settings


def _parse_args(argv: list[str] | None = None) -> tuple[SimulateSettings, argparse.ArgumentParser]:
    argv = sys.argv[1:] if argv is None else argv
    _suppress = argparse.SUPPRESS
    p = argparse.ArgumentParser(
        description=(
            "Simulate a manipulation policy in the UR5 MuJoCo scene. "
            "Defaults come from examples/simulate_policy.example.yaml when present, or pass --config PATH. "
            "CLI options override YAML."
        )
    )
    default_cfg = _default_simulate_config_path()
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML with rollout defaults (policy path, steps, video layout, …).",
    )
    p.add_argument(
        "policy_file",
        type=Path,
        nargs="?",
        default=None,
        help="Policy module path (overrides policy_file in YAML when set).",
    )
    p.add_argument("--symbol", default=_suppress, help="Callable name in policy_file")
    p.add_argument("--steps", type=int, default=_suppress)
    p.add_argument("--episodes", type=int, default=_suppress)
    p.add_argument("--rgb", action="store_true", default=_suppress, help="Enable RGB in observations (needs GL).")
    p.add_argument("--lift-z", type=float, default=_suppress, dest="lift_z", help="Success if box COM z exceeds this.")
    p.add_argument(
        "--strict",
        action="store_true",
        default=_suppress,
        help="Exit with code 1 unless every episode meets the lift threshold.",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=_suppress,
        help="Create this directory and write rollout.mp4, metrics.txt, joints.csv.",
    )
    p.add_argument(
        "--video",
        type=Path,
        default=_suppress,
        help="Write MP4 to this path (legacy). Also writes <stem>.metrics.txt, <stem>_joints.csv.",
    )
    p.add_argument(
        "--joint-log-interval",
        type=int,
        default=_suppress,
        dest="joint_log_interval",
        help="Log joint state every N simulation samples (post-reset is step 0; after each ctrl step +1). "
        "<=0 disables joints.csv.",
    )
    p.add_argument("--video-cell-h", type=int, default=_suppress, dest="video_cell_h")
    p.add_argument("--video-cell-w", type=int, default=_suppress, dest="video_cell_w")
    p.add_argument(
        "--video-separator-frames",
        type=int,
        default=_suppress,
        dest="video_separator_frames",
        help="Black frames inserted between episodes in the video.",
    )
    p.add_argument(
        "--video-fps",
        type=float,
        default=_suppress,
        dest="video_fps",
        help="Output FPS (default: 1 / env.control_dt from YAML or null).",
    )
    p.add_argument(
        "--no-overview-traces",
        action="store_true",
        default=_suppress,
        dest="no_overview_traces",
        help="Disable thin kinematic-chain overlays on the overview RGB and depth tiles.",
    )
    ns = p.parse_args(argv)

    cfg_path = ns.config or (default_cfg if default_cfg.is_file() else None)
    if cfg_path is not None:
        settings = load_simulate_settings(cfg_path)
        if ns.policy_file is not None:
            settings = replace(settings, policy_file=ns.policy_file.resolve())
    else:
        if ns.policy_file is None:
            p.error(
                "policy_file is required when no simulate YAML is found; "
                "add examples/simulate_policy.example.yaml or pass --config PATH"
            )
        settings = builtin_defaults(policy_file=ns.policy_file)

    settings = _merge_cli_overrides(settings, ns)
    return settings, p


def main() -> None:
    _load_dotenv()
    args, _parser = _parse_args()

    video_path, metrics_path, joints_path = _resolve_outputs(
        run_dir=args.run_dir,
        video=args.video,
    )
    log_joints = joints_path is not None and args.joint_log_interval > 0

    policy = load_policy(args.policy_file, args.symbol)
    successes = 0
    writer: Any = None
    video_fps: float | None = None
    episode_final_z: list[float] = []
    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f"policy_file={args.policy_file.resolve()}")
    if video_path is not None:
        log(
            "recording: "
            f"video={video_path}, metrics={metrics_path}, "
            f"joints={joints_path if log_joints else 'disabled'}"
        )

    joints_file: TextIO | None = None
    joints_writer: csv.writer | None = None
    joint_header_written = False

    try:
        for ep in range(args.episodes):
            env = UR5GripperEnv(enable_rgb=args.rgb, seed=1000 + ep)
            obs = env.reset()
            reset_fn = getattr(policy, "reset", None)
            if callable(reset_fn):
                reset_fn()

            if video_path is not None:
                if video_fps is None:
                    video_fps = float(args.video_fps) if args.video_fps is not None else 1.0 / env.control_dt
                overview_traces: dict[str, list[tuple[float, float]]] | None = None
                if not args.no_overview_traces:
                    overview_traces = _new_overview_traces()
                try:
                    if overview_traces is not None:
                        _append_overview_trace_samples(
                            env.model,
                            env.data,
                            overview_traces,
                            args.video_cell_w,
                            args.video_cell_h,
                        )
                    frame0 = _multiview_frame(
                        env,
                        args.video_cell_h,
                        args.video_cell_w,
                        overview_traces=overview_traces,
                    )
                except Exception as exc:  # noqa: BLE001
                    raise SystemExit(
                        "Video capture failed (MuJoCo Renderer needs a GL context). "
                        "On macOS/Linux with a display, run from a desktop session; for headless, "
                        "configure OSMesa/EGL per MuJoCo docs."
                    ) from exc
                if writer is None:
                    writer = _open_video_writer(video_path, video_fps)
                elif ep > 0 and args.video_separator_frames > 0:
                    blank = np.zeros_like(frame0)
                    for _ in range(args.video_separator_frames):
                        writer.append_data(blank)
                writer.append_data(frame0)

                if log_joints:
                    if joints_file is None:
                        joints_file = open(joints_path, "w", newline="", encoding="utf-8")
                        joints_writer = csv.writer(joints_file)
                    if not joint_header_written:
                        joints_writer.writerow(joint_log_header(env.model))
                        joint_header_written = True
                    if 0 % args.joint_log_interval == 0:
                        joints_writer.writerow(
                            joint_log_row(env.model, env.data, ep + 1, 0),
                        )
                        joints_file.flush()

            for k in range(args.steps):
                ctrl = np.asarray(policy(obs, k, env), dtype=np.float64).reshape(-1)
                if ctrl.size != env.nu:
                    raise SystemExit(f"policy returned length {ctrl.size}, need {env.nu}")
                obs = env.step(ctrl)
                if writer is not None:
                    if overview_traces is not None:
                        _append_overview_trace_samples(
                            env.model,
                            env.data,
                            overview_traces,
                            args.video_cell_w,
                            args.video_cell_h,
                        )
                    writer.append_data(
                        _multiview_frame(
                            env,
                            args.video_cell_h,
                            args.video_cell_w,
                            overview_traces=overview_traces,
                        )
                    )
                    if log_joints and joints_writer is not None and joints_file is not None:
                        step_idx = k + 1
                        if step_idx % args.joint_log_interval == 0:
                            joints_writer.writerow(
                                joint_log_row(env.model, env.data, ep + 1, step_idx),
                            )
                            joints_file.flush()

            if env.lift_success(min_height=args.lift_z):
                successes += 1
            fz = env._body_pos_z("grasp_box")
            episode_final_z.append(fz)
            log(f"episode {ep+1}/{args.episodes}: final_box_z={fz:.4f}")

    finally:
        if writer is not None and video_path is not None:
            writer.close()
            fps = float(video_fps) if video_fps is not None else 0.0
            log(f"wrote video {video_path.resolve()} ({fps:.2f} fps)")
        if joints_file is not None:
            joints_file.close()

    log(f"SUCCESS_RATE {successes}/{args.episodes}")

    if args.strict and successes != args.episodes:
        raise SystemExit(1)

    if metrics_path is not None:
        lines = [
            f"SUCCESS_RATE {successes}/{args.episodes}",
            f"episodes {args.episodes}",
            f"steps {args.steps}",
            f"lift_threshold_z {args.lift_z}",
            f"joint_log_interval {args.joint_log_interval}",
        ]
        for i, z in enumerate(episode_final_z, start=1):
            lines.append(f"episode_{i}_final_box_z {z:.6f}")
        metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log(f"wrote metrics {metrics_path.resolve()}")
    if log_joints and joints_path is not None and joints_path.is_file():
        log(f"wrote joints log {joints_path.resolve()} (every {args.joint_log_interval} sim step(s))")


if __name__ == "__main__":
    main()
