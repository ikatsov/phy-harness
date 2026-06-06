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

try:
    import imageio.v2 as imageio
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency 'imageio' (bundled with this repo via pyproject.toml). "
        "From the repo root, install into the interpreter you use to run this script, e.g.:\n"
        "  python -m pip install -e .\n"
        "If you use a venv, prefer:  .venv/bin/python scripts/simulate_policy.py …\n"
        "so you do not pick up another Python (e.g. conda base) without those packages."
    ) from e
import mujoco
import numpy as np

from robot_manipulation_sim.cameras import (
    clear_renderer_cache,
    project_world_positions_to_camera_pixels,
    render_rollout_four_view_grid,
)
from robot_manipulation_sim.env import UR5GripperEnv
from robot_manipulation_sim.simulate_config import (
    AnalyzerSettings,
    SimulateSettings,
    builtin_defaults,
    load_simulate_settings,
)
from robot_manipulation_sim.validation.analyzers.base import AnalyzerResult
from robot_manipulation_sim.validation.analyzers.vlm_video_transcriber import VlmVideoTranscriberAnalyzer
from robot_manipulation_sim.validation.context import SimulationArtifacts, ValidationContext


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
    return _repo_root() / "policies" / "simulate_policy.example.yaml"


class PolicyFn(Protocol):
    def __call__(self, obs: dict[str, Any], step: int, env: UR5GripperEnv) -> np.ndarray: ...


def load_policy(path: Path, policy_symbol: str = "policy") -> PolicyFn:
    spec = importlib.util.spec_from_file_location("user_policy_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_policy_module"] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, policy_symbol, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"{path} must define callable {policy_symbol}(obs, step, env) -> ctrl")
    return fn  # type: ignore[return-value]


def _load_policy_analyzer(task: str, analyzer_type: str, params: dict[str, Any]) -> Any:
    path = (_repo_root() / "policies" / "impl" / task / f"{analyzer_type}.py").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"policy analyzer not found: {path}")
    mod_name = f"_simulate_policy_analyzer_{task}_{analyzer_type}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load analyzer module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build_fn = getattr(module, "build", None)
    if not callable(build_fn):
        raise TypeError(f"{path} must define build(params: dict | None) -> analyzer")
    return build_fn(params)


def _make_analyzer(spec: AnalyzerSettings, task: str | None) -> Any:
    key = spec.type.strip()
    params = dict(spec.params or {})
    if key == "vlm_video_transcriber":
        return VlmVideoTranscriberAnalyzer(params)
    if task:
        return _load_policy_analyzer(task, key, params)
    raise KeyError(f"unknown analyzer type {key!r} (set task or use vlm_video_transcriber)")


def _run_analyzers(
    *,
    analyzers: list[AnalyzerSettings] | None,
    task: str | None,
    video_path: Path | None,
    metrics_path: Path | None,
    joints_path: Path | None,
    log: Any,
) -> list[AnalyzerResult]:
    if not analyzers:
        return []
    if video_path is None:
        log("analyzers: skipped (no video output configured)")
        return []

    metrics_text = None
    if metrics_path is not None and metrics_path.is_file():
        metrics_text = metrics_path.read_text(encoding="utf-8")
    ctx = ValidationContext(
        simulation=SimulationArtifacts(
            video=video_path.resolve(),
            metrics_file=metrics_path.resolve() if metrics_path is not None else None,
            joints_csv=joints_path.resolve() if joints_path is not None else None,
        ),
        metrics_text=metrics_text,
        task_spec=None,
        config_path=None,
    )
    results: list[AnalyzerResult] = []
    for spec in analyzers:
        if not spec.enabled:
            continue
        try:
            analyzer = _make_analyzer(spec, task)
            result: AnalyzerResult = analyzer.analyze(ctx)
        except Exception as exc:  # noqa: BLE001
            result = AnalyzerResult(
                analyzer_type=spec.type,
                ok=False,
                exit_code=2,
                messages=[f"{type(exc).__name__}: {exc}"],
            )
        results.append(result)
        status = "ok" if result.ok else "fail"
        log(f"analyzer {spec.type}: {status} (exit_code={result.exit_code})")
        for m in result.messages:
            log(f"  {m}")
    return results


# Cameras that receive the same kinematic-chain polylines (world COM → pixel paths).
_KINEMATIC_TRACE_CAMERAS: tuple[str, ...] = ("overview", "front_rgb", "topdown")

# Trace markers projected onto rollout RGB tiles.
# Each tuple: (trace_key, kind, mjcf_name, color_rgb). ``kind`` is "body" or "geom".
#
# For gripper fingertips we trace pad geom centers (left_pad2/right_pad2), not pad body origins.
# The body origins sit at the pad frame origin and can look visually detached from contact surfaces.
_KINEMATIC_TRACE_MARKERS: tuple[tuple[str, str, str, tuple[int, int, int]], ...] = (
    ("shoulder_link", "body", "shoulder_link", (255, 80, 80)),
    ("upper_arm_link", "body", "upper_arm_link", (255, 200, 80)),
    ("forearm_link", "body", "forearm_link", (120, 255, 120)),
    ("wrist_1_link", "body", "wrist_1_link", (80, 200, 255)),
    ("wrist_2_link", "body", "wrist_2_link", (180, 120, 255)),
    ("wrist_3_link", "body", "wrist_3_link", (255, 120, 200)),
    ("tool0", "body", "tool0", (240, 240, 255)),
    ("left_finger_tip", "geom", "left_pad2", (255, 240, 0)),
    ("right_finger_tip", "geom", "right_pad2", (0, 230, 220)),
)


def _new_kinematic_traces() -> dict[str, dict[str, list[tuple[float, float]]]]:
    """Per-camera history of body COM pixel paths (same body set for each camera)."""
    return {cam: {k: [] for k, _, _, _ in _KINEMATIC_TRACE_MARKERS} for cam in _KINEMATIC_TRACE_CAMERAS}


def _append_kinematic_trace_samples(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    traces: dict[str, dict[str, list[tuple[float, float]]]],
    cell_w: int,
    cell_h: int,
) -> None:
    for cam in _KINEMATIC_TRACE_CAMERAS:
        cam_tr = traces[cam]
        for key, kind, mj_name, _ in _KINEMATIC_TRACE_MARKERS:
            if kind == "body":
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mj_name)
                if bid < 0:
                    continue
                xyz = np.asarray(data.xpos[bid], dtype=np.float64).reshape(1, 3)
            elif kind == "geom":
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, mj_name)
                if gid < 0:
                    continue
                xyz = np.asarray(data.geom_xpos[gid], dtype=np.float64).reshape(1, 3)
            else:
                continue
            uv = project_world_positions_to_camera_pixels(
                model, data, cam, int(cell_w), int(cell_h), xyz
            )[0]
            if np.all(np.isfinite(uv)):
                cam_tr.setdefault(key, []).append((float(uv[0]), float(uv[1])))


def _body_traces_to_draw_args(
    body_traces: dict[str, list[tuple[float, float]]],
) -> tuple[tuple[np.ndarray, tuple[int, int, int]], ...] | None:
    out: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    for key, _, _, color in _KINEMATIC_TRACE_MARKERS:
        seq = body_traces.get(key, [])
        if len(seq) < 2:
            continue
        out.append((np.asarray(seq, dtype=np.float64), color))
    return tuple(out) if out else None


def _multiview_frame(
    env: UR5GripperEnv,
    cell_h: int,
    cell_w: int,
    *,
    kinematic_traces: dict[str, dict[str, list[tuple[float, float]]]] | None = None,
) -> np.ndarray:
    """2×2 grid: row0 = perspective (``overview``) RGB | wrist RGB; row1 = front isometric | top-down."""
    ov = (
        _body_traces_to_draw_args(kinematic_traces["overview"])
        if kinematic_traces is not None
        else None
    )
    fr = (
        _body_traces_to_draw_args(kinematic_traces["front_rgb"])
        if kinematic_traces is not None
        else None
    )
    td = (
        _body_traces_to_draw_args(kinematic_traces["topdown"])
        if kinematic_traces is not None
        else None
    )
    return render_rollout_four_view_grid(
        env.model,
        env.data,
        cell_h,
        cell_w,
        perspective_camera="overview",
        wrist_camera="wrist_rgb",
        front_camera="front_rgb",
        side_camera="topdown",
        perspective_traces=ov,
        front_traces=fr,
        side_traces=td,
    )


def _open_video_writer(path: Path, fps: float) -> Any:
    return imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=None,
    )


# Decimal places for all floating-point columns in ``joints.csv`` (joints, time, targets, applied ctrl).
JOINT_LOG_DECIMALS: int = 5


def _joint_log_round(x: float) -> float:
    return round(float(x), JOINT_LOG_DECIMALS)


def joint_log_header(model: mujoco.MjModel) -> list[str]:
    """CSV columns: metadata, joint ``qpos``, then per-actuator ``target_*`` and ``ctrl_*``.

    ``target_<actuator>`` is the control vector **returned by the policy** for that interval (passed to
    ``env.step``). ``ctrl_<actuator>`` is ``data.ctrl`` at the log sample time after physics (applied
    command in MuJoCo state; should match ``target_*`` for this harness).
    """
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
        cols.append(f"target_{an}")
        cols.append(f"ctrl_{an}")
    return cols


def joint_log_row(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    episode: int,
    sim_step: int,
    *,
    target_ctrl: np.ndarray | None = None,
) -> list[Any]:
    """One CSV data row; numeric floats are rounded to :data:`JOINT_LOG_DECIMALS`.

    ``target_ctrl`` must have length ``model.nu`` (policy command for this row). If omitted, uses
    ``data.ctrl`` for both target and applied columns (e.g. post-reset initial sample).
    """
    tc = np.asarray(data.ctrl, dtype=np.float64).reshape(-1).copy() if target_ctrl is None else np.asarray(
        target_ctrl, dtype=np.float64
    ).reshape(-1)
    if tc.shape[0] != int(model.nu):
        raise ValueError(f"target_ctrl length {tc.shape[0]} != model.nu {model.nu}")

    row: list[Any] = [int(episode), int(sim_step), _joint_log_round(float(data.time))]
    for j in range(model.njnt):
        adr = int(model.jnt_qposadr[j])
        jt = int(model.jnt_type[j])
        if jt == mujoco.mjtJoint.mjJNT_FREE:
            row.extend(_joint_log_round(float(x)) for x in data.qpos[adr : adr + 7])
        elif jt == mujoco.mjtJoint.mjJNT_BALL:
            row.extend(_joint_log_round(float(x)) for x in data.qpos[adr : adr + 4])
        else:
            row.append(_joint_log_round(float(data.qpos[adr])))
    for a in range(int(model.nu)):
        row.append(_joint_log_round(float(tc[a])))
        row.append(_joint_log_round(float(data.ctrl[a])))
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
            "Defaults come from policies/simulate_policy.example.yaml when present, or pass --config PATH. "
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
    p.add_argument("--policy-symbol", default=_suppress, dest="policy_symbol", help="Callable name in policy_file")
    p.add_argument("--steps", type=int, default=_suppress)
    p.add_argument(
        "--include-rgb-observation",
        action="store_true",
        default=_suppress,
        dest="include_rgb_observation",
        help="Include camera images in policy observations (needs GL).",
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
        "--joint-log-every-steps",
        type=int,
        default=_suppress,
        dest="joint_log_every_steps",
        help="Log joint state every N simulation samples (post-reset is step 0; after each ctrl step +1). "
        "<=0 disables joints.csv.",
    )
    p.add_argument("--video-tile-height", type=int, default=_suppress, dest="video_tile_height")
    p.add_argument("--video-tile-width", type=int, default=_suppress, dest="video_tile_width")
    p.add_argument(
        "--video-output-fps",
        type=float,
        default=_suppress,
        dest="video_output_fps",
        help="Output FPS (default: 1 / env.control_dt from YAML or null).",
    )
    p.add_argument(
        "--disable-kinematic-overlays",
        action="store_true",
        default=_suppress,
        dest="disable_kinematic_overlays",
        help="Disable thin kinematic-chain overlays on overview, front, and top-down RGB tiles.",
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
                "add policies/simulate_policy.example.yaml or pass --config PATH"
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
    log_joints = joints_path is not None and args.joint_log_every_steps > 0

    policy = load_policy(args.policy_file, args.policy_symbol)
    writer: Any = None
    video_fps: float | None = None
    final_box_z = 0.0
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
        env = UR5GripperEnv(enable_rgb=args.include_rgb_observation, seed=1000)
        obs = env.reset()
        reset_fn = getattr(policy, "reset", None)
        if callable(reset_fn):
            reset_fn()

        if video_path is not None:
            if video_fps is None:
                video_fps = float(args.video_output_fps) if args.video_output_fps is not None else 1.0 / env.control_dt
            kinematic_traces: dict[str, dict[str, list[tuple[float, float]]]] | None = None
            if not args.disable_kinematic_overlays:
                kinematic_traces = _new_kinematic_traces()
            try:
                if kinematic_traces is not None:
                    _append_kinematic_trace_samples(
                        env.model,
                        env.data,
                        kinematic_traces,
                        args.video_tile_width,
                        args.video_tile_height,
                    )
                frame0 = _multiview_frame(
                    env,
                    args.video_tile_height,
                    args.video_tile_width,
                    kinematic_traces=kinematic_traces,
                )
            except Exception as exc:  # noqa: BLE001
                raise SystemExit(
                    "Video capture failed (MuJoCo Renderer needs a GL context). "
                    "On macOS/Linux with a display, run from a desktop session; for headless, "
                    "configure OSMesa/EGL per MuJoCo docs."
                ) from exc
            if writer is None:
                writer = _open_video_writer(video_path, video_fps)
            writer.append_data(frame0)

            if log_joints:
                if joints_file is None:
                    joints_file = open(joints_path, "w", newline="", encoding="utf-8")
                    joints_writer = csv.writer(joints_file)
                if not joint_header_written:
                    joints_writer.writerow(joint_log_header(env.model))
                    joint_header_written = True
                if 0 % args.joint_log_every_steps == 0:
                    joints_writer.writerow(
                        joint_log_row(
                            env.model,
                            env.data,
                            1,
                            0,
                            target_ctrl=np.asarray(obs["ctrl"], dtype=np.float64),
                        ),
                    )
                    joints_file.flush()

        for k in range(args.steps):
            ctrl = np.asarray(policy(obs, k, env), dtype=np.float64).reshape(-1)
            if ctrl.size != env.nu:
                raise SystemExit(f"policy returned length {ctrl.size}, need {env.nu}")
            obs = env.step(ctrl)
            if writer is not None:
                if kinematic_traces is not None:
                    _append_kinematic_trace_samples(
                        env.model,
                        env.data,
                        kinematic_traces,
                        args.video_tile_width,
                        args.video_tile_height,
                    )
                writer.append_data(
                    _multiview_frame(
                        env,
                        args.video_tile_height,
                        args.video_tile_width,
                        kinematic_traces=kinematic_traces,
                    )
                )
                if log_joints and joints_writer is not None and joints_file is not None:
                    step_idx = k + 1
                    if step_idx % args.joint_log_every_steps == 0:
                        joints_writer.writerow(
                            joint_log_row(
                                env.model,
                                env.data,
                                1,
                                step_idx,
                                target_ctrl=ctrl,
                            ),
                        )
                        joints_file.flush()

        final_box_z = env._body_pos_z("grasp_box")
        log(f"final_box_z={final_box_z:.4f}")
        clear_renderer_cache(env.model)

    finally:
        if writer is not None and video_path is not None:
            writer.close()
            fps = float(video_fps) if video_fps is not None else 0.0
            log(f"wrote video {video_path.resolve()} ({fps:.2f} fps)")
        if joints_file is not None:
            joints_file.close()

    if metrics_path is not None:
        lines = [
            f"steps {args.steps}",
            f"joint_log_every_steps {args.joint_log_every_steps}",
            f"final_box_z {final_box_z:.6f}",
        ]
        metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log(f"wrote metrics {metrics_path.resolve()}")
    if log_joints and joints_path is not None and joints_path.is_file():
        log(f"wrote joints log {joints_path.resolve()} (every {args.joint_log_every_steps} sim step(s))")

    analyzer_results = _run_analyzers(
        analyzers=args.analyzers,
        task=args.task,
        video_path=video_path,
        metrics_path=metrics_path,
        joints_path=joints_path if log_joints else None,
        log=log,
    )

    analyzer_failures = [r for r in analyzer_results if r.exit_code != 0]
    if analyzer_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
