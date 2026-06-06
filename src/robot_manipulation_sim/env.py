"""UR5e + Robotiq 2F-85 gripper MuJoCo environment with multi-camera RGB observations."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile
from typing import Any

import mujoco
import numpy as np

from robot_manipulation_sim.cameras import CameraSpec, render_cameras, render_rgb

MJCF_NAME = "ur5e_two_finger_scene.xml"
# Cameras for ``obs["images"]`` when ``enable_rgb``; names must match MJCF. Rollout MP4 (``simulate_policy``)
# uses ``overview`` + ``wrist_rgb`` (row 0) and ``front_rgb`` + top-down (row 1) — see ``render_rollout_four_view_grid``.
DEFAULT_CAMERAS: tuple[CameraSpec, ...] = (
    CameraSpec("overview", 640, 480),
    CameraSpec("wrist_rgb", 480, 360),
)


def default_mjcf_path() -> Path:
    """Path to bundled scene MJCF (next to this package)."""
    return Path(__file__).resolve().parent / "mjcf" / MJCF_NAME


def default_scene_files() -> tuple[Path, ...]:
    """Default composed scene: base descriptor + orange box object."""
    mjcf_dir = Path(__file__).resolve().parent / "mjcf"
    return (
        mjcf_dir / "ur5e_two_finger_scene.xml",
        mjcf_dir / "scene_objects" / "orange_box.xml",
    )


@dataclass
class UR5GripperEnv:
    """MuJoCo scene with UR5e arm, Robotiq 2F-85 adaptive gripper (tendon drive, ctrl 0–255), and RGB cameras."""

    mjcf_path: Path = field(default_factory=default_mjcf_path)
    control_dt: float = 0.02
    cameras: tuple[CameraSpec, ...] = DEFAULT_CAMERAS
    seed: int | None = None
    enable_rgb: bool = True
    scene_files: tuple[Path, ...] | None = field(default_factory=default_scene_files)

    def _composed_scene_path(self) -> Path:
        files = tuple(Path(p).resolve() for p in self.scene_files)
        base = files[0]
        if not base.is_file():
            raise FileNotFoundError(f"base scene file not found: {base}")
        base_xml = base.read_text(encoding="utf-8")
        if len(files) == 1:
            return base
        marker = "<!-- SCENE_IMPORTS -->"
        if marker not in base_xml:
            raise ValueError(f"base scene missing marker {marker!r}: {base}")
        include_lines = "\n".join(f'    <include file="{p.resolve()}"/>' for p in files[1:])
        composed_xml = base_xml.replace(marker, include_lines)
        fd, tmp_name = tempfile.mkstemp(
            prefix="_composed_scene_",
            suffix=".xml",
            dir=base.parent,
            text=True,
        )
        tmp_path = Path(tmp_name)
        # ``mkstemp`` returns a low-level fd; write and close it explicitly.
        with os.fdopen(fd, "w", encoding="utf-8", closefd=True) as f:
            f.write(composed_xml)
        return tmp_path

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        if self.scene_files:
            composed_path = self._composed_scene_path()
            try:
                self.model = mujoco.MjModel.from_xml_path(str(composed_path))
            finally:
                if composed_path.name.startswith("_composed_scene_"):
                    composed_path.unlink(missing_ok=True)
        else:
            self.model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self.data = mujoco.MjData(self.model)
        self._substeps = max(1, int(round(self.control_dt / self.model.opt.timestep)))
        self.nu = int(self.model.nu)
        # Last value: ``a_gripper`` (0–255). Settled finger geometry vs ``ctrl`` is non-obvious;
        # we use ``0`` so reset matches policies that treat low ``ctrl`` as open — see
        # ``tests/test_gripper_control_finger_geometry.py``.
        self._home = np.array(
            [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0, 0.0],
            dtype=np.float64,
        )

    def reset(self, *, box_xy_noise: float = 0.04) -> dict[str, Any]:
        mujoco.mj_resetData(self.model, self.data)
        noise = self._rng.uniform(-box_xy_noise, box_xy_noise, size=2)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "grasp_box")
        if bid < 0:
            raise RuntimeError("MJCF missing body 'grasp_box'")
        jid = self.model.body_jntadr[bid]
        qadr = int(self.model.jnt_qposadr[jid])
        # free joint: x y z quat (w x y z)
        self.data.qpos[qadr : qadr + 3] = np.array([0.52 + noise[0], 0.0 + noise[1], 0.035])
        self.data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.ctrl[:] = self._home[: self.nu]
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def set_control(self, ctrl: np.ndarray) -> None:
        """Set actuator targets (length must equal nu)."""
        ctrl = np.asarray(ctrl, dtype=np.float64).reshape(-1)
        if ctrl.shape[0] != self.nu:
            raise ValueError(f"ctrl has length {ctrl.shape[0]}, expected {self.nu}")
        self.data.ctrl[:] = ctrl

    def step(self, ctrl: np.ndarray | None = None) -> dict[str, Any]:
        if ctrl is not None:
            self.set_control(ctrl)
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)
        return self.get_observation()

    def get_observation(self) -> dict[str, Any]:
        if self.enable_rgb:
            try:
                imgs = render_cameras(self.model, self.data, self.cameras)
            except Exception as exc:  # noqa: BLE001 — GL backends vary by platform
                raise RuntimeError(
                    "RGB rendering failed (no GL context). Set UR5GripperEnv(enable_rgb=False) "
                    "for state-only observations, or configure a MuJoCo GL backend (e.g. "
                    "MUJOCO_GL=glfw on desktop)."
                ) from exc
        else:
            imgs = {}
        box_height = float(self._body_pos_z("grasp_box"))
        return {
            "images": imgs,
            "qpos": np.array(self.data.qpos, copy=True),
            "qvel": np.array(self.data.qvel, copy=True),
            "ctrl": np.array(self.data.ctrl, copy=True),
            "box_height": box_height,
            "time": float(self.data.time),
        }

    def render_camera(self, name: str, width: int = 640, height: int = 480) -> np.ndarray:
        return render_rgb(self.model, self.data, name, width, height)

    def _body_pos_z(self, body_name: str) -> float:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return float(self.data.xpos[bid, 2])

    def lift_success(self, min_height: float = 0.12) -> bool:
        """Heuristic success: grasp box center of mass above table threshold."""
        return self._body_pos_z("grasp_box") >= min_height


def map_normalized_actions(ctrl_normalized: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    """Map [-1, 1]^nu to actuator ctrlrange centers (handy for RL / scripted policies)."""
    ctrl_normalized = np.clip(np.asarray(ctrl_normalized, dtype=np.float64), -1.0, 1.0)
    out = np.zeros(model.nu, dtype=np.float64)
    for i in range(model.nu):
        lo, hi = model.actuator_ctrlrange[i]
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        out[i] = mid + half * float(ctrl_normalized[i])
    return out
