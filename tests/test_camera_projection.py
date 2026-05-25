"""Overview camera projection for rollout overlays (no GL)."""

from __future__ import annotations

import mujoco
import numpy as np

from robot_manipulation_sim.cameras import draw_polylines_on_tile, project_world_positions_to_camera_pixels
from robot_manipulation_sim.env import default_mjcf_path


def test_project_shoulder_link_inside_overview_tile() -> None:
    m = mujoco.MjModel.from_xml_path(str(default_mjcf_path()))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "shoulder_link")
    xyz = np.asarray(d.xpos[bid], dtype=np.float64).reshape(1, 3)
    w, h = 426, 360
    uv = project_world_positions_to_camera_pixels(m, d, "overview", w, h, xyz)[0]
    assert np.all(np.isfinite(uv))
    assert 0 <= uv[0] < w and 0 <= uv[1] < h


def test_draw_polylines_mutates_uint8_tile() -> None:
    tile = np.zeros((40, 50, 3), dtype=np.uint8)
    poly = np.array([[5.0, 10.0], [45.0, 30.0]], dtype=np.float64)
    draw_polylines_on_tile(tile, [poly], [(255, 0, 0)])
    assert int(tile[..., 0].max()) >= 200
