"""Unit tests for depth visualization (no GL)."""

from __future__ import annotations

import mujoco
import numpy as np

from robot_manipulation_sim.cameras import depth_to_grayscale_rgb, depth_to_rgb
from robot_manipulation_sim.env import default_mjcf_path


def _tiny_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(default_mjcf_path()))


def test_depth_to_grayscale_shape_dtype_and_channels():
    m = _tiny_model()
    d = (np.random.default_rng(0).random((24, 32)).astype(np.float32) * 1.5 + 0.1).astype(np.float32)
    rgb = depth_to_grayscale_rgb(d, m)
    assert rgb.shape == (24, 32, 3)
    assert rgb.dtype == np.uint8
    assert np.all(rgb[:, :, 0] == rgb[:, :, 1])
    assert np.all(rgb[:, :, 1] == rgb[:, :, 2])


def test_depth_to_rgb_alias_matches_grayscale():
    m = _tiny_model()
    d = (np.random.default_rng(1).random((16, 20)).astype(np.float32) + 0.2).astype(np.float32)
    assert np.array_equal(depth_to_rgb(d, m), depth_to_grayscale_rgb(d, m))


def test_depth_to_grayscale_all_invalid_returns_black():
    d = np.full((5, 7), np.nan, dtype=np.float32)
    rgb = depth_to_grayscale_rgb(d, _tiny_model())
    assert np.all(rgb == 0)


def test_depth_to_grayscale_uses_dynamic_range():
    """Regression: output should not collapse to a single gray level for a spread depth map."""
    m = _tiny_model()
    rng = np.random.default_rng(2)
    d = (rng.random((64, 64), dtype=np.float32) * 0.8 + 0.35).astype(np.float32)
    g = depth_to_grayscale_rgb(d, m)[:, :, 0].astype(np.int32)
    assert int(g.max()) - int(g.min()) >= 40


def test_depth_exp_decay_stronger_near_contrast_than_linear():
    """With exp_decay > 0, small depth deltas near the camera span more gray levels than a linear ramp."""
    m = _tiny_model()
    # Two near depths and two far depths; near pair should be more separated in output than far pair.
    d = np.array(
        [
            [0.5, 0.52, 1.5, 1.55],
            [0.5, 0.52, 1.5, 1.55],
        ],
        dtype=np.float32,
    )
    g_exp = depth_to_grayscale_rgb(d, m, exp_decay=3.2)[:, :, 0].astype(np.int32)
    g_lin = depth_to_grayscale_rgb(d, m, exp_decay=1e-9)[:, :, 0].astype(np.int32)
    near_sep_exp = int(abs(g_exp[0, 0] - g_exp[0, 1]))
    far_sep_exp = int(abs(g_exp[0, 2] - g_exp[0, 3]))
    near_sep_lin = int(abs(g_lin[0, 0] - g_lin[0, 1]))
    far_sep_lin = int(abs(g_lin[0, 2] - g_lin[0, 3]))
    assert near_sep_exp >= near_sep_lin - 1  # allow tie within 1 level
    # Exponential allocates more slope near u=0: near pair should gain more than far pair vs linear.
    assert (near_sep_exp - near_sep_lin) > (far_sep_exp - far_sep_lin)


def test_depth_scene_scale_nearer_brighter_on_ramp():
    m = _tiny_model()
    d = np.array([[1.0, 1.5, 2.0], [2.0, 1.0, 1.5]], dtype=np.float32)
    g = depth_to_grayscale_rgb(d, m)[:, :, 0].astype(np.int32)
    assert g[0, 0] > g[0, 2]  # 1.0 vs 2.0 (nearer brighter)
    assert g[1, 1] > g[1, 0]
