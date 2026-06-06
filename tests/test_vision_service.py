"""Tests for the VisionService: colour detection and pixel→3D back-projection.

These tests do **not** require a GL context (no rendering) — they use synthesised
images and pre-known camera parameters to verify detection and projection math.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from robot_manipulation_sim.env import default_mjcf_path
from robot_manipulation_sim.vision.service import (
    Detection,
    VisionService,
    _rgb_to_hsv_opencv_scale,
    unproject_pixel_to_world,
)
from robot_manipulation_sim.cameras import project_world_positions_to_camera_pixels


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_data():
    m = mujoco.MjModel.from_xml_path(str(default_mjcf_path()))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    return m, d


# ---------------------------------------------------------------------------
# 1. RGB→HSV helper
# ---------------------------------------------------------------------------

class TestRgbToHsv:
    def test_pure_red(self):
        img = np.array([[[255, 0, 0]]], dtype=np.float32)
        hsv = _rgb_to_hsv_opencv_scale(img)
        h, s, v = hsv[0, 0]
        assert abs(h - 0.0) < 1.0 or abs(h - 180.0) < 1.0, f"H for red: {h}"
        assert s > 200, f"S for saturated red: {s}"
        assert v > 200

    def test_pure_orange(self):
        """MJCF box colour rgba(0.85, 0.55, 0.2) ≈ RGB(217, 140, 51)."""
        img = np.array([[[217, 140, 51]]], dtype=np.float32)
        hsv = _rgb_to_hsv_opencv_scale(img)
        h, s, v = hsv[0, 0]
        # Orange: H in [10, 30] (OpenCV scale), high S and V.
        assert 8 <= h <= 35, f"Hue for orange: {h:.1f}"
        assert s > 150, f"S: {s:.1f}"
        assert v > 150, f"V: {v:.1f}"

    def test_pure_green(self):
        img = np.array([[[0, 255, 0]]], dtype=np.float32)
        hsv = _rgb_to_hsv_opencv_scale(img)
        h, s, v = hsv[0, 0]
        assert 55 <= h <= 65, f"H for green: {h}"

    def test_white_gives_zero_saturation(self):
        img = np.array([[[255, 255, 255]]], dtype=np.float32)
        hsv = _rgb_to_hsv_opencv_scale(img)
        assert hsv[0, 0, 1] < 1.0, "S for white must be 0"

    def test_black_gives_zero_value(self):
        img = np.array([[[0, 0, 0]]], dtype=np.float32)
        hsv = _rgb_to_hsv_opencv_scale(img)
        assert hsv[0, 0, 2] < 1.0, "V for black must be 0"


# ---------------------------------------------------------------------------
# 2. Colour detection
# ---------------------------------------------------------------------------

class TestColorDetection:
    @pytest.fixture(autouse=True)
    def setup(self, model_data):
        m, d = model_data
        self.svc = VisionService(m, camera_name="topdown", width=64, height=64)

    def _make_image_with_orange_square(self, x1=20, y1=20, x2=44, y2=44) -> np.ndarray:
        """Synthetic 64×64 RGB image: grey background, orange square at (x1,y1)-(x2,y2)."""
        img = np.full((64, 64, 3), 80, dtype=np.uint8)
        img[y1:y2, x1:x2, 0] = 217   # R
        img[y1:y2, x1:x2, 1] = 140   # G
        img[y1:y2, x1:x2, 2] = 51    # B
        return img

    def test_detects_orange_square(self):
        img = self._make_image_with_orange_square()
        # HSV orange lower/upper (OpenCV scale).
        lo = np.array([8, 100, 100])
        hi = np.array([35, 255, 255])
        dets = self.svc.detect_by_color(img, lo, hi)
        assert len(dets) >= 1, "Expected at least one detection"
        d0 = dets[0]
        assert d0.label == "color"
        # Centre should be within the orange square.
        cx, cy = d0.center_uv
        assert 20 <= cx <= 44, f"center x {cx} outside [20,44]"
        assert 20 <= cy <= 44, f"center y {cy} outside [20,44]"

    def test_no_detection_on_grey_image(self):
        img = np.full((64, 64, 3), 80, dtype=np.uint8)
        lo = np.array([8, 100, 100])
        hi = np.array([35, 255, 255])
        dets = self.svc.detect_by_color(img, lo, hi)
        assert len(dets) == 0

    def test_detection_bbox_correct(self):
        """Bounding box should tightly wrap the orange region."""
        img = self._make_image_with_orange_square(x1=10, y1=15, x2=30, y2=50)
        lo, hi = np.array([8, 100, 100]), np.array([35, 255, 255])
        dets = self.svc.detect_by_color(img, lo, hi)
        assert len(dets) >= 1
        x1, y1, x2, y2 = dets[0].bbox_xyxy
        assert abs(x1 - 10) <= 1 and abs(y1 - 15) <= 1
        assert abs(x2 - 29) <= 1 and abs(y2 - 49) <= 1

    def test_min_area_filters_small_blobs(self):
        """Regions smaller than min_area should be dropped."""
        img = np.full((64, 64, 3), 80, dtype=np.uint8)
        # 4 orange pixels — below default min_area=50.
        img[30:32, 30:32, 0] = 217
        img[30:32, 30:32, 1] = 140
        img[30:32, 30:32, 2] = 51
        lo, hi = np.array([8, 100, 100]), np.array([35, 255, 255])
        dets = self.svc.detect_by_color(img, lo, hi, min_area=50)
        assert len(dets) == 0

    def test_detection_area_matches_region(self):
        img = self._make_image_with_orange_square(x1=5, y1=5, x2=25, y2=25)
        lo, hi = np.array([8, 100, 100]), np.array([35, 255, 255])
        dets = self.svc.detect_by_color(img, lo, hi)
        assert len(dets) >= 1
        # Area should be approximately 20×20 = 400.
        assert dets[0].area >= 300, f"area {dets[0].area}"

    def test_two_blobs_both_detected(self):
        img = np.full((64, 64, 3), 80, dtype=np.uint8)
        # Left blob.
        img[5:20, 2:18, :] = [217, 140, 51]
        # Right blob.
        img[5:20, 40:56, :] = [217, 140, 51]
        lo, hi = np.array([8, 100, 100]), np.array([35, 255, 255])
        dets = self.svc.detect_by_color(img, lo, hi)
        assert len(dets) == 2, f"Expected 2 blobs, got {len(dets)}"


# ---------------------------------------------------------------------------
# 3. Back-projection: pixel → 3D world
# ---------------------------------------------------------------------------

class TestBackProjection:
    """Round-trip test: forward-project a known 3D point → pixel → back-project → 3D."""

    # World points to test back-projection with each camera.
    _TEST_POINTS = [
        np.array([0.52, 0.00, 0.035]),    # nominal box position
        np.array([0.30, 0.20, 0.035]),
        np.array([0.00, 0.00, 0.035]),    # base of robot
        np.array([0.40, -0.20, 0.10]),
    ]

    @pytest.fixture(params=["topdown", "front_rgb", "overview"])
    def camera_setup(self, request, model_data):
        cam_name = request.param
        m, d = model_data
        w, h = 320, 240
        return m, d, cam_name, w, h

    def test_roundtrip_world_to_pixel_to_world(self, camera_setup):
        """Forward-project a 3D point, back-project the pixel, compare to original."""
        m, d, cam_name, w, h = camera_setup
        svc = VisionService(m, camera_name=cam_name, width=w, height=h)

        for pt in self._TEST_POINTS:
            # Forward: 3D → pixel.
            uv_arr = project_world_positions_to_camera_pixels(m, d, cam_name, w, h, pt.reshape(1, 3))
            uv = uv_arr[0]
            if not np.all(np.isfinite(uv)):
                continue  # Point outside frustum for this camera — skip.

            # Backward: pixel → 3D at known z.
            target_z = float(pt[2])
            pt_recovered = svc.unproject_pixel_to_world(d, uv, target_z=target_z)

            np.testing.assert_allclose(
                pt_recovered, pt, atol=1e-4,
                err_msg=f"camera={cam_name}, pt={pt}, uv={uv}, recovered={pt_recovered}"
            )

    def test_standalone_unproject_matches_service(self, model_data):
        m, d = model_data
        svc = VisionService(m, camera_name="topdown", width=320, height=240)
        pt = np.array([0.52, 0.0, 0.035])
        uv = project_world_positions_to_camera_pixels(m, d, "topdown", 320, 240, pt.reshape(1, 3))[0]
        if not np.all(np.isfinite(uv)):
            pytest.skip("Point not visible from topdown camera")

        via_service = svc.unproject_pixel_to_world(d, uv, target_z=0.035)
        standalone = unproject_pixel_to_world(m, d, "topdown", 320, 240, uv, target_z=0.035)
        np.testing.assert_allclose(via_service, standalone, atol=1e-12)

    def test_unproject_z_equals_target(self, model_data):
        """Back-projected point always has the exact requested Z."""
        m, d = model_data
        svc = VisionService(m, camera_name="topdown", width=320, height=240)
        for target_z in [0.0, 0.035, 0.2, 0.5]:
            pt = svc.unproject_pixel_to_world(d, np.array([160.0, 120.0]), target_z=target_z)
            assert abs(pt[2] - target_z) < 1e-9, f"z mismatch at target_z={target_z}"

    def test_detection_center_consistent_with_unproject(self, model_data):
        """Centre of a detected colour region round-trips through back-projection."""
        m, d = model_data
        svc = VisionService(m, camera_name="topdown", width=64, height=64)

        # Place an orange square at a known pixel position.
        img = np.full((64, 64, 3), 80, dtype=np.uint8)
        # Put orange square pixels at pixel rows 20-40, cols 20-40.
        img[20:40, 20:40, 0] = 217
        img[20:40, 20:40, 1] = 140
        img[20:40, 20:40, 2] = 51
        lo, hi = np.array([8, 100, 100]), np.array([35, 255, 255])
        dets = svc.detect_by_color(img, lo, hi)
        assert len(dets) >= 1

        # Centre should be near pixel (30, 30).
        cx, cy = dets[0].center_uv
        assert abs(cx - 30) <= 2
        assert abs(cy - 30) <= 2

        # Back-project to world at z=0.035.
        pt = svc.unproject_pixel_to_world(d, dets[0].center_uv, target_z=0.035)
        assert abs(pt[2] - 0.035) < 1e-9


# ---------------------------------------------------------------------------
# 4. VisionService construction
# ---------------------------------------------------------------------------

class TestVisionServiceConstruction:
    def test_default_construction(self, model_data):
        m, d = model_data
        svc = VisionService(m)
        assert svc.camera_name == "topdown"
        assert svc.width == 640
        assert svc.height == 480

    def test_unknown_camera_raises(self, model_data):
        m, d = model_data
        with pytest.raises(KeyError, match="nonexistent_cam"):
            VisionService(m, camera_name="nonexistent_cam")

    def test_detection_result_props(self):
        bbox = np.array([10.0, 20.0, 50.0, 60.0])
        det = Detection(label="test", confidence=0.9, bbox_xyxy=bbox)
        np.testing.assert_allclose(det.center_uv, [30.0, 40.0])
        assert abs(det.area - 40.0 * 40.0) < 0.1
