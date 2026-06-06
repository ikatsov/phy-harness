"""Color detection + pixel-to-world back-projection for MuJoCo cameras."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

from robot_manipulation_sim.cameras import camera_id, render_rgb


@dataclass
class Detection:
    """Image-space detection."""
    label: str
    confidence: float
    bbox_xyxy: NDArray[np.float64]

    @property
    def center_uv(self) -> NDArray[np.float64]:
        """Bounding-box center in pixels."""
        return np.array(
            [
                0.5 * (self.bbox_xyxy[0] + self.bbox_xyxy[2]),
                0.5 * (self.bbox_xyxy[1] + self.bbox_xyxy[3]),
            ],
            dtype=np.float64,
        )

    @property
    def area(self) -> float:
        """Bounding-box area in px^2."""
        w = float(self.bbox_xyxy[2] - self.bbox_xyxy[0])
        h = float(self.bbox_xyxy[3] - self.bbox_xyxy[1])
        return max(0.0, w) * max(0.0, h)


class VisionService:
    """Detect by color and unproject pixels using one camera."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        camera_name: str = "topdown",
        width: int = 640,
        height: int = 480,
    ) -> None:
        self.model = model
        self.camera_name = camera_name
        self.width = int(width)
        self.height = int(height)
        camera_id(model, camera_name)

    def render_detection_image(self, data: mujoco.MjData) -> NDArray[np.uint8]:
        """Render RGB image from the configured camera."""
        return render_rgb(self.model, data, self.camera_name, self.width, self.height)

    def detect_by_color(
        self,
        image: NDArray[np.uint8],
        hsv_lower: NDArray[np.float64 | np.int_],
        hsv_upper: NDArray[np.float64 | np.int_],
        *,
        min_area: int = 50,
    ) -> list[Detection]:
        """Return connected HSV detections sorted by area."""
        rgb = np.asarray(image, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3) image, got {rgb.shape}")

        hsv = _rgb_to_hsv_opencv_scale(rgb)
        lo = np.asarray(hsv_lower, dtype=np.float32)
        hi = np.asarray(hsv_upper, dtype=np.float32)

        if lo[0] <= hi[0]:
            mask = (
                (hsv[:, :, 0] >= lo[0]) & (hsv[:, :, 0] <= hi[0]) &
                (hsv[:, :, 1] >= lo[1]) & (hsv[:, :, 1] <= hi[1]) &
                (hsv[:, :, 2] >= lo[2]) & (hsv[:, :, 2] <= hi[2])
            )
        else:
            mask_h = (hsv[:, :, 0] >= lo[0]) | (hsv[:, :, 0] <= hi[0])
            mask = (
                mask_h &
                (hsv[:, :, 1] >= lo[1]) & (hsv[:, :, 1] <= hi[1]) &
                (hsv[:, :, 2] >= lo[2]) & (hsv[:, :, 2] <= hi[2])
            )

        return _mask_to_detections(mask, label="color", min_area=min_area)

    def unproject_pixel_to_world(
        self,
        data: mujoco.MjData,
        pixel_uv: NDArray[np.float64],
        target_z: float,
    ) -> NDArray[np.float64]:
        """Back-project [u, v] to world on plane z=target_z."""
        return unproject_pixel_to_world(
            self.model, data, self.camera_name, self.width, self.height, pixel_uv, target_z
        )


def unproject_pixel_to_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    width: int,
    height: int,
    pixel_uv: NDArray[np.float64],
    target_z: float,
) -> NDArray[np.float64]:
    """Project pixel ray and intersect with plane z=target_z."""
    cid = camera_id(model, camera_name)
    cam_pos = np.asarray(data.cam_xpos[cid], dtype=np.float64).reshape(3)
    r = np.asarray(data.cam_xmat[cid], dtype=np.float64).reshape(3, 3)

    fovy = float(model.cam_fovy[cid])
    if fovy <= 1e-6:
        fovy = 45.0
    tan_half = math.tan(math.radians(fovy) * 0.5)
    aspect = float(width) / float(max(height, 1))
    w1, h1 = max(int(width), 1), max(int(height), 1)

    u, v = float(pixel_uv[0]), float(pixel_uv[1])
    gl_y = (h1 - 1) - v
    ndc_x = 2.0 * u / (w1 - 1) - 1.0
    ndc_y = 2.0 * gl_y / (h1 - 1) - 1.0
    d_cam = np.array(
        [ndc_x * tan_half * aspect, ndc_y * tan_half, -1.0], dtype=np.float64
    )
    d_world = r @ d_cam

    if abs(d_world[2]) < 1e-9:
        raise ValueError(
            f"Camera ray is nearly parallel to z={target_z} plane "
            f"(d_world[2]={d_world[2]:.2e}); cannot back-project."
        )

    t = (target_z - cam_pos[2]) / d_world[2]
    pt_world = cam_pos + t * d_world
    return pt_world.copy()


def _rgb_to_hsv_opencv_scale(rgb_f32: np.ndarray) -> np.ndarray:
    """Convert RGB image [0..255] to HSV in OpenCV scale."""
    r = rgb_f32[:, :, 0] / 255.0
    g = rgb_f32[:, :, 1] / 255.0
    b = rgb_f32[:, :, 2] / 255.0

    v = np.maximum(np.maximum(r, g), b)
    c = v - np.minimum(np.minimum(r, g), b)

    s = np.where(v > 1e-7, c / np.where(v > 1e-7, v, 1.0), 0.0)

    eps = 1e-7
    h = np.zeros_like(v)
    m_rg = c > eps
    mask_r = m_rg & (v == r)
    h = np.where(mask_r, 60.0 * ((g - b) / np.where(mask_r, c + eps, 1.0) % 6.0), h)
    mask_g = m_rg & (v == g) & ~mask_r
    h = np.where(mask_g, 60.0 * ((b - r) / np.where(mask_g, c + eps, 1.0) + 2.0), h)
    mask_b = m_rg & ~mask_r & ~mask_g
    h = np.where(mask_b, 60.0 * ((r - g) / np.where(mask_b, c + eps, 1.0) + 4.0), h)
    h = h % 360.0

    out = np.stack([h / 2.0, s * 255.0, v * 255.0], axis=-1).astype(np.float32)
    return out


def _mask_to_detections(
    mask: np.ndarray,
    *,
    label: str,
    min_area: int,
) -> list[Detection]:
    """Convert a binary mask into connected-component detections."""
    from collections import deque  # noqa: PLC0415

    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    detections: list[Detection] = []

    ys_all, xs_all = np.where(mask)
    if len(ys_all) == 0:
        return detections

    seed_set = set(zip(ys_all.tolist(), xs_all.tolist()))

    for y0, x0 in zip(ys_all.tolist(), xs_all.tolist()):
        if visited[y0, x0]:
            continue
        region: list[tuple[int, int]] = []
        queue: deque[tuple[int, int]] = deque()
        queue.append((y0, x0))
        visited[y0, x0] = True
        while queue:
            y, x = queue.popleft()
            region.append((x, y))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and (ny, nx) in seed_set:
                    visited[ny, nx] = True
                    queue.append((ny, nx))

        area = len(region)
        if area < min_area:
            continue

        xs = [p[0] for p in region]
        ys = [p[1] for p in region]
        x1, x2 = float(min(xs)), float(max(xs))
        y1, y2 = float(min(ys)), float(max(ys))
        bbox = np.array([x1, y1, x2, y2], dtype=np.float64)
        conf = min(1.0, area / max(float(w * h), 1.0) * 100.0)
        detections.append(Detection(label=label, confidence=conf, bbox_xyxy=bbox))

    detections.sort(key=lambda d: d.area, reverse=True)
    return detections
