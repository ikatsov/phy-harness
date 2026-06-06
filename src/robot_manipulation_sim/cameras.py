"""RGB and depth rendering from named MJCF cameras."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mujoco
import numpy as np


@dataclass(frozen=True)
class CameraSpec:
    """Named camera in the MJCF."""

    name: str
    width: int = 640
    height: int = 480


def camera_id(model: mujoco.MjModel, name: str) -> int:
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cid < 0:
        known = [model.camera(i).name for i in range(model.ncam) if model.camera(i).name]
        raise KeyError(f"Unknown camera {name!r}. Available: {known}")
    return cid


# ---------------------------------------------------------------------------
# Persistent renderer pool
# ---------------------------------------------------------------------------
# mujoco.Renderer allocates a GL/GPU context on construction, which is expensive.
# Reusing renderers across render_rgb / render_depth calls on the same model
# eliminates thousands of needless alloc/free cycles per rollout.
#
# Key: (id(model), height, width)  — id() is the CPython object address, so
# distinct MjModel instances get distinct pools even if they describe the same MJCF.
# Call clear_renderer_cache() when a model is about to be discarded.
# ---------------------------------------------------------------------------

_RendererKey = tuple[int, int, int]  # (id(model), height, width)
_renderer_cache: dict[_RendererKey, mujoco.Renderer] = {}


def _get_renderer(model: mujoco.MjModel, height: int, width: int) -> mujoco.Renderer:
    key = (id(model), int(height), int(width))
    rend = _renderer_cache.get(key)
    if rend is None:
        rend = mujoco.Renderer(model, height=int(height), width=int(width))
        _renderer_cache[key] = rend
    return rend


def clear_renderer_cache(model: mujoco.MjModel | None = None) -> None:
    """Close and remove cached renderers.

    Pass a specific *model* to release only renderers for that model instance
    (call before discarding an ``MjModel``). Omit to clear the entire cache.
    """
    if model is None:
        keys = list(_renderer_cache.keys())
    else:
        target = id(model)
        keys = [k for k in _renderer_cache if k[0] == target]
    for key in keys:
        rend = _renderer_cache.pop(key, None)
        if rend is not None:
            try:
                rend.close()
            except Exception:  # noqa: BLE001
                pass


def render_rgb(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    width: int,
    height: int,
    *,
    scene_option: mujoco.MjvOption | None = None,
) -> np.ndarray:
    """Return uint8 HxWx3 RGB image from a fixed or body-mounted camera.

    Reuses a cached ``mujoco.Renderer`` for this ``(model, height, width)`` combination
    to avoid repeated GL context allocation. Call :func:`clear_renderer_cache` when the
    model is about to be discarded.
    """
    renderer = _get_renderer(model, height, width)
    renderer.update_scene(data, camera=camera_id(model, camera_name), scene_option=scene_option)
    return renderer.render()


def render_depth(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    width: int,
    height: int,
    *,
    scene_option: mujoco.MjvOption | None = None,
) -> np.ndarray:
    """Return float32 HxW **metric depth** (meters) from the same camera as ``render_rgb``.

    Reuses a cached renderer; temporarily enables depth rendering and restores RGB mode.
    """
    renderer = _get_renderer(model, height, width)
    renderer.update_scene(data, camera=camera_id(model, camera_name), scene_option=scene_option)
    renderer.enable_depth_rendering()
    try:
        return renderer.render()
    finally:
        renderer.disable_depth_rendering()


def depth_to_grayscale_rgb(
    depth: np.ndarray,
    model: mujoco.MjModel | None = None,
    *,
    extent_far_cap: float = 3.5,
    exp_decay: float = 2.85,
    gamma: float = 1.0,
) -> np.ndarray:
    """Map metric depth (HxW float, meters) to uint8 HxWx3 **grayscale** (R=G=B) for video / VLM.

    Contrast is anchored to the model **scene extent** (``mjModel.stat.extent``) and the same
    near/far convention as the MuJoCo renderer (``vis.map.znear`` / ``vis.map.zfar`` × extent), then
    refined with robust percentiles so invalid pixels are excluded. ``extent_far_cap`` limits how far
    out (in ×extent) the far clip is for the display ramp so tabletop-scale geometry dominates the
    normalization band.

    **Near = bright**, **far = dark**. After linear normalization ``u ∈ [0,1]`` (0 at the near end of
    the ramp, 1 at the far end), intensity uses an **exponential decay** in ``u``:

    ``t = (exp(-k·u) - exp(-k)) / (1 - exp(-k))``

    so most grayscale steps are spent on **closest** depths (large ``∂t/∂u`` near ``u=0``), improving
    local contrast on the manipuland and gripper. ``exp_decay`` is ``k``; larger ``k`` pushes the curve
    harder toward the near field. For ``exp_decay`` near zero, the mapping approaches ``t ≈ 1-u``.

    Optional ``gamma`` (default 1) applies ``t ← clip(t,0,1)**gamma`` after the exponential map.
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"expected HxW depth, got {d.shape}")
    valid = np.isfinite(d) & (d > 1e-9)
    h, w = d.shape
    if not np.any(valid):
        return np.zeros((h, w, 3), dtype=np.uint8)

    if model is not None:
        extent = max(float(model.stat.extent), 1e-6)
        zn = float(model.vis.map.znear) * extent
        zf_cap = min(float(model.vis.map.zfar) * extent, float(extent_far_cap) * extent)
    else:
        extent = 1.0
        zn = 0.01
        zf_cap = 5.0

    p_lo = float(np.percentile(d[valid], 1.0))
    p_hi = float(np.percentile(d[valid], 99.0))

    lo = float(np.clip(p_lo - 0.04 * extent, zn, zf_cap - 1e-3))
    hi = float(np.clip(p_hi + 0.06 * extent, lo + 0.08 * extent, zf_cap))

    min_span = 0.12 * extent
    if hi - lo < min_span:
        mid = 0.5 * (lo + hi)
        lo = max(zn, mid - 0.5 * min_span)
        hi = min(zf_cap, mid + 0.5 * min_span)
    if hi <= lo:
        hi = lo + min_span

    u = (d - lo) / (hi - lo)
    u = np.clip(u, 0.0, 1.0)
    # u = 0 at shallow depth (near), u = 1 at deep (far) within the display band.
    k = float(exp_decay)
    if k > 1e-6:
        ek = float(np.exp(-k))
        den = max(1.0 - ek, 1e-12)
        t = (np.exp(-k * u) - ek) / den
    else:
        t = 1.0 - u
    t = np.clip(t, 0.0, 1.0)
    if abs(float(gamma) - 1.0) > 1e-6:
        t = np.power(t, float(gamma))
    g = (255.0 * t).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=-1)
    rgb[~valid] = 0
    return rgb


def depth_to_rgb(
    depth: np.ndarray,
    model: mujoco.MjModel | None = None,
    *,
    extent_far_cap: float = 3.5,
    exp_decay: float = 2.85,
    gamma: float = 1.0,
) -> np.ndarray:
    """Backward-compatible name: same as :func:`depth_to_grayscale_rgb`."""
    return depth_to_grayscale_rgb(depth, model, extent_far_cap=extent_far_cap, exp_decay=exp_decay, gamma=gamma)


def render_cameras(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cameras: Iterable[CameraSpec],
    *,
    scene_option: mujoco.MjvOption | None = None,
) -> dict[str, np.ndarray]:
    """Render several cameras (one Renderer per camera for clarity and correctness)."""
    out: dict[str, np.ndarray] = {}
    for spec in cameras:
        out[spec.name] = render_rgb(
            model, data, spec.name, spec.width, spec.height, scene_option=scene_option
        )
    return out


def resize_nn(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbor resize of uint8 HxWx3 (or float) to (height, width)."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3, got {rgb.shape}")
    ih, iw = int(rgb.shape[0]), int(rgb.shape[1])
    if ih == height and iw == width:
        return rgb
    y_idx = (np.arange(height, dtype=np.float64) * ih / height).astype(np.int32)
    x_idx = (np.arange(width, dtype=np.float64) * iw / width).astype(np.int32)
    y_idx = np.clip(y_idx, 0, ih - 1)
    x_idx = np.clip(x_idx, 0, iw - 1)
    return rgb[y_idx[:, np.newaxis], x_idx]


def stitch_camera_row(
    views: dict[str, np.ndarray],
    names: tuple[str, ...],
    cell_h: int,
    cell_w: int,
) -> np.ndarray:
    """Place named camera images left-to-right after resizing each to cell_h x cell_w."""
    tiles: list[np.ndarray] = []
    for name in names:
        if name not in views:
            raise KeyError(f"missing camera {name!r}; have {sorted(views)}")
        tiles.append(resize_nn(views[name], cell_h, cell_w))
    return np.concatenate(tiles, axis=1)


def stitch_camera_grid(
    views: dict[str, np.ndarray],
    names: tuple[str, ...],
    cell_h: int,
    cell_w: int,
    *,
    nrows: int,
    ncols: int,
) -> np.ndarray:
    """Tile cameras in row-major order: row0 left→right, row1 left→right, each cell cell_h×cell_w."""
    if nrows < 1 or ncols < 1:
        raise ValueError("nrows and ncols must be positive")
    if len(names) != nrows * ncols:
        raise ValueError(f"need nrows*ncols={nrows * ncols} names, got {len(names)}")
    rows: list[np.ndarray] = []
    k = 0
    for _ in range(nrows):
        row_tiles: list[np.ndarray] = []
        for _ in range(ncols):
            name = names[k]
            k += 1
            if name not in views:
                raise KeyError(f"missing camera {name!r}; have {sorted(views)}")
            row_tiles.append(resize_nn(views[name], cell_h, cell_w))
        rows.append(np.concatenate(row_tiles, axis=1))
    return np.concatenate(rows, axis=0)


def pad_to_even_hw(rgb: np.ndarray) -> np.ndarray:
    """Pad bottom/right with zeros so H and W are even (required for libx264 + yuv420p)."""
    rgb = np.asarray(rgb)
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    ph = h % 2
    pw = w % 2
    if ph == 0 and pw == 0:
        return rgb
    return np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="constant")


def render_multiview_strip(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    record_specs: tuple[CameraSpec, ...],
    *,
    order: tuple[str, ...],
    cell_h: int,
    cell_w: int,
    scene_option: mujoco.MjvOption | None = None,
) -> np.ndarray:
    """Render all record_specs, then return one horizontal strip in ``order``."""
    views = render_cameras(model, data, record_specs, scene_option=scene_option)
    strip = stitch_camera_row(views, order, cell_h, cell_w)
    return pad_to_even_hw(strip)


def render_multiview_grid(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    record_specs: tuple[CameraSpec, ...],
    *,
    order: tuple[str, ...],
    cell_h: int,
    cell_w: int,
    nrows: int,
    ncols: int,
    scene_option: mujoco.MjvOption | None = None,
) -> np.ndarray:
    """Render cameras, then tile them in an ``nrows``×``ncols`` grid (``order`` is row-major)."""
    views = render_cameras(model, data, record_specs, scene_option=scene_option)
    grid = stitch_camera_grid(views, order, cell_h, cell_w, nrows=nrows, ncols=ncols)
    return pad_to_even_hw(grid)


def project_world_positions_to_camera_pixels(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    width: int,
    height: int,
    points_xyz: np.ndarray,
) -> np.ndarray:
    """Project world-space points to pixel coordinates matching ``Renderer`` RGB output.

    Returns an ``(N, 2)`` float64 array ``[u, v]`` with ``u`` in ``[0, width-1]``, ``v`` in ``[0, height-1]``
    (top-left origin, same indexing as post-``flipud`` MuJoCo images). Rows are ``nan`` when the point is
    behind the camera or falls outside a loose frustum clip.

    ``points_xyz`` must have shape ``(N, 3)``. Uses ``mjData.cam_xpos`` / ``cam_xmat`` after ``mj_forward``.
    """
    cid = camera_id(model, camera_name)
    pos = np.asarray(data.cam_xpos[cid], dtype=np.float64).reshape(3)
    r_flat = np.asarray(data.cam_xmat[cid], dtype=np.float64)
    if r_flat.size != 9:
        raise ValueError(f"cam_xmat for {camera_name!r} must have length 9, got {r_flat.size}")
    r = r_flat.reshape(3, 3)
    fovy = float(model.cam_fovy[cid])
    if fovy <= 1e-6:
        fovy = 45.0
    aspect = float(width) / float(max(height, 1))
    tan_half = float(np.tan(np.deg2rad(fovy) * 0.5))
    w1, h1 = max(int(width), 1), max(int(height), 1)

    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    out = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    for i in range(pts.shape[0]):
        pc = r.T @ (pts[i] - pos)
        z_forward = -float(pc[2])
        if z_forward <= 1e-9:
            continue
        invz = 1.0 / z_forward
        ndc_x = float(pc[0] * invz) / (tan_half * aspect)
        ndc_y = float(pc[1] * invz) / tan_half
        if abs(ndc_x) > 2.0 or abs(ndc_y) > 2.0:
            continue
        gl_x = (0.5 * ndc_x + 0.5) * (w1 - 1)
        gl_y = (0.5 * ndc_y + 0.5) * (h1 - 1)
        iy = (h1 - 1) - gl_y
        out[i, 0] = float(np.clip(gl_x, 0.0, w1 - 1))
        out[i, 1] = float(np.clip(iy, 0.0, h1 - 1))
    return out


def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Integer Bresenham line (inclusive endpoints)."""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    points: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return points


def draw_polylines_on_tile(
    tile: np.ndarray,
    polylines: Iterable[np.ndarray],
    colors: Iterable[tuple[int, int, int]],
) -> None:
    """Draw 1 px polylines on ``uint8`` ``HxWx3`` image in-place (clips to bounds)."""
    h, w = int(tile.shape[0]), int(tile.shape[1])
    for poly, col in zip(polylines, colors, strict=True):
        arr = np.asarray(poly, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
            continue
        r, g, b = (int(c) for c in col)
        for j in range(arr.shape[0] - 1):
            a = arr[j]
            b_ = arr[j + 1]
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b_))):
                continue
            x0, y0 = int(round(a[0])), int(round(a[1]))
            x1, y1 = int(round(b_[0])), int(round(b_[1]))
            for px, py in _bresenham_line(x0, y0, x1, y1):
                if 0 <= px < w and 0 <= py < h:
                    tile[py, px, 0] = r
                    tile[py, px, 1] = g
                    tile[py, px, 2] = b


def render_rollout_rgb_depth_grid(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cell_h: int,
    cell_w: int,
    *,
    perspective_camera: str = "overview",
    wrist_camera: str = "wrist_rgb",
    scene_option: mujoco.MjvOption | None = None,
    perspective_traces: tuple[tuple[np.ndarray, tuple[int, int, int]], ...] | None = None,
) -> np.ndarray:
    """2×2 rollout frame: row0 = perspective RGB | wrist RGB; row1 = matching **grayscale** depth tiles.

    ``perspective_camera`` defaults to ``overview`` (scene perspective). ``wrist_camera`` is typically
    ``wrist_rgb``. Each tile is resized to ``cell_h``×``cell_w`` before stitching. Depth tiles use
    :func:`depth_to_grayscale_rgb` with ``model`` (extent + clip planes + percentiles + exponential decay in depth).

    When ``perspective_traces`` is set, each ``(polyline, rgb)`` pair is drawn on **both** the perspective
    RGB tile (top-left) and the perspective depth tile (bottom-left). ``polyline`` is ``(T, 2)`` float pixel
    coords in the same ``cell_w``×``cell_h`` space as the resized overview tiles.
    """
    ov_rgb = render_rgb(model, data, perspective_camera, cell_w, cell_h, scene_option=scene_option)
    wr_rgb = render_rgb(model, data, wrist_camera, cell_w, cell_h, scene_option=scene_option)
    ov_dep = depth_to_grayscale_rgb(
        render_depth(model, data, perspective_camera, cell_w, cell_h, scene_option=scene_option),
        model,
    )
    wr_dep = depth_to_grayscale_rgb(
        render_depth(model, data, wrist_camera, cell_w, cell_h, scene_option=scene_option),
        model,
    )
    ov_rgb = resize_nn(ov_rgb, cell_h, cell_w)
    wr_rgb = resize_nn(wr_rgb, cell_h, cell_w)
    ov_dep = resize_nn(ov_dep, cell_h, cell_w)
    wr_dep = resize_nn(wr_dep, cell_h, cell_w)

    if perspective_traces:
        polys = tuple(p for p, _ in perspective_traces)
        cols = tuple(c for _, c in perspective_traces)
        draw_polylines_on_tile(ov_rgb, polys, cols)
        draw_polylines_on_tile(ov_dep, polys, cols)

    row0 = np.concatenate([ov_rgb, wr_rgb], axis=1)
    row1 = np.concatenate([ov_dep, wr_dep], axis=1)
    return pad_to_even_hw(np.concatenate([row0, row1], axis=0))


def render_rollout_four_view_grid(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cell_h: int,
    cell_w: int,
    *,
    perspective_camera: str = "overview",
    wrist_camera: str = "wrist_rgb",
    front_camera: str = "front_rgb",
    side_camera: str = "side_rgb",
    scene_option: mujoco.MjvOption | None = None,
    perspective_traces: tuple[tuple[np.ndarray, tuple[int, int, int]], ...] | None = None,
    front_traces: tuple[tuple[np.ndarray, tuple[int, int, int]], ...] | None = None,
    side_traces: tuple[tuple[np.ndarray, tuple[int, int, int]], ...] | None = None,
) -> np.ndarray:
    """2×2 rollout frame: row0 = perspective RGB | wrist RGB; row1 = front isometric | side isometric.

    ``front_camera`` looks along +Y (frontal view of robot workspace); ``side_camera`` looks along +X
    (profile view of the lift arc). Each tile is resized to ``cell_h``×``cell_w`` before stitching.

    Optional kinematic overlays (each entry is ``(polyline, rgb)``; ``polyline`` is ``(T, 2)`` float
    pixel coords in ``cell_w``×``cell_h`` space after resize):

    * ``perspective_traces`` — drawn on the **top-left** tile (``perspective_camera``).
    * ``front_traces`` — drawn on the **bottom-left** tile (``front_camera``).
    * ``side_traces`` — drawn on the **bottom-right** tile (``side_camera``, often top-down).

    The **wrist** tile (top-right) is left without polylines unless extended later.
    """
    ov_rgb = render_rgb(model, data, perspective_camera, cell_w, cell_h, scene_option=scene_option)
    wr_rgb = render_rgb(model, data, wrist_camera, cell_w, cell_h, scene_option=scene_option)
    fr_rgb = render_rgb(model, data, front_camera, cell_w, cell_h, scene_option=scene_option)
    si_rgb = render_rgb(model, data, side_camera, cell_w, cell_h, scene_option=scene_option)
    ov_rgb = resize_nn(ov_rgb, cell_h, cell_w)
    wr_rgb = resize_nn(wr_rgb, cell_h, cell_w)
    fr_rgb = resize_nn(fr_rgb, cell_h, cell_w)
    si_rgb = resize_nn(si_rgb, cell_h, cell_w)

    def _draw(tile: np.ndarray, traces: tuple[tuple[np.ndarray, tuple[int, int, int]], ...] | None) -> None:
        if traces:
            draw_polylines_on_tile(tile, tuple(p for p, _ in traces), tuple(c for _, c in traces))

    _draw(ov_rgb, perspective_traces)
    _draw(fr_rgb, front_traces)
    _draw(si_rgb, side_traces)

    row0 = np.concatenate([ov_rgb, wr_rgb], axis=1)
    row1 = np.concatenate([fr_rgb, si_rgb], axis=1)
    return pad_to_even_hw(np.concatenate([row0, row1], axis=0))
