import numpy as np

from robot_manipulation_sim.cameras import pad_to_even_hw, resize_nn, stitch_camera_grid, stitch_camera_row


def test_pad_to_even_hw():
    assert pad_to_even_hw(np.zeros((3, 5, 3), dtype=np.uint8)).shape == (4, 6, 3)
    assert pad_to_even_hw(np.zeros((4, 6, 3), dtype=np.uint8)).shape == (4, 6, 3)


def test_resize_nn_identity():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    assert resize_nn(img, 10, 20) is img


def test_resize_nn_and_stitch():
    a = np.full((4, 6, 3), 10, dtype=np.uint8)
    b = np.full((8, 10, 3), 20, dtype=np.uint8)
    views = {"c0": a, "c1": b}
    row = stitch_camera_row(views, ("c0", "c1"), cell_h=2, cell_w=3)
    assert row.shape == (2, 6, 3)


def test_stitch_camera_grid():
    a = np.full((2, 2, 3), 1, dtype=np.uint8)
    b = np.full((2, 2, 3), 2, dtype=np.uint8)
    c = np.full((2, 2, 3), 3, dtype=np.uint8)
    d = np.full((2, 2, 3), 4, dtype=np.uint8)
    views = {"c0": a, "c1": b, "c2": c, "c3": d}
    g = stitch_camera_grid(views, ("c0", "c1", "c2", "c3"), cell_h=2, cell_w=2, nrows=2, ncols=2)
    assert g.shape == (4, 4, 3)
