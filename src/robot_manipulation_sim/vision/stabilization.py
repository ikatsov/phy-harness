"""Reusable temporal smoothing/stability helpers for position detections."""

from __future__ import annotations

from collections import deque

import numpy as np


class PositionStabilizer:
    """Track one primary position and multiple labeled positions over a fixed window."""

    def __init__(self, *, labels: tuple[str, ...], window_size: int) -> None:
        self._labels = labels
        self._window_size = int(window_size)
        self.primary: deque[np.ndarray] = deque(maxlen=self._window_size)
        self.labeled: dict[str, deque[np.ndarray]] = {k: deque(maxlen=self._window_size) for k in self._labels}

    def push(self, primary_world: np.ndarray, labeled_world: dict[str, np.ndarray]) -> None:
        """Append one observation sample for primary and all configured labels."""
        self.primary.append(np.asarray(primary_world, dtype=np.float64))
        for k in self._labels:
            self.labeled[k].append(np.asarray(labeled_world[k], dtype=np.float64))

    @staticmethod
    def _mean_std(samples: deque[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        arr = np.stack(list(samples), axis=0)
        return np.mean(arr, axis=0), np.std(arr, axis=0)

    def stable_means(
        self,
        *,
        primary_xy_std_max: float,
        labeled_xy_std_max: float,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
        """Return means only when every stream is fully buffered and low-variance in XY."""
        if len(self.primary) < self._window_size:
            return None
        p_mean, p_std = self._mean_std(self.primary)
        if float(np.linalg.norm(p_std[:2])) > primary_xy_std_max:
            return None
        out: dict[str, np.ndarray] = {}
        for k in self._labels:
            q = self.labeled[k]
            if len(q) < self._window_size:
                return None
            m, s = self._mean_std(q)
            if float(np.linalg.norm(s[:2])) > labeled_xy_std_max:
                return None
            out[k] = m
        return p_mean, out

    def early_means(self, *, min_samples: int) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
        """Return means from partial buffers once ``min_samples`` are available."""
        if len(self.primary) < int(min_samples):
            return None
        p_mean = np.mean(np.stack(list(self.primary), axis=0), axis=0)
        out: dict[str, np.ndarray] = {}
        for k in self._labels:
            if len(self.labeled[k]) == 0:
                return None
            out[k] = np.mean(np.stack(list(self.labeled[k]), axis=0), axis=0)
        return p_mean, out
