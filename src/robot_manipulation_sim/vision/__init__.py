"""Computer vision helpers for detection and pixel→world back-projection."""

from __future__ import annotations

__all__ = [
    "VisionService",
    "Detection",
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robot_manipulation_sim.vision.service import Detection as Detection
    from robot_manipulation_sim.vision.service import VisionService as VisionService


def __getattr__(name: str):
    if name in ("VisionService", "Detection"):
        from robot_manipulation_sim.vision import service as _svc  # noqa: PLC0415

        return getattr(_svc, name)
    raise AttributeError(name)
