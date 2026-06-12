"""Computer vision helpers for detection and pixel→world back-projection."""

from __future__ import annotations

__all__ = [
    "VisionService",
    "Detection",
    "ColorRange",
    "ColorDetectionOperation",
    "PositionStabilizer",
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robot_manipulation_sim.vision.service import ColorDetectionOperation as ColorDetectionOperation
    from robot_manipulation_sim.vision.service import ColorRange as ColorRange
    from robot_manipulation_sim.vision.service import Detection as Detection
    from robot_manipulation_sim.vision.service import VisionService as VisionService
    from robot_manipulation_sim.vision.stabilization import PositionStabilizer as PositionStabilizer


def __getattr__(name: str):
    if name in (
        "VisionService",
        "Detection",
        "ColorRange",
        "ColorDetectionOperation",
    ):
        from robot_manipulation_sim.vision import service as _svc  # noqa: PLC0415

        return getattr(_svc, name)
    if name == "PositionStabilizer":
        from robot_manipulation_sim.vision import stabilization as _stab  # noqa: PLC0415

        return getattr(_stab, name)
    raise AttributeError(name)
