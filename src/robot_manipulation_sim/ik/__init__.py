"""Inverse kinematics helpers for the MuJoCo UR5e harness (optional ``mink`` extra).

Install: ``pip install -e ".[ik]"`` (see ``pyproject.toml`` optional-dependencies ``ik``).

- ``MujocoMinkIkService`` — differential IK on the **same** ``MjModel`` as the simulator (requires ``mink``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "check_ik_dependencies",
    "MujocoMinkIkService",
]

if TYPE_CHECKING:
    from robot_manipulation_sim.ik.service import MujocoMinkIkService as MujocoMinkIkService


def check_ik_dependencies() -> None:
    """Import ``mink`` + QP backend; raise ``ImportError`` with install hint if missing."""
    try:
        import mink  # noqa: F401, PLC0415
        import qpsolvers  # noqa: F401, PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'Install IK extras: pip install -e ".[ik]" (needs mink and qpsolvers[daqp]).'
        ) from exc


def __getattr__(name: str):
    if name == "MujocoMinkIkService":
        check_ik_dependencies()
        from robot_manipulation_sim.ik.service import MujocoMinkIkService as _cls

        return _cls
    raise AttributeError(name)
