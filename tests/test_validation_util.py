"""Tests for ``robot_manipulation_sim.validation.util``."""

from __future__ import annotations

import pytest

from robot_manipulation_sim.validation.util import coerce_bool


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        (True, True),
        (False, False),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("true", True),
        ("", False),
        ("0", False),
        ("1", True),
        ("no", False),
        ("yes", True),
        (0, False),
        (1, True),
    ],
)
def test_coerce_bool(value, expected: bool) -> None:
    assert coerce_bool(value, default=False) is expected


def test_coerce_bool_default_when_unknown_string() -> None:
    assert coerce_bool("maybe", default=False) is False
    assert coerce_bool("maybe", default=True) is True
