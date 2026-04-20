"""Tests for the tiny :class:`tsut.core.common.version.Version` model."""

from __future__ import annotations

from tsut.core.common.version import Version


def test_version_str_is_semver() -> None:
    assert str(Version(major=1, minor=2, patch=3)) == "1.2.3"


def test_version_round_trips_through_json() -> None:
    v = Version(major=0, minor=1, patch=0)
    restored = Version.model_validate_json(v.model_dump_json())
    assert restored == v
