"""Tests for :class:`tsut.core.common.registry.Registry` CRUD behaviour.

These tests exercise the generic CRUD layer in isolation, without any
node-specific schema.  Node-level registration concerns (entity class +
config class + metadata bundle) live in
``tests/core/nodes/test_node_registry.py``.
"""

from __future__ import annotations

import pytest

from tsut.core.common.registry import Registry


class _Dummy:
    """Placeholder class used as a registry value."""


class _DummyConfig:
    """Placeholder config class used as a registry value."""


def _fresh_registry() -> Registry:
    """Return an empty registry so test order does not matter."""
    return Registry(entity="widget")


class TestRegistryCRUD:
    def test_register_and_get_round_trips(self) -> None:
        reg = _fresh_registry()
        reg._register("a", _Dummy, _DummyConfig, family="foo")
        entry = reg.get("a")

        assert entry["widget_class"] is _Dummy
        assert entry["widget_config_class"] is _DummyConfig
        assert entry["family"] == "foo"

    def test_register_entry_stores_arbitrary_fields(self) -> None:
        reg = _fresh_registry()
        reg.register_entry("a", x=1, y="two", tags=["alpha"])

        assert reg["a"] == {"x": 1, "y": "two", "tags": ["alpha"]}

    def test_register_twice_raises(self) -> None:
        reg = _fresh_registry()
        reg.register_entry("a", x=1)
        with pytest.raises(ValueError, match="already registered"):
            reg.register_entry("a", x=2)

    def test_unregister_removes_entry(self) -> None:
        reg = _fresh_registry()
        reg.register_entry("a", x=1)
        reg.unregister("a")
        assert "a" not in reg

    def test_unregister_missing_raises(self) -> None:
        reg = _fresh_registry()
        with pytest.raises(ValueError, match="not registered"):
            reg.unregister("ghost")

    def test_get_missing_raises(self) -> None:
        reg = _fresh_registry()
        with pytest.raises(ValueError, match="not registered"):
            reg.get("ghost")


class TestRegistryContainerProtocol:
    """``Registry`` advertises dict-like behaviour (len/iter/contains)."""

    def test_container_protocol(self) -> None:
        reg = _fresh_registry()
        reg.register_entry("a", v=1)
        reg.register_entry("b", v=2)

        assert len(reg) == 2
        assert "a" in reg
        assert "c" not in reg
        assert set(iter(reg)) == {"a", "b"}
        assert set(reg.keys()) == {"a", "b"}
        assert dict(reg.items()) == {"a": {"v": 1}, "b": {"v": 2}}

    def test_getitem_delegates_to_get(self) -> None:
        reg = _fresh_registry()
        reg.register_entry("a", v=1)
        assert reg["a"] == reg.get("a")
