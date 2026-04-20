"""Tests for :class:`tsut.core.nodes.transform.transform.TransformNode`."""

from __future__ import annotations

import pytest

from tests.shims.nodes import IdentityTransform, IdentityTransformConfig
from tests.shims.tabular import numerical_pair


class TestTransformFittedGate:
    def test_transform_before_fit_raises(self) -> None:
        node = IdentityTransform(config=IdentityTransformConfig())
        payload = {"input": numerical_pair()}
        with pytest.raises(ValueError, match="fitted before"):
            node.node_transform(payload)

    def test_node_fit_marks_as_fitted(self) -> None:
        node = IdentityTransform(config=IdentityTransformConfig())
        payload = {"input": numerical_pair()}
        node.node_fit(payload)
        # Re-calling ``node_transform`` should now succeed.
        out = node.node_transform(payload)
        assert "output" in out

    def test_node_fit_transform_is_equivalent_to_fit_then_transform(self) -> None:
        node = IdentityTransform(config=IdentityTransformConfig())
        payload = {"input": numerical_pair()}
        out = node.node_fit_transform(payload)
        assert node.fit_calls == 1
        assert node.transform_calls == 1
        assert "output" in out
