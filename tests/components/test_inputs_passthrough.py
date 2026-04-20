"""Tests for the :class:`InputsPassthrough` data-source node."""

from __future__ import annotations

import pytest

from tsut.components.nodes.data_sources.inputs_passthrough import (
    InputsPassthrough,
    InputsPassthroughConfig,
)
from tsut.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
)
from tsut.core.common.enums import NodeExecutionMode
from tsut.core.nodes.node import Port


def _x_port() -> Port:
    return Port(
        arr_type=ArrayLikeEnum.PANDAS,
        data_structure=DataStructureEnum.TABULAR,
        data_category=DataCategoryEnum.NUMERICAL,
        data_shape="batch feature",
        desc="X",
    )


def _y_port() -> Port:
    return Port(
        arr_type=ArrayLikeEnum.PANDAS,
        data_structure=DataStructureEnum.TABULAR,
        data_category=DataCategoryEnum.NUMERICAL,
        data_shape="batch 1",
        desc="y",
        mode=[NodeExecutionMode.TRAINING, NodeExecutionMode.EVALUATION],
    )


class TestInputsPassthroughFetch:
    def test_fetch_returns_payload_unchanged(self, numerical_pair) -> None:
        cfg = InputsPassthroughConfig(out_ports={"X": _x_port()})
        node = InputsPassthrough(config=cfg)
        node.set_execution_mode(NodeExecutionMode.INFERENCE)

        payload = {"X": numerical_pair}
        out = node.fetch_data(payload)
        assert out == payload

    def test_fetch_detects_unknown_ports(self, numerical_pair) -> None:
        cfg = InputsPassthroughConfig(out_ports={"X": _x_port()})
        node = InputsPassthrough(config=cfg)
        node.set_execution_mode(NodeExecutionMode.INFERENCE)

        with pytest.raises(ValueError, match="unknown ports"):
            node.fetch_data({"bogus": numerical_pair})


class TestInputsPassthroughModeGating:
    """Mode-gated ports are only required when active in the current mode."""

    def test_y_is_required_during_training(self, numerical_pair) -> None:
        cfg = InputsPassthroughConfig(
            out_ports={"X": _x_port(), "y": _y_port()},
        )
        node = InputsPassthrough(config=cfg)
        node.set_execution_mode(NodeExecutionMode.TRAINING)

        with pytest.raises(ValueError, match="missing required inputs"):
            node.fetch_data({"X": numerical_pair})

    def test_y_is_skipped_during_inference(self, numerical_pair) -> None:
        cfg = InputsPassthroughConfig(
            out_ports={"X": _x_port(), "y": _y_port()},
        )
        node = InputsPassthrough(config=cfg)
        node.set_execution_mode(NodeExecutionMode.INFERENCE)

        out = node.fetch_data({"X": numerical_pair})
        assert "X" in out
        assert "y" not in out


class TestInputsPassthroughMetadata:
    def test_advertises_accepts_inputs(self) -> None:
        node = InputsPassthrough(config=InputsPassthroughConfig())
        assert node.accepts_inputs is True
