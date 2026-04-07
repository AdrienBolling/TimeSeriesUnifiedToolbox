"""Sink Node module."""

import pandas as pd

from tsut.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    TabularDataContext,
)
from tsut.core.nodes.node import Node, NodeConfig, NodeMetadata, NodeType, Port


class SinkConfig(NodeConfig):
    """Configuration for a Sink node.

    Provides a default ``dump`` input port and a placeholder ``_`` output port.
    Both accept mixed-category pandas data with arbitrary shape. The actual
    ports are redefined during pipeline compilation when connections are made.
    """

    node_type: NodeType = NodeType.SINK
    in_ports: dict[str, Port] = {
        "dump": Port(
            arr_type=ArrayLikeEnum.PANDAS,
            data_category=DataCategoryEnum.MIXED,
            data_shape="_ _",
            desc="All inputs will be sent here for dumping. All inputs will be sent here for dumping. The input nodes will be redefined during pipeline compilation",
        )
    }
    out_ports: dict[str, Port] = {
        "_": Port(
            arr_type=ArrayLikeEnum.PANDAS,
            data_category=DataCategoryEnum.MIXED,
            data_shape="_ _",
            desc="This port is for compatibility with the rest of the pipeline. The output nodes will be redefined during pipeline compilation",
        )
    }


class SinkMetadata(NodeMetadata):
    """Sink Node metadata."""


class Sink(Node[pd.DataFrame, TabularDataContext, pd.DataFrame, TabularDataContext]):
    """Terminal node that collects pipeline outputs.

    A Sink consumes data on its input ports and passes it through unchanged.
    New input/output port pairs are added dynamically during pipeline
    compilation via :meth:`add_port`.
    """

    metadata = SinkMetadata()

    def __init__(self, *, config: SinkConfig) -> None:
        """Initialize the Sink Node with the given configuration."""
        self._config = config

    def node_fit(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> None:
        """Fit the Sink Node. This method will be called during the training phase of the pipeline."""
        # Nothing to be done here

    def node_transform(
        self, data: dict[str, tuple[pd.DataFrame, TabularDataContext]]
    ) -> dict[str, tuple[pd.DataFrame, TabularDataContext]]:
        """Transform with the Sink Node. This method will be called during all phase of the pipeline."""
        # Return the data to be dumped.
        return data

    def add_port(self, port_name: str) -> None:
        """Add a matching input/output port pair to the Sink.

        Called during pipeline compilation when connecting upstream nodes.

        Args:
            port_name: Name for the new port pair.

        """
        self.in_ports[port_name] = Port(
            arr_type=ArrayLikeEnum.PANDAS,
            data_category=DataCategoryEnum.MIXED,
            data_shape="_ _",
            desc=f"Auto Input port for {port_name}.",
        )
        self.out_ports[port_name] = Port(
            arr_type=ArrayLikeEnum.PANDAS,
            data_category=DataCategoryEnum.MIXED,
            data_shape="_ _",
            desc=f"Auto Output port for {port_name}.",
        )
