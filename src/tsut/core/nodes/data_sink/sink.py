"""Sink Node module."""

import pandas as pd

from tsut.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    TabularDataContext,
)
from tsut.core.nodes.node import Node, NodeConfig, NodeMetadata, NodeType, Port


class SinkConfig(NodeConfig):
    """Sink Node configuration."""

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
    """Sink Node.

    A Sink Node is a Node that consumes data from its input ports and does not produce any output.
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
        """Add a port to the Sink Node. This method will be called during pipeline compilation when connecting the Sink Node to other nodes."""
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
