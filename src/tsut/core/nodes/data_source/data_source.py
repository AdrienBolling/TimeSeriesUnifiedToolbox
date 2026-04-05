"""Define the base DataSource class for the TSUT Framework."""

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from tsut.core.nodes.node import Node, NodeConfig, NodeType


class DataSourceMetadata(BaseModel):
    """Metadata for a DataSource node."""

    _node_type: NodeType = NodeType.SOURCE


class DataSourceRunningConfig(BaseModel):
    """Configuration for running a DataSource node."""

    # Add any specific fields needed for running the data source if necessary


R = TypeVar("R", bound=DataSourceRunningConfig)


class DataSourceConfig[R](NodeConfig):
    """Base metadata configuration for all DataSource nodes in the TSUT Framework."""

    node_type: NodeType = NodeType.SOURCE
    running_config: R | None = None


class DataSourceNode[D_O, D_C_O](Node[None, None, D_O, D_C_O], ABC):
    """Base class for all data source nodes in the TSUT Framework."""

    metadata = DataSourceMetadata()

    def __init__(self, *, config: DataSourceConfig) -> None:
        """Initialize the DataSourceNode with the given configuration."""
        self._config = config

    @abstractmethod
    def setup_source(self) -> None:
        """Set up the data source (e.g., establish connections, load resources)."""
        ...

    @abstractmethod
    def fetch_data(self) -> dict[str, tuple[D_O, D_C_O]]:
        """Fetch data from the source.

        Returns:
            Fetched data

        """
        ...

    # --- Node API implementation --- Don't touch these unless you know what you're doing ---

    def node_fit(self, data: dict[str, tuple[None, None]]) -> None:
        """Fit the data source node with the given data.

        Args:
            data: Dictionary (unused for data sources)

        """
        _ = data  # Unused for data sources
        self.setup_source()

    def node_transform(
        self, data: dict[str, tuple[None, None]]
    ) -> dict[str, tuple[D_O, D_C_O]]:
        """Transform data through the Node by fetching data.

        Args:
            data: Dictionary (unused for data sources)

        Returns:
            Fetched data

        """
        _ = data  # Unused for data sources
        return self.fetch_data()

    # --- API convenience ---

    @property
    def running_config(self) -> DataSourceRunningConfig | None:
        """Convenience property to access the running configuration of the data source."""
        return self._config.running_config

    @property
    def config(self) -> DataSourceConfig:
        """Convenience property to access the full configuration of the data source."""
        return self._config
