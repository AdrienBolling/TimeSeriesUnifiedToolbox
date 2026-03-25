"""Define the base DataSource class for the TSUT Framework."""

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from tsut.core.common.data.types import Data
from tsut.core.nodes.node import Node, NodeConfig, NodeType

D_O = TypeVar("D_O", bound=Data)

class DataSourceMetadata(BaseModel):
    """Metadata for a DataSource node."""

    _node_type: NodeType = NodeType.SOURCE
    trainable = False  # Data sources are not trainable by default, but this can be overridden for specific data sources that are trainable (e.g., a data source that is just a wrapper around a pre-trained model that cannot be further trained).

class DataSourceRunningConfig(BaseModel):
    """Configuration for running a DataSource node."""
    # Add any specific fields needed for running the data source if necessary

R = TypeVar("R", bound=DataSourceRunningConfig)
class DataSourceConfig[R](NodeConfig):
    """Base metadata configuration for all DataSource nodes in the TSUT Framework."""

    node_type: NodeType = NodeType.SOURCE
    running_config: R


class DataSourceNode[D_O](Node[None, D_O], ABC):
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
    def fetch_data(self) -> D_O:
        """Fetch data from the source.

        Returns:
            Fetched data

        """
        ...

    # --- Node API implementation --- Don't touch these unless you know what you're doing ---

    def node_fit(self, data: None = None) -> None:
        """Fit the data source node with the given data.

        Args:
            data: Dictionary (unused for data sources)

        """
        _ = data  # Unused for data sources
        self.setup_source()

    def node_transform(self, data: None) -> D_O:
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
    def running_config(self) -> DataSourceRunningConfig:
        """Convenience property to access the running configuration of the data source."""
        return self._config.running_config

    @property
    def config(self) -> DataSourceConfig:
        """Convenience property to access the full configuration of the data source."""
        return self._config
