"""Define the base DataSource class for the TSUT Framework."""

from abc import ABC, abstractmethod
from typing import TypeVar

from tsut.core.common.data.types import Data
from tsut.core.nodes.base import Node, NodeConfig, NodeType

D_O = TypeVar("D_O", bound=Data)


class DataSourceConfig(NodeConfig):
    """Base metadata configuration for all DataSource nodes in the TSUT Framework."""

    node_type: NodeType = NodeType.SOURCE


class DataSourceNode[D_O](Node[None, D_O], ABC):
    """Base class for all data source nodes in the TSUT Framework."""

    def __init__(self, *, config: DataSourceConfig) -> None:
        """Initialize the DataSourceNode with the given configuration."""
        super().__init__(config=config)

    @abstractmethod
    def fetch_data(self) -> D_O:
        """Fetch data from the source.
        
        Returns:
            Fetched data
        """
        ...

    def node_fit(self, data: dict) -> None:
        """Fit the data source node with the given data.
        
        Args:
            data: Dictionary (unused for data sources)
        """
        # Default implementation does nothing (data sources don't need fitting)
        pass

    def node_transform(self, data: dict) -> dict[str, D_O]:
        """Transform data through the Node by fetching data.
        
        Args:
            data: Dictionary (unused for data sources)
            
        Returns:
            Dictionary with fetched data under "output" key
        """
        return {"output": self.fetch_data()}
