"""Define the base Transform class for the TSUT Framework."""

from abc import ABC, abstractmethod
from typing import TypeVar

from tsut.core.common.data.types import Data
from tsut.core.nodes.base import Node, NodeConfig, NodeType

D_I = TypeVar("D_I", bound=Data)
D_O = TypeVar("D_O", bound=Data)


class TransformConfig(NodeConfig):
    """Base metadata configuration for all Transform nodes in the TSUT Framework."""

    node_type: NodeType = NodeType.TRANSFORM


class TransformNode[D_I, D_O](Node[D_I, D_O], ABC):
    """Base class for all transform nodes in the TSUT Framework."""

    def __init__(self, *, config: TransformConfig) -> None:
        """Initialize the TransformNode with the given configuration."""
        super().__init__(config=config)

    @abstractmethod
    def transform(self, data: D_I) -> D_O:
        """Transform the input data to output data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed output data
        """
        ...

    def node_fit(self, data: dict[str, D_I]) -> None:
        """Fit the transform node with the given data.
        
        Args:
            data: Dictionary of input data
        """
        # Default implementation does nothing (stateless transform)
        pass

    def node_transform(self, data: dict[str, D_I]) -> dict[str, D_O]:
        """Transform data through the Node by applying transform to each value.
        
        Args:
            data: Dictionary of input data
            
        Returns:
            Dictionary of transformed output data
        """
        return {key: self.transform(value) for key, value in data.items()}
