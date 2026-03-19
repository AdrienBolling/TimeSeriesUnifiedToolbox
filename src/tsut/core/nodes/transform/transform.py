"""Define the base Transform class for the TSUT Framework."""

from abc import ABC, abstractmethod
from typing import TypeVar

from tsut.core.nodes.base import Node, NodeConfig, NodeType

D_I = TypeVar("D_I")
D_O = TypeVar("D_O")


class TransformConfig(NodeConfig):
    """Base metadata configuration for all Transform nodes in the TSUT Framework."""

    node_type: NodeType = NodeType.TRANSFORM


class TransformNode[D_I, D_O](Node[D_I, D_O], ABC):
    """Base class for all transform nodes in the TSUT Framework."""

    def __init__(self, *, config: TransformConfig) -> None:
        """Initialize the TransformNode with the given configuration."""
        super().__init__(config=config)

    def node_fit(self, data: D_I) -> None:
        """Fit the transform node with the given data.

        Args:
            data: Input data

        """
        # Default implementation does nothing (stateless transform)

    @abstractmethod
    def node_transform(self, data: D_I) -> D_O:
        """Transform data through the Node by applying transform to each value.
        
        Args:
            data: Input data
            
        Returns:
            Transformed output data

        """
        raise NotImplementedError
