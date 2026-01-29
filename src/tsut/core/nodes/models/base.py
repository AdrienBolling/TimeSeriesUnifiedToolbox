"""Define the base Model class for the TSUT Framework."""

from abc import ABC, abstractmethod
from typing import TypeVar, override

from tsut.core.common.data.types import ConfigData, ContextData, Data
from tsut.core.nodes.base import Node, NodeConfig, NodeType

D_I = TypeVar("D_I", bound=Data)
D_O = TypeVar("D_O", bound=Data)
C = TypeVar("C", bound=ConfigData)


class ModelConfig(NodeConfig):
    """Base metadata configuration for all Models in the TSUT Framework."""

    node_type: NodeType = NodeType.MODEL


class Model[D_I, D_O, C](
    Node[D_I, D_O], ABC
):  # Model is already implicitely an ABC via Node but explicit is better.
    """Base class for all models in the TSUT Framework."""

    def __init__(self, *, config: ModelConfig) -> None:
        """Minimal constructor for Model class."""
        super().__init__(config=config)

    # --- Abstract Methods to reimplement ---

    @abstractmethod
    def fit(self, data: dict[str, D_I | ContextData]) -> None:
        """Fit the model with the given data."""
        ...

    @abstractmethod
    def predict(
        self, data: dict[str, D_I | ContextData]
    ) -> dict[str, D_O | ContextData]:
        """Predict using the model with the given data."""
        ...

    @abstractmethod
    def get_params(self) -> C:
        """Get the model parameters."""
        ...

    @abstractmethod
    def restore_params(self, params: C) -> None:
        """Restore the model parameters."""
        ...

    # --- Overrides for Node interface ---

    @override
    def node_fit(self, data: dict[str, D_I | ContextData]) -> None:
        return self.fit(data=data)

    @override
    def node_transform(
        self, data: dict[str, D_I | ContextData]
    ) -> dict[str, D_O | ContextData]:
        return self.predict(data=data)
