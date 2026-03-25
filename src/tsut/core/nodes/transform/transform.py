"""Define the base Transform class for the TSUT Framework."""

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from tsut.core.nodes.node import Node, NodeConfig, NodeMetadata, NodeType

D_I = TypeVar("D_I")  # Input data type for the TransformNode
D_O = TypeVar("D_O") # Output data type for the TransformNode
T = TypeVar("T")  # Configuration type for the TransformNode, if needed. This is optional and can be set to None if not used.

class TransformMetadata(NodeMetadata):
    """Metadata for a TransformNode in a TSUT Pipeline."""

    _node_type: NodeType = NodeType.TRANSFORM

class TransformRunningConfig(BaseModel):
    """Running configuration for a TransformNode in the TSUT Framework.

    This will usually be used for execution parameters that are not relevant for the definition of the transform itself, but rather for how to run it.
    For example, in some transforms, this could be very specific parameters such as the backend to use for computations etc.
    """


class TransformHyperParameters(BaseModel):
    """Hyperparameters for a TransformNode in the TSUT Framework.

    This will usually be used for parameters that are relevant for the definition of the transform itself, and that are relevant to be tuned during hyperparameter tuning.
    For example, in some transforms, this could be the window size for a rolling window transform, etc.
    """

R = TypeVar("R", bound=TransformRunningConfig)
H = TypeVar("H", bound=TransformHyperParameters)

class TransformConfig[R, H](NodeConfig):
    """Base metadata configuration for all Transform nodes in the TSUT Framework."""

    node_type: NodeType = NodeType.TRANSFORM
    running_config: R
    hyperparameters: H


class TransformNode[D_I, D_O, T](Node[D_I, D_O], ABC):
    """Base class for all transform nodes in the TSUT Framework."""

    metadata = TransformMetadata()

    def __init__(self, *, config: TransformConfig) -> None:
        """Initialize the TransformNode with the given configuration."""
        self._config = config

    # --- Abstract Methods to reimplement ---

    @abstractmethod
    def fit(self, data: D_I) -> None:
        """Fit the transform with the given data."""
        ...

    @abstractmethod
    def transform(self, data: D_I) -> D_O:
        """Apply the transform to the given data."""
        ...

    @abstractmethod
    def get_params(self) -> T:
        """Get the current parameters of the transform."""
        ...

    @abstractmethod
    def set_params(self, params: T) -> None:
        """Set the parameters of the transform."""
        ...

    # --- API convenience ---

    @property
    def running_config(self) -> TransformRunningConfig:
        """Convenience property to access the running configuration of the transform."""
        return self._config.running_config

    @property
    def hyperparameters(self) -> TransformHyperParameters:
        """Convenience property to access the hyperparameters of the transform."""
        return self._config.hyperparameters

    @property
    def config(self) -> TransformConfig:
        """Convenience property to access the full configuration of the transform."""
        return self._config

    @property
    def params(self) -> T:
        """Convenience property to access the current parameters of the transform."""
        return self.get_params()

    # --- Implementations for Node interface, don't touch without a very good reason ---

    def node_fit(self, data: D_I) -> None:
        """Override of the Node's fit method to call the TransformNode's fit method."""
        return self.fit(data=data)

    def node_transform(self, data: D_I) -> D_O:
        """Override of the Node's transform method to call the TransformNode's transform method."""
        return self.transform(data=data)
