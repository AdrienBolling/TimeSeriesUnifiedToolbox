"""Define the base Model class for the TSUT Framework."""

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic.main import BaseModel

from tsut.core.nodes.node import Node, NodeConfig, NodeMetadata, NodeType

D_I = TypeVar("D_I")
D_O = TypeVar("D_O")
P = TypeVar("P")  # To be used for the model's parameters's type.

class ModelMetadata(NodeMetadata):
    """Metadata for a Model in a TSUT Pipeline."""

    _node_type: NodeType = NodeType.MODEL
class ModelRunningConfig(BaseModel):
    """Running configuration for a Model in the TSUT Framework.

    This will usually be used for execution parameters that are not relevant for the definition of the model itself, but rather for how to run it.
    For example, in most ML models, this could be very specific training parameters such as whether to enable bootstrapping, the precise backend to use for computations etc.
    """

class ModelHyperParameters(BaseModel):
    """Hyperparameters for a Model in the TSUT Framework.

    This will usually be used for parameters that are relevant for the definition of the model itself, and that are relevant to be tuned during hyperparameter tuning.
    For example, in most ML models, this could be the learning rate, the number of layers, etc.
    """

class ModelConfig(NodeConfig):
    """Base configuration for all Models in the TSUT Framework."""

    node_type: NodeType = NodeType.MODEL
    running_config: ModelRunningConfig = ModelRunningConfig()
    hyperparameters: ModelHyperParameters = ModelHyperParameters()



class Model[D_I, D_O, P](
    Node[D_I, D_O], ABC
):  # Model is already implicitely an ABC via Node but explicit is better.
    """Base class for all models in the TSUT Framework."""

    metadata = ModelMetadata()

    def __init__(self, *, config: ModelConfig) -> None:
        """Minimal constructor for Model class."""
        self._config = config

    # --- Abstract Methods to reimplement ---

    @abstractmethod
    def fit(self, data: D_I) -> None:
        """Fit the model with the given data."""
        ...

    @abstractmethod
    def predict(
        self, data: D_I
    ) -> D_O:
        """Predict using the model with the given data."""
        ...

    @abstractmethod
    def get_params(self) -> P:
        """Get the model parameters."""
        ...

    @abstractmethod
    def set_params(self, params: P) -> None:
        """Set the model parameters."""
        ...

    # --- API convenience ---

    @property
    def running_config(self) -> ModelRunningConfig:
        """Property to get the model running configuration."""
        return self._config.running_config

    @property
    def hyperparameters(self) -> ModelHyperParameters:
        """Property to get the model hyperparameters."""
        return self._config.hyperparameters

    @property
    def parameters(self) -> P:
        """Property to get the model parameters."""
        return self.get_params()

    @property
    def config(self) -> ModelConfig:
        """Property to get the full model configuration."""
        return self._config

    # --- Implementations for Node interface, don't touch without a very good reason ---

    def node_fit(self, data: D_I) -> None:
        """Override of the Node's fit method to call the Model's fit method."""
        return self.fit(data=data)

    def node_transform(
        self, data: D_I
    ) -> D_O:
        """Override of the Node's transform method to call the Model's predict method."""
        return self.predict(data=data)
