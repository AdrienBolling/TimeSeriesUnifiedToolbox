"""A module to define all Protocol used for typechecking and custom Typeguard checks in the TSUT Framework."""

from typing import Any, Protocol

from pydantic import BaseModel


# --------------
# has Hyperparameters Protocols
class HasHyperparametersConfig(Protocol):
    """Protocol for objects that have a configuration with hyperparameters."""

    hyperparameters: BaseModel

    def model_copy(self, update: dict[str, Any]) -> BaseModel: ...


class HasHyperparametersNode(Protocol):
    """Protocol for nodes that have hyperparameters in their configuration."""

    @property
    def config(self) -> HasHyperparametersConfig: ...


# --------------
# has Params Protocols
class HasParamsNode(Protocol):
    """Protocol for objects that have parameters."""

    def get_params(self) -> dict[str, Any]: ...
    def set_params(self, params: dict[str, Any]) -> None: ...


# --------------
# has RunningConfig Protocols
class HasRunningConfigConfig(Protocol):
    """Protocol for objects that have a configuration with a running configuration."""

    running_config: BaseModel


class HasRunningConfigNode(Protocol):
    """Protocol for nodes that have a running configuration."""

    @property
    def config(self) -> HasRunningConfigConfig: ...


# --------------
# has Hyperparameter_space Protocols
class HasHyperparameterSpace(Protocol):
    """Protocol for objects that have a hyperparameter space."""

    hyperparameter_space: dict[str, Any]


# --------------
# accepts_inputs Protocols
class AcceptsInputsSourceNode(Protocol):
    """Protocol for source nodes that have an 'accepts_inputs' attribute to indicate whether they can accept inputs from other nodes (i.e., are not pure data sources)."""

    accepts_inputs: bool
