from typing import TypeGuard

from pydantic import BaseModel

from tsut.core.common.typechecking.protocols import (
    AcceptsInputsSourceNode,
    HasHyperparametersConfig,
    HasHyperparametersNode,
    HasHyperparameterSpace,
    HasParamsNode,
    HasRunningConfigConfig,
    HasRunningConfigNode,
)
from tsut.core.nodes.data_sink.sink import Sink
from tsut.core.nodes.node import Node, NodeConfig


# has Hyperparameters typeguards
def has_hyperparameters(obj: Node) -> TypeGuard[HasHyperparametersNode]:
    """Typeguard to check if a Node has hyperparameters in its configuration.

    We check if the Node's configuration has a 'hyperparameters' attribute and that it is not None or an empty BaseModel.
    """
    hp = getattr(obj.config, "hyperparameters", None)
    # Check if hp exists and is not None
    if hp is None:
        return False
    # Check if hp is a BaseModel and is not empty
    if len(hp.model_fields) == 0:
        return False
    # If hp is a BaseModel, check if it is not empty by checking if its model_dump() is not empty
    return not (isinstance(hp, BaseModel) and not hp.model_dump())


def has_hyperparameters_config(obj: NodeConfig) -> TypeGuard[HasHyperparametersConfig]:
    """Typeguard to check if a Node's configuration has hyperparameters."""
    hp = getattr(obj, "hyperparameters", None)
    # Check if hp exists and is not None
    if hp is None:
        return False
    # Check if hp is a BaseModel and is not empty
    if len(hp.model_fields) == 0:
        return False
    # If hp is a BaseModel, check if it is not empty by checking if its model_dump() is not empty
    return isinstance(hp, BaseModel) and bool(hp.model_dump())


# has RunningConfig typeguards
def has_running_config(obj: Node) -> TypeGuard[HasRunningConfigNode]:
    """Typeguard to check if a Node has a running configuration in its configuration."""
    rc = getattr(obj.config, "running_config", None)
    # Check if rc exists and is not None
    if rc is None:
        return False
    # Check if rc is a BaseModel and is not empty
    if len(rc.model_fields) == 0:
        return False
    # If rc is a BaseModel, check if it is not empty by checking if its model_dump() is not empty
    return isinstance(rc, BaseModel) and bool(rc.model_dump())


def has_running_config_config(obj: NodeConfig) -> TypeGuard[HasRunningConfigConfig]:
    """Typeguard to check if a Node's configuration has a running configuration."""
    rc = getattr(obj, "running_config", None)
    # Check if rc exists and is not None
    if rc is None:
        return False
    # Check if rc is a BaseModel and is not empty
    if len(rc.model_fields) == 0:
        return False
    # If rc is a BaseModel, check if it is not empty by checking if its model_dump() is not empty
    return isinstance(rc, BaseModel) and bool(rc.model_dump())


# has Hyperparameter_space typeguards
def has_hyperparameter_space(obj: Node) -> TypeGuard[HasHyperparameterSpace]:
    """Typeguard to check if a Node has a hyperparameter space."""
    return hasattr(obj, "hyperparameter_space") and isinstance(
        obj.hyperparameter_space,  # type: ignore
        dict,
    )


# has Params typeguards
def has_params(obj: Node) -> TypeGuard[HasParamsNode]:
    """Typeguard to check if a Node has parameters."""
    return (
        hasattr(obj, "get_params")
        and callable(obj.get_params)  # type: ignore
        and hasattr(obj, "set_params")
        and callable(obj.set_params)  # type: ignore
    )


# is Sink Node typeguard
def is_sink_node(obj: Node) -> TypeGuard[Sink]:
    """Typeguard to check if a Node is a Sink Node."""
    return obj.config.node_type == "SINK" and isinstance(obj, Sink)


# is List typeguard
def is_list(obj: object) -> TypeGuard[list]:
    """Typeguard to check if an object is a list."""
    return isinstance(obj, list)


# accepts inputs typeguard
def accepts_inputs_source_node(node: Node) -> TypeGuard[AcceptsInputsSourceNode]:
    """Typeguard to check if a Node accepts inputs (i.e., is not a pure data source)."""
    if hasattr(node, "accepts_inputs") and node.config.node_type == "SOURCE":
        return node.accepts_inputs  # type: ignore
    return False
