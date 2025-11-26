"""Define the atomic component of a TSUT Pipeline: the Node."""

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel

from tsut.core.common.data.types import ContextData, Data

D_I = TypeVar("D_I", bound=Data)
D_O = TypeVar("D_O", bound=Data)


class NodeType(StrEnum):
    """Define the types of Nodes available in a TSUT Pipeline."""

    BASE = "base"
    SOURCE = "source"
    SINK = "sink"
    TRANSFORM = "transform"
    MODEL = "model"


class Port(BaseModel):
    """Model of a port in a TSUT Node."""

    type: type
    desc: str


class NodeConfig(BaseModel):
    """Configuration for a Node in a TSUT Pipeline."""

    node_type: NodeType = NodeType.BASE
    in_ports: dict[str, Port] = {}
    out_ports: dict[str, Port] = {}


class Node[D_I, D_O](ABC):
    """Base class for a Node in a TSUT Pipeline."""

    _is_node: bool = True

    def __init__(self, *, config: NodeConfig) -> None:
        """Initialize the Node with the given configuration.

        The only common denomination across all Nodes are their input and output ports.
        """
        self.node_type: NodeType = config.node_type
        self.in_ports: dict[str, Port] = config.in_ports
        self.out_ports: dict[str, Port] = config.out_ports

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        """Ensure that all subclasses of Node are placed before Mixin classes in the inheritance order. (To ensure proper initialization order)."""
        super().__init_subclass__(**kwargs)

        bases = cls.__bases__
        base_idx = None
        for idx, base in enumerate(bases):
            if getattr(base, "_is_node", False):
                base_idx = idx
        if base_idx is None:
            return

        for idx, base in enumerate(bases):
            if getattr(base, "_is_mixin", False) and idx < base_idx:
                message = f"Mixin class {base.__name__} must be placed after the base class in the inheritance order.\n"
                message += f"Use 'class {cls.__name__}({bases[base_idx].__name__}, {base.__name__}):' instead of 'class {cls.__name__}({base.__name__}, {bases[base_idx].__name__}):'."
                raise TypeError(message)

    @abstractmethod
    def node_fit(self, data: dict[str, D_I | ContextData] | None) -> None:
        """Define the base logic for fitting a Node with the given data. Can also be called with  no data to implement setup logic."""
        raise NotImplementedError

    @abstractmethod
    def node_transform(
        self, data: dict[str, D_I | ContextData]
    ) -> dict[str, D_O | ContextData]:
        """Define the base logic for transforming data through the Node."""
        raise NotImplementedError

    def node_fit_transform(
        self, data: dict[str, D_I | ContextData]
    ) -> dict[str, D_O | ContextData]:
        """Define the base logic for fitting and transforming data through the Node."""
        self.node_fit(data)
        return self.node_transform(data)
