"""Define the atomic component of a TSUT Pipeline: the Node."""

import uuid
from abc import ABC, ABCMeta, abstractmethod
from enum import StrEnum
from typing import Any, ParamSpec, TypeVar, override

from pydantic import BaseModel, PrivateAttr

from tsut.core.common.data.types import ContextData, Data
from tsut.core.common.mixins.base import MixinSettings

D_I = TypeVar("D_I", bound=Data)
D_O = TypeVar("D_O", bound=Data)

P = ParamSpec("P")


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

    _id: uuid.UUID = PrivateAttr(uuid.uuid4())
    node_type: NodeType = NodeType.BASE
    in_ports: dict[str, Port] = {}
    out_ports: dict[str, Port] = {}
    mixins_config: dict[str, MixinSettings] = {}

    @override
    def __hash__(self) -> int:
        """Hash the NodeConfig based on its UUID. Assume that two Nodes with the same config can be considered different."""
        return hash(self._id)

    @property
    def id(self) -> uuid.UUID:
        """Property to get the unique identifier of the NodeConfig."""
        return self._id


class MetaPostInitHook(ABCMeta):
    """Metaclass to add an automatic call to a '__post_init__' method after '__init__'."""

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Add automatic call to 'post_init' hook after '__init__'."""
        # Before we do __new__ and __init__
        instance: object = super().__call__(*args, **kwargs)
        # After we do __post_init__
        post_hook = getattr(instance, "__post_init__", None)
        if callable(post_hook):
            _ = post_hook(*args, **kwargs)
        return instance


class Node[D_I, D_O](ABC, metaclass=MetaPostInitHook):
    """Base class for a Node in a TSUT Pipeline."""

    _is_node: bool = True

    def __init__(self, *, config: NodeConfig) -> None:
        """Initialize the Node with the given configuration.

        The only common denomination across all Nodes are their input and output ports.
        """
        self.node_type: NodeType = config.node_type
        self.in_ports: dict[str, Port] = config.in_ports
        self.out_ports: dict[str, Port] = config.out_ports

    def __init_subclass__(cls, *args: ParamSpec, **kwargs: ParamSpec) -> None:
        """Ensure several things.

        - That all subclasses of Node are placed before Mixin classes in the inheritance order. (To ensure proper initialization order).
        """
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

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        """Define common post-initialization hook for all Nodes.

        If 'config' is passed as kwarg, set it as the Node's '_config' attribute.
        Ensure a call to super().init() with the proper args.
        """
        if "config" not in kwargs:
            message = "Node must be initialized with a 'config' keyword argument, subtype of 'NodeConfig'."
            raise ValueError(message)
        self._config: NodeConfig = kwargs["config"]
        super().__init__(*args, **kwargs)
        # If super has a post_init, call it
        if getattr(
            super(), "__post_init__", None
        ):  # INFO: Ignore the error here if pyright raises one, it's being dumb
            super().__post_init__(*args, **kwargs)
            # INFO: Although it may seem useless, it's used in case of multiple inheritance with Mixins and such

    @property
    def id(self) -> uuid.UUID:
        """Get the unique identifier of the Node based on its configuration."""
        return self._config.id

    @abstractmethod
    def node_fit(self, data: dict[str, D_I | ContextData]) -> None:
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
