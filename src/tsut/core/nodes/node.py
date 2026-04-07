"""Define the atomic component of a TSUT Pipeline: the Node."""

import uuid
from abc import ABC, ABCMeta, abstractmethod
from enum import StrEnum
from functools import wraps
from typing import Any, ParamSpec, override

from pydantic import BaseModel, PrivateAttr

from tsut.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
)
from tsut.core.common.mixins.mixin import MixinSettings

P = ParamSpec("P")


class NodeMetadata(BaseModel):
    """Metadata for a Node in a TSUT Pipeline."""


class NodeType(StrEnum):
    """Define the types of Nodes available in a TSUT Pipeline."""

    BASE = "base"
    SOURCE = "source"
    SINK = "sink"
    TRANSFORM = "transform"
    MODEL = "model"
    METRIC = "metric"


class Port(BaseModel):
    """Model of a port in a TSUT Node."""

    arr_type: ArrayLikeEnum  # The type of data array that this port accepts or outputs, e.g. pd.DataFrame, np.ndarray, etc.
    data_structure: DataStructureEnum = (
        DataStructureEnum.DATA  # The data structure that this port accepts or outputs, e.g. TabularData, TimeSeriesData, etc.
    )
    data_category: DataCategoryEnum  # The category of data that this port accepts or outputs, e.g. "numerical", "categorical", "mixed", etc.
    data_shape: str  # the shape of the data that this ports accepts or outputs for jaxtyping checking. Find the convention for the data shape string at https://docs.kidger.site/jaxtyping/api/array/#shape
    optional: bool = False  # Whether this port is optional or not. Optional ports can be left unconnected in the pipeline, and the node should be able to handle that case gracefully (e.g., by using default values or by skipping certain computations).
    desc: str
    mode: list[str] = ["all"]


class NodeConfig(BaseModel):
    """Configuration for a Node in a TSUT Pipeline."""

    _id: uuid.UUID = PrivateAttr(default_factory=uuid.uuid4)
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


class Node[D_I, D_C_I, D_O, D_C_O](ABC, metaclass=MetaPostInitHook):
    """Base class for a Node in a TSUT Pipeline."""

    _is_node: bool = True
    metadata = NodeMetadata()

    def __init__(self, *, config: NodeConfig) -> None:
        """Initialize the Node with the given configuration.

        The only common denomination across all Nodes are their input and output ports.
        """
        if not self._config:
            self._config = config

    @property
    def in_ports(self) -> dict[str, Port]:
        """Get the input ports of the Node."""
        return self._config.in_ports

    @property
    def out_ports(self) -> dict[str, Port]:
        """Get the output ports of the Node."""
        return self._config.out_ports

    @property
    def node_type(self) -> NodeType:
        """Get the type of the Node."""
        return self._config.node_type

    @property
    def config(self) -> NodeConfig:
        """Get the configuration of the Node."""
        return self._config

    def __init_subclass__(cls, *args: ParamSpec, **kwargs: ParamSpec) -> None:
        """Ensure several things.

        - That all subclasses of Node are placed before Mixin classes in the inheritance order. (To ensure proper initialization order).
        """
        super().__init_subclass__(**kwargs)

        orig_init = cls.__init__

        @wraps(orig_init)
        def wrapped_init(self, *args, **kwargs):
            if "config" not in kwargs:
                raise TypeError("config must be passed as a keyword argument")
            self._config = kwargs["config"]
            orig_init(self, *args, **kwargs)

        cls.__init__ = wrapped_init

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

    def __post_init__(self, *, config=None) -> None:
        """Define common post-initialization hook for all Nodes.

        If 'config' is passed as kwarg, set it as the Node's '_config' attribute.
        Ensure a call to super().init() with the proper args.
        """
        try:
            super().__init__(config=config)  # type: ignore
        except TypeError:  # In case we go back to object, object.__init__ doesn't take any argument, so we need to catch the TypeError and call it without arguments
            super().__init__()  # type: ignore
        # If super has a post_init, call it
        if getattr(
            super(), "__post_init__", None
        ):  # INFO: Ignore the error here if pyright raises one, it's being dumb
            super().__post_init__(config=config)  # type: ignore
            # INFO: Although it may seem useless, it's used in case of multiple inheritance with Mixins and such

    @property
    def id(self) -> uuid.UUID:
        """Get the unique identifier of the Node based on its configuration."""
        return self._config.id

    @abstractmethod
    def node_fit(self, data: dict[str, tuple[D_I, D_C_I]]) -> None:
        """Define the base logic for fitting a Node with the given data. Can also be called with  no data to implement setup logic."""
        raise NotImplementedError

    @abstractmethod
    def node_transform(
        self, data: dict[str, tuple[D_I, D_C_I]]
    ) -> dict[str, tuple[D_O, D_C_O]]:
        """Define the base logic for transforming data through the Node."""
        raise NotImplementedError

    def node_fit_transform(
        self, data: dict[str, tuple[D_I, D_C_I]]
    ) -> dict[str, tuple[D_O, D_C_O]]:
        """Define the base logic for fitting and transforming data through the Node."""
        self.node_fit(data)
        return self.node_transform(data)
