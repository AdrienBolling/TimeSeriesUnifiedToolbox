"""Define the base classes for TSUT Pipelines.

TSUT Pipelines are akin to graphs. The nodes are the components of the pipeline (Models, Data sources, Transforms, etc.), and edges represent the data flow between these nodes.
"""

from collections.abc import Callable
from functools import _Wrapped, wraps
from typing import Any

import networkx as nx
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from tsut.core.common.data.types import Data
from tsut.core.nodes.base import Node, NodeConfig

_ = Data  # For type checking purposes


def decompile(method: Callable[..., Any]) -> _Wrapped[..., Any, ..., Any]:
    """Return a decorator that marks a method to decompile the pipeline on modification."""

    @wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(self, Pipeline):
            message = f"Expected 'self' to be an instance of Pipeline, got {type(self)} instead."
            raise TypeError(message)
        result: Any = method(self, *args, **kwargs)
        self.uncompile()
        return result

    return wrapper


class PipelineSettings(BaseSettings):
    """Settings for a TSUT Pipeline."""

    autoprune: bool = True
    validate_on_init: bool = True
    validate_on_modification: bool = True


class Edge(BaseModel):
    """Define a node-to-node connection in a Pipeline."""

    source: str
    target: str
    ports_map: dict[str, str]  # Output port of source node to input port of target node


class PipelineConfig(BaseModel):
    """Define the global configuration for a TSUT Pipeline (Not the layout)."""

    nodes: dict[str, NodeConfig] = {}
    edges: list[Edge] = []
    settings: PipelineSettings = PipelineSettings()


class Pipeline:
    """Base class for a TSUT Pipeline.

    A Pipeline is a directed graph where nodes represent processing units (like models, data sources, transforms) and edges represent the flow of data between these nodes.
    """

    def __init__(
        self,
        *,
        config: PipelineConfig | None = None,
    ) -> None:
        """Initialize the Pipeline with the given configuration.

        Description:
            More often than not, pipelines will be either initiliazed with no nodes/edges, and will be built programmatically or loaded from a file later. In such cases, the default empty PipelineConfig will be used.
            Compile the pipeline before using it to effectively initialize all nodes.

        Args:
            config (PipelineConfig, optional): Global configuration for the pipeline.

        Returns:
            None

        """
        # The compile state of the pipeline. Can't be set to True from the outside. Resets to False on modification.
        self._compiled: bool = False

        self._config: PipelineConfig = (
            config if config is not None else PipelineConfig()
        )

        # Initialize the core attributes of the Pipeline
        self.node_objects: dict[str, Node[Data, Data]] = {}
        # Create the inner nx graph for representation and graph-related operations
        self._init_graph_repr()

        # Check if there are any nodes, if so initialize them
        if self._config.nodes:
            self._init_nodes()
        # Check if there are any edges, if so initialize them
        if self._config.edges:
            self._init_edges()

        # Validate the pipeline if specified in settings
        if self._config.settings.validate_on_init:
            self._validate()

    @decompile
    def _init_graph_repr(self) -> None:
        """Initialize the internal graph representation using networkx."""
        self._graph: nx.DiGraph[str] = nx.DiGraph()
        self._compiled = False

    @decompile
    def _init_nodes(self) -> None:
        """Initialize nodes in the internal graph representation."""
        for node_name, node_config in self._config.nodes.items():
            self._add_node_to_repr(node_name, node_config)

    @decompile
    def _init_edges(self) -> None:
        """Initialize edges in the internal graph representation."""
        for edge in self._config.edges:
            self._add_edge_to_repr(edge)

    def _validate(self) -> None:
        """Validate the pipeline layout, ensure the graph is directed and acyclic."""
        valid = nx.is_directed_acyclic_graph(self._graph)
        if not valid:
            message = "The pipeline graph must be a directed acyclic graph (DAG).\n"
            message += (
                "Please check the nodes and edges for cycles or invalid connections."
            )
            raise ValueError(message)

    @decompile
    def _prune(self) -> None:
        """Prune unused nodes from the pipeline."""
        # Find nodes with no incoming or outgoing edges
        isolates: list[str] = list(nx.isolates(self._graph))
        for n in isolates:
            self._graph.remove_node(n)
        isolates_idx = [self.node_objects[node_name].id for node_name in isolates]
        for node_name, node_config in list(self._config.nodes.items()):
            if node_config.id in isolates_idx:
                # Remove from the config
                del self._config.nodes[node_name]
                # Remove from the objects
                del self.node_objects[node_name]

    def _add_node_to_repr(self, node_name: str, node_config: NodeConfig) -> None:
        """Add a node to the internal graph representation."""
        self._graph.add_node(node_for_adding=node_name, **node_config.model_dump())

    def _add_edge_to_repr(self, edge: Edge) -> None:
        """Add an edge to the internal graph representation."""
        self._graph.add_edge(
            u_of_edge=edge.source,
            v_of_edge=edge.target,
            ports_map=edge.ports_map,
        )

    def _remove_node_from_repr(self, node_name: str) -> None:
        """Remove a node from the internal graph representation."""
        self._graph.remove_node(node_name)

    def _remove_edge_from_repr(self, source: str, target: str) -> None:
        """Remove an edge from the internal graph representation."""
        self._graph.remove_edge(u=source, v=target)

    def compiled(self) -> bool:
        """Check if the pipeline is compiled.

        Returns:
            bool: True if the pipeline is compiled, False otherwise.

        """
        return self._compiled

    def uncompile(self) -> None:
        """Uncompile the pipeline, marking it as needing recompilation."""
        self._compiled = False
