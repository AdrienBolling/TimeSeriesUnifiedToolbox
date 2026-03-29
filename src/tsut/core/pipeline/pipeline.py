"""Define the base classes for TSUT Pipelines.

TSUT Pipelines are akin to graphs. The nodes are the components of the pipeline (Models, Data sources, Transforms, etc.), and edges represent the data flow between these nodes.
"""

from collections.abc import Callable
from functools import wraps
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from tsut.core.common.data.data import Data
from tsut.core.nodes.metrics.metric_node import MetricNodeConfig
from tsut.core.nodes.node import Node, NodeConfig, NodeType, Port
from tsut.core.nodes.registry.node_registry import NODE_REGISTRY
from tsut.core.pipeline.render import plot_graph_with_metadata

_ = Data  # For type checking purposes


def decompile(method: Callable[..., Any]):
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
    ports_map: dict[str, str]  # Input port of target node to output port of source node

class PipelineConfig(BaseModel):
    """Define the global configuration for a TSUT Pipeline (Not the layout)."""

    nodes: dict[str, tuple[str, NodeConfig | None]] = {}
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

        # --- Initialize the core attributes of the Pipeline
        # Will be updated during compilation of the pipeline, and used during execution by the runners.
        self.node_objects: dict[str, Node] = {}
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

    # --- Convenience API
    @property
    def config(self) -> PipelineConfig:
        """Get the pipeline configuration."""
        return self._config

    @property
    def settings(self) -> PipelineSettings:
        """Get the pipeline settings."""
        return self._config.settings

    @property
    def nodes(self) -> dict[str, tuple[str, NodeConfig | None]]:
        """Get the pipeline nodes configuration."""
        return self._config.nodes

    @property
    def edges(self) -> list[Edge]:
        """Get the pipeline edges configuration."""
        return self._config.edges

    @property
    def graph(self) -> nx.DiGraph:
        """Get the internal graph representation of the pipeline."""
        return self._graph

    @property
    def graph_wo_metrics(self) -> nx.DiGraph:
        """Get the internal graph representation of the pipeline without metric edges."""
        def filter_node(n: str) -> bool:
            # Filter out metric nodes
            return self.config.nodes[n][1].node_type != "metric"

        def filter_edge(u: str, v: str) -> bool:
            # Filter out edges connected to metric nodes
            return not filter_node(u) or not filter_node(v)
        return nx.subgraph_view(self.graph, filter_node=filter_node, filter_edge=filter_edge)

    @property
    def compiled(self) -> bool:
        """Check if the pipeline is compiled.

        Returns:
            bool: True if the pipeline is compiled, False otherwise.

        """
        return self._compiled

    # --- Public API for pipeline management (adding/removing nodes and edges, compiling, etc.)

    def add_node(self, node_class_name: str,node_name: str, node_config: NodeConfig | None) -> None:
        """Add a node to the pipeline."""
        if node_name in self._config.nodes:
            message = f"Node with name '{node_name}' already exists in the pipeline."
            raise ValueError(message)
        self._config.nodes[node_name] = (node_class_name, node_config)
        self._add_node_to_repr(node_name, node_class_name, node_config)

    def remove_node(self, node_name: str) -> None:
        """Remove a node from the pipeline."""
        if node_name not in self._config.nodes:
            message = f"Node with name '{node_name}' does not exist in the pipeline."
            raise ValueError(message)
        del self._config.nodes[node_name]
        self._remove_node_from_repr(node_name)

    def add_edge(self, source: str, target: str, ports_map: dict[str, str]) -> None:
        """Add an edge to the pipeline."""
        edge = Edge(source=source, target=target, ports_map=ports_map)
        self._config.edges.append(edge)
        self._add_edge_to_repr(edge)

    def add_edge_from_object(self, edge: Edge) -> None:
        """Add an edge to the pipeline from an Edge object."""
        self._config.edges.append(edge)
        self._add_edge_to_repr(edge)

    def remove_edge(self, source: str, target: str) -> None:
        """Remove an edge from the pipeline."""
        edge_to_remove = None
        for edge in self._config.edges:
            if edge.source == source and edge.target == target:
                edge_to_remove = edge
                break
        if edge_to_remove is None:
            message = f"Edge from '{source}' to '{target}' does not exist in the pipeline."
            raise ValueError(message)
        self._config.edges.remove(edge_to_remove)
        self._remove_edge_from_repr(source, target)

    def get_edge(self, source: str, target: str) -> Edge:
        """Get an edge from the pipeline."""
        for edge in self._config.edges:
            if edge.source == source and edge.target == target:
                return edge
        message = f"Edge from '{source}' to '{target}' does not exist in the pipeline."
        raise ValueError(message)

    def update_node_config(self, node_name: str, new_config: NodeConfig) -> None:
        """Update the configuration of a node in the pipeline."""
        if node_name not in self._config.nodes:
            message = f"Node with name '{node_name}' does not exist in the pipeline."
            raise ValueError(message)
        node_class_name, _ = self._config.nodes[node_name]
        self._config.nodes[node_name] = (node_class_name, new_config)
        # Update the node attributes in the graph representation
        self._graph.nodes[node_name].update(new_config.model_dump())

    def render(self, title: str = "Pipeline Graph", layout: str = "spring") -> Any:
        """Render the pipeline graph using iplotx.

        Args:
            title (str): The title of the graph.
            layout (str): The layout algorithm to use for positioning the nodes. Options include "spring", "kamada_kawai", etc.

        Returns:
            The rendered graph object from iplotx.

        """
        fig, ax, artist = plot_graph_with_metadata(
            G=self._graph,
            node_objects=self.node_objects,
            title=title,
            layout=layout,
        )
        plt.plot

    def get_params(self) -> dict[str, dict[str, Any]]:
        """Get the aggregate of all parameters of the nodes in the pipeline"""
        params = {}
        for node_name, node in self.node_objects.items():
            params[node_name] = node.get_params() if hasattr(node, "get_params") else {}
        return params

    def set_params(self, params: dict[str, dict[str, Any]]) -> None:
        """Set the parameters of the nodes in the pipeline.

        Args:
            params (dict[str, dict[str, Any]]): A dictionary where keys are node names and values are dictionaries of parameters to set for each node.

        Returns:
            None

        """
        for node_name, node_params in params.items():
            if node_name not in self.node_objects:
                message = f"Node with name '{node_name}' does not exist in the pipeline."
                raise ValueError(message)
            node = self.node_objects[node_name]
            if hasattr(node, "set_params"):
                node.set_params(**node_params)
            else:
                if node_params == {}:
                    continue  # No parameters to set, skip
                message = f"Node '{node_name}' does not have a 'set_params' method, cannot set parameters {node_params}."
                raise ValueError(message)

    def get_sink_node_name(self) -> str:
        """Get the name of the sink node in the pipeline."""
        for node_name, (_, node_config) in self._config.nodes.items():
            if node_config.node_type == NodeType.SINK:
                return node_name
        message = "No sink node found in the pipeline."
        raise ValueError(message)

    # --- Internal methods for object instantiation

    def _instantiate_nodes(self) -> None:
        """Instantiate node objects from the pipeline configuration."""
        for node_name, (node_class_name, node_config) in self._config.nodes.items():
            node_class = NODE_REGISTRY[node_class_name]["node_class"]
            self.node_objects[node_name] = node_class(config=node_config)

    def _instantiate_sink_ports(self) -> None:
        """Instantiate sink ports by connecting them to the appropriate nodes based on the edges configuration."""
        incoming_edges_idx: list[int] = [] # List of indices of edges that are incoming to sink nodes, used to identify which nodes are connected to the sink ports.
        for idx, edge in enumerate(self._config.edges):
            if edge.target in self._config.nodes and self._config.nodes[edge.target][1].node_type == NodeType.SINK:
                incoming_edges_idx.append(idx)

        for idx in incoming_edges_idx:
            edge = self._config.edges[idx]
            sink_node_name = edge.target
            source_node_name = edge.source
            # Connect the source node's output port to the sink node's "dump" input port
            original_port = edge.ports_map["dump"]
            self._config.edges[idx].ports_map = {f"{source_node_name}:{original_port}": original_port}
            # Add this port to the sink node's ports
            self.node_objects[sink_node_name].add_port(port_name=f"{source_node_name}:{original_port}")
    # --- Internal methods for graph management

    @decompile
    def _init_graph_repr(self) -> None:
        """Initialize the internal graph representation using networkx."""
        self._graph: nx.DiGraph[str] = nx.DiGraph()
        self._compiled = False

    @decompile
    def _init_metric_edges(self) -> None:
        """Initialize metric edges in the internal graph representation."""
        for edge in self._config.metric_edges:
            self._add_edge_to_repr(edge)

    @decompile
    def _init_nodes(self) -> None:
        """Initialize nodes in the internal graph representation."""
        for node_name, (node_class_name, node_config) in self._config.nodes.items():
            self._add_node_to_repr(node_name, node_class_name, node_config)

    @decompile
    def _init_edges(self) -> None:
        """Initialize edges in the internal graph representation."""
        for edge in self._config.edges:
            self._add_edge_to_repr(edge)

    @decompile
    def _prune(self) -> None:
        """Prune unused nodes from the pipeline."""
        # Find nodes with no incoming or outgoing edges
        isolates: list[str] = list(nx.isolates(self._graph))
        for n in isolates:
            self._graph.remove_node(n)
        isolates_idx = [self.node_objects[node_name].id for node_name in isolates]
        for node_name, (_, node_config) in list(self._config.nodes.items()):
            if node_config.id in isolates_idx:
                # Remove from the config
                del self._config.nodes[node_name]
                # Remove from the objects
                del self.node_objects[node_name]

    def _add_node_to_repr(self, node_name: str, node_class_name: str, node_config: NodeConfig | None) -> None:
        """Add a node to the internal graph representation."""
        if node_config is None:
            node_config = NODE_REGISTRY[node_class_name]["node_config_class"]()  # Use default config if None is provided
            self._config.nodes[node_name] = (node_class_name, node_config)  # Update the config with the default config
        features = node_config.model_dump() # TODO : Maybe not all features need to be added as node attributes, since its mostly display purposes.
        features.update({"node_class_name": node_class_name})
        self._graph.add_node(node_for_adding=node_name, **features)

    def _add_edge_to_repr(self, edge: Edge) -> None:
        """Add an edge to the internal graph representation."""
        self._graph.add_edge(
            u_of_edge=edge.source,
            v_of_edge=edge.target,
            ports_map=edge.ports_map,
        )

    def _add_metric_edge_to_repr(self, edge: Edge) -> None:
        """Add a metric edge to the internal graph representation."""
        self._graph.add_edge(
            u_of_edge=edge.source,
            v_of_edge=edge.target,
            ports_map=edge.ports_map,
            is_metric_edge=True,  # Add an attribute to differentiate metric edges from regular edges
        )

    def remove_metric_edge_from_repr(self, source: str, target: str) -> None:
        """Remove a metric edge from the internal graph representation."""
        if self._graph.has_edge(source, target):
            if self._graph.edges[source, target].get("is_metric_edge", False):
                self._graph.remove_edge(u=source, v=target)
            else:
                message = f"Edge from '{source}' to '{target}' exists but is not a metric edge."
                raise ValueError(message)
        else:
            message = f"Edge from '{source}' to '{target}' does not exist in the pipeline."
            raise ValueError(message)

    def _remove_node_from_repr(self, node_name: str) -> None:
        """Remove a node from the internal graph representation."""
        self._graph.remove_node(node_name)

    def _remove_edge_from_repr(self, source: str, target: str) -> None:
        """Remove an edge from the internal graph representation."""
        self._graph.remove_edge(u=source, v=target)

    # --- Validation methods

    def _validate(self) -> None:
        """Validate the pipeline layout, ensure the graph is directed and acyclic."""
        valid = nx.is_directed_acyclic_graph(self._graph)
        # Check that there's only one sink node
        sink_nodes = [n for n, n_conf in self._config.nodes.items() if n_conf[1].node_type == NodeType.SINK]
        if len(sink_nodes) > 1:
            valid = False
            message = f"Pipeline has multiple sink nodes: {sink_nodes}. Only one sink node is allowed."
            raise ValueError(message)
        if len(sink_nodes) == 0:
            valid = False
            message = f"Pipeline has no sink node. At least one sink node is required."
            raise ValueError(message)
        if not valid:
            message = "The pipeline graph must be a directed acyclic graph (DAG).\n"
            message += (
                "Please check the nodes and edges for cycles or invalid connections."
            )
            raise ValueError(message)

    # --- Compilation methods

    def compile(self) -> None:
        """Compile the pipeline, marking it as ready for execution. This is the ONLY way self._compiled can be set to True, and it will raise an error if the pipeline is not valid."""
        # Clear the node_objects dict to ensure a clean slate for compilation, and re-instantiate the nodes from the config.
        self.node_objects = {}

        if self.settings.autoprune:
            self._prune()
        self._validate()

        # If everything is valid, instantiate the nodes from the config to prepare for execution.
        self._instantiate_nodes()

        # Instantiate sink ports
        self._instantiate_sink_ports()

        self._compiled = True

    def uncompile(self) -> None:
        """Uncompile the pipeline, marking it as needing recompilation."""
        self._compiled = False
