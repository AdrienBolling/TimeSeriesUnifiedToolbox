"""Define the base classes for TSUT Pipelines.

TSUT Pipelines are akin to graphs. The nodes are the components of the pipeline (Models, Data sources, Transforms, etc.), and edges represent the data flow between these nodes.
"""

from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any

import networkx as nx
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from tsut.core.common.data.data import (
    DATA_CATEGORY_MAPPING,
    DATA_STRUCTURE_MAPPING,
    Data,
)
from tsut.core.common.enums import NodeExecutionMode
from tsut.core.common.typechecking.typeguards import has_params
from tsut.core.common.version import Version
from tsut.core.nodes.node import Node, NodeConfig, NodeType, Port
from tsut.core.nodes.registry.node_registry import NODE_REGISTRY
from tsut.core.pipeline.render import render_pipeline_graph_plotly

_ = Data  # For type checking purposes


def decompile(method: Callable[..., Any]):  # noqa: ANN201
    """Return a decorator that marks a method to decompile the pipeline on modification."""

    @wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(self, Pipeline):
            message = f"Expected 'self' to be an instance of Pipeline, got {type(self)} instead."
            raise TypeError(message)
        result: Any = method(self, *args, **kwargs)
        self._uncompile()
        return result

    return wrapper


# TODO : use Fields to both imrpove validation and provide descriptions to the Models


class PipelineSettings(BaseSettings):
    """Settings for a TSUT Pipeline."""

    autoprune: bool = True


class Edge(BaseModel):
    """Define a node-to-node connection in a Pipeline."""

    source: str
    target: str
    ports_map: list[
        tuple[str, str]
    ]  # Mappings of (source port, target port), if the target is the Sink node these will be ignored


class PipelineConfig(BaseModel):
    """Define the global configuration for a TSUT Pipeline (Not the layout)."""

    nodes: Mapping[str, tuple[str, NodeConfig | None]] = {}
    edges: list[Edge] = []
    name: str = "My Pipeline"
    version: Version = Version(major=0, minor=1, patch=0)
    settings: PipelineSettings = PipelineSettings()


class _InternalPipelineConfig(BaseModel):
    """Internal configuration for a TSUT Pipeline.

    This is used to store additional information that is not part of the user-facing PipelineConfig, such as the compiled graph structure.
    """

    # Compiled graph structure
    nodes: dict[str, tuple[str, NodeConfig]] = {}
    edges: dict[tuple[str, str], Edge] = {}
    name: str = "My Pipeline"
    version: Version = Version(major=0, minor=1, patch=0)
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
        # Initialize the pipeline internals
        self._compiled = False
        user_provided_config = config if config is not None else PipelineConfig()

        # Fully instantiate the user_provided_config
        self._config = self._full_init_config(user_provided_config)

        # Initialize the node objects dictionary (node name to node instance)
        self._node_objects: dict[str, Node] = {}
        # Initialize the graph structure (adjacency list representation)
        self._graph: nx.DiGraph = nx.DiGraph()
        # Initialize the sink node name
        self._sink_node_name: str | None = self._init_sink_node_name()
        # Instantiate the Sink Edges indexes
        self._sink_edges_idx: list[tuple[str, str]] = self._init_sink_edges_idx()
        # Initialize the graph representation according to the provided config
        self._init_graph_repr()

    #### Properties ####
    # --- Properties for pipeline attributes ---
    @property
    def config(self) -> PipelineConfig:
        """Get the user-facing pipeline configuration."""
        return PipelineConfig(
            nodes=self._config.nodes,
            edges=list(self._config.edges.values()),
            name=self._config.name,
            version=self._config.version,
            settings=self._config.settings,
        )

    @property
    def internal_config(self) -> _InternalPipelineConfig:
        """Get the internal pipeline configuration."""
        return self._config

    @property
    def node_objects(self) -> dict[str, Node]:
        """Get the dictionary of node objects in the pipeline."""
        return self._node_objects

    @property
    def graph(self) -> nx.DiGraph:
        """Get the graph structure of the pipeline."""
        return self._graph

    # --- Properties for quick access ---
    @property
    def nodes(self) -> dict[str, tuple[str, NodeConfig]]:
        """Get the nodes in the pipeline."""
        return self._config.nodes

    @property
    def edges(self) -> list[Edge]:
        """Get the edges in the pipeline."""
        return list(self._config.edges.values())

    @property
    def edges_dict(self) -> dict[tuple[str, str], Edge]:
        """Get the internal edges of the pipeline."""
        return self._config.edges

    @property
    def name(self) -> str:
        """Get the name of the pipeline."""
        return self._config.name

    @property
    def version(self) -> Version:
        """Get the version of the pipeline."""
        return self._config.version

    @property
    def settings(self) -> PipelineSettings:
        """Get the settings of the pipeline."""
        return self._config.settings

    @property
    def compiled(self) -> bool:
        """Check if the pipeline is compiled."""
        return self._compiled

    @property
    def sink_node_name(self) -> str | None:
        """Get the name of the sink node in the pipeline, if it exists."""
        return self._sink_node_name

    @property
    def sink_edges_dict(self) -> dict[tuple[str, str], Edge]:
        """Get the edges connected to the sink node."""
        if self.sink_node_name is None:
            return {}
        return {idx: self._config.edges[idx] for idx in self._sink_edges_idx}

    @property
    def graph_wo_metrics(self) -> nx.DiGraph:
        def filter_nodes(node_name: str) -> bool:
            node_type, _ = self._config.nodes[node_name]
            return node_type != NodeType.METRIC

        def filter_edges(source: str, target: str) -> bool:
            edge = self._config.edges.get((source, target))
            if edge is None:
                return True
            source_node_type, _ = self._config.nodes[source]
            target_node_type, _ = self._config.nodes[target]
            return NodeType.METRIC not in (source_node_type, target_node_type)

        return nx.subgraph_view(
            self._graph, filter_node=filter_nodes, filter_edge=filter_edges
        )

    #### Public API ####
    # --- Public API for pipeline management ---

    def add_node(
        self, node_name: str, node_type: str, node_config: NodeConfig | None = None
    ) -> None:
        """Add a node to the pipeline."""
        if node_name in self._config.nodes:
            message = f"Node '{node_name}' already exists in the pipeline."
            raise ValueError(message)
        if node_config is None:
            node_config = self._get_default_node_config(node_type)
        if node_config.node_type == NodeType.SINK and self.sink_node_name is not None:
            message = f"Sink node '{self.sink_node_name}' already exists in the pipeline. Cannot add another sink node '{node_name}'."
            raise ValueError(message)
        # Add the node to the pipeline configuration
        self._config.nodes[node_name] = (node_type, node_config)
        # Add the node to the graph structure
        self._add_node_to_nx(node_name)

    def remove_node(self, node_name: str) -> None:
        """Remove a node from the pipeline."""
        if node_name not in self._config.nodes:
            message = f"Node '{node_name}' does not exist in the pipeline."
            raise ValueError(message)
        # If this node is the Sink node, we need to remove all edges connected to it from the list of sink edges
        if node_name == self.sink_node_name:
            for idx in self._sink_edges_idx:
                del self._config.edges[idx]
            self._sink_edges_idx = []
            self._sink_node_name = None
        # Remove the node from the pipeline configuration
        del self._config.nodes[node_name]
        # Remove the node from the graph structure
        self._remove_node_from_nx(node_name)

    def update_node(self, node_name: str, node_config: NodeConfig) -> None:
        """Update a node's configuration in the pipeline."""
        if node_name not in self._config.nodes:
            message = f"Node '{node_name}' does not exist in the pipeline."
            raise ValueError(message)
        node_type, _ = self._config.nodes[node_name]
        # Update the node configuration in the pipeline configuration
        self._config.nodes[node_name] = (node_type, node_config)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the pipeline."""
        # Validate the edge before adding it to the pipeline
        self._validate_edge(
            edge
        )  # The edges are also checked at compile time just in case, but we want to raise errors early mainly for rendering
        # If an edge already exists between the source and target nodes, we will just update the ports_map
        if edge.target == self.sink_node_name:
            # if the target is the Sink node, we need to add this edge to the list of sink edges
            self._sink_edges_idx.append((edge.source, edge.target))
        if edge.source == self.sink_node_name:
            message = f"Edges with source as the Sink node '{self.sink_node_name}' are not allowed. Cannot add edge from '{edge.source}' to '{edge.target}'."
            raise ValueError(message)
        if self._graph.has_edge(edge.source, edge.target):
            existing_edge = self._config.edges[(edge.source, edge.target)]
            existing_edge.ports_map = list(
                set(existing_edge.ports_map + edge.ports_map)
            )  # Merge the ports_map while avoiding duplicates
            self._config.edges[(edge.source, edge.target)] = existing_edge
            self._update_edge_in_nx(existing_edge)
        else:
            # Add the edge to the pipeline configuration
            self._config.edges[(edge.source, edge.target)] = edge
            # Add the edge to the graph structure
            self._add_edge_to_nx(edge)

    def remove_edge(self, source: str, target: str) -> None:
        """Remove an edge from the pipeline."""
        if (source, target) not in self._config.edges:
            message = (
                f"Edge from '{source}' to '{target}' does not exist in the pipeline."
            )
            raise ValueError(message)
        # If this edge is connected to the Sink node, we need to remove it from the list of sink edges
        if target == self.sink_node_name:
            self._sink_edges_idx.remove((source, target))
        # Remove the edge from the pipeline configuration
        del self._config.edges[(source, target)]
        # Remove the edge from the graph structure
        self._remove_edge_from_nx(Edge(source=source, target=target, ports_map=[]))

    def update_edge(self, edge: Edge) -> None:
        """Update an edge's ports_map in the pipeline."""
        if (edge.source, edge.target) not in self._config.edges:
            message = f"Edge from '{edge.source}' to '{edge.target}' does not exist in the pipeline."
            raise ValueError(message)
        if edge.target == self.sink_node_name:
            # if the target is the Sink node, we need to update this edge in the list of sink edges
            idx = self._sink_edges_idx.index((edge.source, edge.target))
            self._sink_edges_idx[idx] = (edge.source, edge.target)
        # Update the edge in the pipeline configuration
        self._config.edges[(edge.source, edge.target)] = edge
        # Update the edge in the graph structure
        self._update_edge_in_nx(edge)

    def compile(self) -> None:
        """Compile the pipeline by instantiating all node objects and validating the graph structure."""
        if self.compiled:
            message = "Pipeline is already compiled."
            raise ValueError(message)
        self._compile()

    ## --- Public API for getters ---
    def get_node_config(self, node_name: str) -> NodeConfig:
        """Get a node's configuration from the pipeline."""
        if node_name not in self._config.nodes:
            message = f"Node '{node_name}' does not exist in the pipeline."
            raise ValueError(message)
        return self._config.nodes[node_name][1]

    def get_validated_sink_node_name(self) -> str:
        """Get the name of the sink node in the pipeline, if it exists. Raise an error if the sink node is not properly configured."""
        if self.sink_node_name is None:
            message = "No Sink node found in the pipeline. A pipeline must have one Sink node to be valid."
            raise ValueError(message)
        return self.sink_node_name

    def get_edge(self, source: str, target: str) -> Edge:
        """Get an edge's configuration from the pipeline."""
        if (source, target) not in self._config.edges:
            message = (
                f"Edge from '{source}' to '{target}' does not exist in the pipeline."
            )
            raise ValueError(message)
        return self._config.edges[(source, target)]

    def get_source_node_names(self) -> list[str]:
        """Get the list of source node names in the pipeline."""
        source_node_names = []
        for node_name, (_, node_config) in self._config.nodes.items():
            if node_config.node_type == NodeType.SOURCE:
                source_node_names.append(node_name)
        return source_node_names

    ## --- Public API for params management ---
    def get_params(self) -> dict[str, Any]:
        """Get the parameters of all nodes in the pipeline, for all nodes that have parameters."""
        return {
            node_name: node_object.get_params()
            for node_name, node_object in self.node_objects.items()
            if has_params(node_object)
        }

    def set_params(self, params: dict[str, dict[str, Any]]) -> None:
        """Set the parameters of all nodes in the pipeline, for all nodes that have parameters."""
        for node_name, node_params in params.items():
            if node_name not in self.node_objects:
                message = f"Node '{node_name}' does not exist in the pipeline. Cannot set parameters for non-existing node."
                raise ValueError(message)
            node_object = self.node_objects[node_name]
            if not has_params(node_object):
                message = f"Node '{node_name}' does not have parameters. Cannot set parameters for node that does not have parameters."
                raise ValueError(message)
            node_object.set_params(node_params)

    #### Internal methods ####
    # --- Internal methods for pipeline management ---
    def _full_init_config(self, config: PipelineConfig) -> _InternalPipelineConfig:
        """Fully instantiate the given PipelineConfig.

        This method ensures that all NodeConfig instances in the PipelineConfig are fully instantiated, which is necessary for proper validation and type checking.
        The key difference usually lies in the fact that all NodeConfig will be fully instantiated.

        Args:
            config (PipelineConfig): The PipelineConfig to fully instantiate.

        Returns:
            _InternalPipelineConfig: The fully instantiated _InternalPipelineConfig.

        """
        fully_instantiated_nodes = self._full_init_node_configs(config.nodes)
        edges_dict = {(edge.source, edge.target): edge for edge in config.edges}
        return _InternalPipelineConfig(
            nodes=fully_instantiated_nodes,
            edges=edges_dict,
            name=config.name,
            version=config.version,
            settings=config.settings,
        )

    def _instantiate_node_objects(self) -> None:
        """Instantiate the node objects based on the current pipeline configuration."""
        for node_name, (node_type, node_config) in self._config.nodes.items():
            node_class = NODE_REGISTRY.get_node_class(node_type)
            node_object = node_class(config=node_config)
            self._node_objects[node_name] = node_object

    def _prune(self) -> None:
        """Prune the pipeline by removing any nodes that are not connected to any Node."""
        # Get the set of all nodes that are connected to at least one edge
        connected_nodes = set()
        for edge in self.edges:
            connected_nodes.add(edge.source)
            connected_nodes.add(edge.target)

        # Get the set of all nodes in the pipeline
        all_nodes = set(self.nodes.keys())

        # Get the set of unconnected nodes
        unconnected_nodes = all_nodes - connected_nodes

        # Remove unconnected nodes from the pipeline
        for node_name in unconnected_nodes:
            self.remove_node(node_name)

    # --- Internal methods for Nodes ---
    def _full_init_node_configs(
        self, nodes: Mapping[str, tuple[str, NodeConfig | None]]
    ) -> dict[str, tuple[str, NodeConfig]]:
        """Fully instantiate all NodeConfig instances in the given nodes dictionary."""
        fully_instantiated_nodes = {}
        for node_name, (node_type, node_config) in nodes.items():
            if node_config is not None:
                fully_instantiated_nodes[node_name] = (node_type, node_config)
            else:
                # If node_config is None, we need to create a default NodeConfig based on the node_type
                default_node_config = self._get_default_node_config(node_type)
                fully_instantiated_nodes[node_name] = (node_type, default_node_config)
        return fully_instantiated_nodes

    def _get_default_node_config(self, node_class_name: str) -> NodeConfig:
        """Get a default NodeConfig instance based on the given node_class_name."""
        node_conf_class = NODE_REGISTRY.get_node_config_class(node_class_name)
        return node_conf_class()

    def _init_sink_node_name(self) -> str | None:
        """Initialize the sink node name based on the current pipeline configuration."""
        sink_node_name = None
        for node_name, (_, node_config) in self._config.nodes.items():
            if node_config.node_type == NodeType.SINK:
                if sink_node_name is not None:
                    message = f"Multiple Sink nodes found in the pipeline ('{sink_node_name}' and '{node_name}'). A pipeline can only have one Sink node."
                    raise ValueError(message)
                sink_node_name = node_name
        return sink_node_name

    # --- Internal methods for edges ---
    def _init_sink_edges_idx(self) -> list[tuple[str, str]]:
        """Initialize the list of sink edges indexes based on the current pipeline configuration."""
        sink_edges_idx: list[tuple[str, str]] = []
        if self.sink_node_name is not None:
            sink_edges_idx = [
                (edge.source, edge.target)
                for edge in self.edges
                if edge.target == self.sink_node_name
            ]
        return sink_edges_idx

    # --- Internal methods for NX graph management ---
    def _init_graph_repr(self) -> None:
        """Initialize the graph structure of the pipeline based on the current configuration."""
        for node_name in self._config.nodes.keys():
            self._add_node_to_nx(node_name)
        for edge in self._config.edges.values():
            self._add_edge_to_nx(edge)

    def _add_edge_to_nx(self, edge: Edge) -> None:
        """Add an edge to the graph structure of the pipeline."""
        source_node_name = edge.source
        target_node_name = edge.target

        # Add the edge to the graph structure
        self._graph.add_edge(source_node_name, target_node_name)

        # Set the ports map as edge attributes in the graph
        self._graph.edges[source_node_name, target_node_name]["ports_map"] = (
            edge.ports_map
        )

    def _add_node_to_nx(self, node_name: str) -> None:
        """Add a node to the graph structure of the pipeline."""
        self._graph.add_node(node_name)

    def _update_edge_in_nx(self, edge: Edge) -> None:
        """Update an edge in the graph structure of the pipeline."""
        source_node_name = edge.source
        target_node_name = edge.target

        # Update the ports map as edge attributes in the graph
        if self._graph.has_edge(source_node_name, target_node_name):
            self._graph.edges[source_node_name, target_node_name]["ports_map"] = (
                edge.ports_map
            )
        else:
            message = f"Edge from '{source_node_name}' to '{target_node_name}' does not exist in the graph. Cannot update non-existing edge."
            raise ValueError(message)

    def _remove_edge_from_nx(self, edge: Edge) -> None:
        """Remove an edge from the graph structure of the pipeline."""
        source_node_name = edge.source
        target_node_name = edge.target

        # Remove the edge from the graph structure
        if self._graph.has_edge(source_node_name, target_node_name):
            self._graph.remove_edge(source_node_name, target_node_name)
        else:
            message = f"Edge from '{source_node_name}' to '{target_node_name}' does not exist in the graph. Cannot remove non-existing edge."
            raise ValueError(message)

    def _remove_node_from_nx(self, node_name: str) -> None:
        """Remove a node from the graph structure of the pipeline."""
        # Remove the node from the graph structure
        if self._graph.has_node(node_name):
            self._graph.remove_node(node_name)
        else:
            message = f"Node '{node_name}' does not exist in the graph. Cannot remove non-existing node."
            raise ValueError(message)

    # --- Internal methods for validation ---
    def _validate(self) -> None:
        """Validate the pipeline layout and configuration.

        This method checks for issues such as:
        - Presence of a Sink node
        - Validity of edges (source and target nodes should exist, ports_map should be valid, etc.)
        - Absence of cycles in the graph (if not allowed)

        Raises:
            ValueError: If any validation check fails.

        """
        # Check for the presence of a Sink node
        if self.sink_node_name is None:
            message = "Pipeline must contain a Sink node, but no Sink node was found."
            raise ValueError(message)

        # Check the validity of edges
        for edge in self.edges:
            self._validate_edge(edge)

        # Check for cycles in the graph (if not allowed)
        if not nx.is_directed_acyclic_graph(self._graph):
            message = "Pipeline graph contains cycles, but cycles are not allowed."
            raise ValueError(message)

    def _validate_edge(self, edge: Edge) -> None:
        """Validate an edge's configuration and compatibility with the source and target nodes.

        This method checks for issues such as:
        - Existence of source and target nodes in the pipeline
        - Validity of ports_map (source ports should exist in the source node, target ports should exist in the target node, etc.)
        - Compatibility of data types between source and target ports
        - Check that no edge originates from the Sink node

        Args:
            edge (Edge): The edge to validate.

        Raises:
            ValueError: If any validation check fails.

        """
        source_node_conf = self.get_node_config(edge.source)
        target_node_conf = self.get_node_config(edge.target)

        # For now only do a limited check of the compatibility of the ports
        source_out_ports = source_node_conf.out_ports
        target_in_ports = target_node_conf.in_ports

        for source_port, target_port in edge.ports_map:
            # Check that all source ports in the ports_map exist in the source node's out_ports
            if source_port not in source_out_ports:
                message = f"Source port '{source_port}' in edge from '{edge.source}' to '{edge.target}' does not exist in the source node's out_ports."
                raise ValueError(message)
            if target_port not in target_in_ports:
                message = f"Target port '{target_port}' in edge from '{edge.source}' to '{edge.target}' does not exist in the target node's in_ports."
                raise ValueError(message)

            # Check that data structures match between the source and target ports
            self._is_compatible_structures(
                source_out_ports,
                target_in_ports,
                edge.ports_map,
                edge.source,
                edge.target,
            )

            # Check that data categories match between the source and target ports
            self._is_compatible_categories(
                source_out_ports,
                target_in_ports,
                edge.ports_map,
                edge.source,
                edge.target,
            )

            # Check that the execution modes of the ports are compatible (can't have an edge from a port that only executes in "train" mode to a port that only executes in "inference" mode for example)
            self._are_execution_modes_compatible(
                source_out_ports,
                target_in_ports,
                edge.ports_map,
                edge.source,
                edge.target,
            )

            # Finally, check that no edge originates from the Sink node
            if source_node_conf.node_type == NodeType.SINK:
                message = f"Edge from '{edge.source}' to '{edge.target}' is not valid because it originates from a Sink node, which cannot have outgoing edges."
                raise ValueError(message)

    def _are_execution_modes_compatible(
        self,
        source_out_ports: dict[str, Port],
        target_in_ports: dict[str, Port],
        ports_map: list[tuple[str, str]],
        source: str,
        target: str,
    ) -> None:
        """Check that the execution modes of the source and target ports in the given ports_map are compatible."""
        for source_port, target_port in ports_map:
            source_modes = source_out_ports[source_port].mode
            target_modes = target_in_ports[target_port].mode
            if not (set(source_modes).intersection(set(target_modes))) and not (
                NodeExecutionMode.ALL in source_modes
                or NodeExecutionMode.ALL in target_modes
            ):
                message = f"Execution modes of source port '{source_port}' in edge from '{source}' to '{target}' are not compatible with execution modes of target port '{target_port}'.\n"
                message += (
                    f"Source port '{source_port}' execution modes: {source_modes}\n"
                )
                message += (
                    f"Target port '{target_port}' execution modes: {target_modes}\n"
                )
                message += "Please ensure that the execution modes are compatible."
                raise ValueError(message)

    def _is_compatible_structures(
        self,
        source_out_ports: dict[str, Port],
        target_in_ports: dict[str, Port],
        ports_map: list[tuple[str, str]],
        source: str,
        target: str,
    ) -> None:
        """Check that the data structures of the source and target ports in the given ports_map are compatible."""
        for source_port, target_port in ports_map:
            source_structure = DATA_STRUCTURE_MAPPING[
                source_out_ports[source_port].data_structure
            ]
            target_structure = DATA_STRUCTURE_MAPPING[
                target_in_ports[target_port].data_structure
            ]
            target_sub_source = issubclass(target_structure, source_structure)
            source_sub_target = issubclass(source_structure, target_structure)
            if not (target_sub_source or source_sub_target):
                message = f"Data structure of source port '{source_port}' in edge from '{source}' to '{target}' is not compatible with data structure of target port '{target_port}'."
                raise ValueError(message)
            if not (target_sub_source and source_sub_target):
                # If the structures are not the same but are compatible, we can print a warning message to inform the user that there might be a potential issue with data structure compatibility, even though it is technically valid.
                print(
                    f"Warning: Data structure of source port '{source_port}' in edge from '{source}' to '{target}' is not the same as data structure of target port '{target_port}', but they are compatible. Please ensure that this is intentional and that the nodes can handle the data structure conversion if needed."
                )  # TODO : replace with logging when implemented

    def _is_compatible_categories(
        self,
        source_out_ports: dict[str, Port],
        target_in_ports: dict[str, Port],
        ports_map: list[tuple[str, str]],
        source: str,
        target: str,
    ) -> None:
        """Check that the data categories of the source and target ports in the given ports_map are compatible."""
        for source_port, target_port in ports_map:
            source_categories = set(
                DATA_CATEGORY_MAPPING[
                    source_out_ports[source_port].data_category
                ].dtypes
            )
            target_categories = set(
                DATA_CATEGORY_MAPPING[target_in_ports[target_port].data_category].dtypes
            )
            if not (
                source_categories.issubset(target_categories)
                or target_categories.issubset(source_categories)
            ):
                message = f"Data category of source port '{source_port}' in edge from '{source}' to '{target}' is not compatible with data category of target port '{target_port}'.\n"
                message += f"Source port '{source_port}' data categories: {source_categories}\n"
                message += f"Target port '{target_port}' data categories: {target_categories}\n"
                message += "Please ensure that the data categories are compatible"
                raise ValueError(message)
            if target_categories.issubset(
                source_categories
            ) and not source_categories.issubset(target_categories):
                message = f"Data category of source port '{source_port}' in edge from '{source}' to '{target}' is a superset of data category of target port '{target_port}'. Please ensure that this is intentional and that the target node can handle the additional data categories if needed."
                print(
                    f"Warning: {message}"
                )  # TODO : replace with logging when implemented
            # In other cases the edge is valid no questions asked

    # --- Internal methods for compilation ---
    def _uncompile(self) -> None:
        """Signals the Pipeline is no longer fit for running."""
        self._compiled = False

    def _compile(self) -> None:
        """Signals the Pipeline is fit for running.

        This function runs :
        - clearing of the previous node instances
        - pruning if needed
        - validation of the layout
        - instantiation of the nodes
        - instantation of the true sink ports
        """
        # Clear the node objects
        self._node_objects = {}
        # If needed, prune the graph (remove Nodes without outgoing or incoming edges)
        if self.settings.autoprune:
            self._prune()

        # Validate the layout
        self._validate()

        # Now that everything is validated, instantiate the nodes
        self._instantiate_node_objects()

        # Finish compilation
        self._compiled = True

    # --- Rendering utils ---
    def render(
        self,
        title: str | None = None,
        backend: str = "plotly",
        figsize: tuple[int, int] = (12, 8),
    ):
        """Render the pipeline graph using the specified backend."""
        if title is None:
            title = f"Pipeline: {self.name}"
        if backend == "plotly":
            fig = render_pipeline_graph_plotly(
                pipeline=self, title=title, figsize=figsize
            )
            fig.show()
        else:
            message = f"Backend '{backend}' is not supported for rendering. Supported backends are: ['plotly']."
            raise ValueError(message)
