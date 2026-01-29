"""Naive implementation of a TSUT PipelineRunner.

This module provides a straightforward, sequential implementation of the PipelineRunner
that executes nodes in topological order without optimizations.
"""

from enum import StrEnum
from typing import Any

import networkx as nx

from tsut.core.common.data.types import ContextData, Data
from tsut.core.pipeline.base import Pipeline
from tsut.core.pipeline.runners.base import RunnerConfig


class ExecutionMode(StrEnum):
    """Define execution modes for the pipeline runner."""

    FIT = "fit"
    TRANSFORM = "transform"
    FIT_TRANSFORM = "fit_transform"


class NaivePipelineRunner:
    """Naive implementation of a PipelineRunner.

    This runner executes pipeline nodes sequentially in topological order,
    ensuring dependencies are satisfied before execution. It stores all
    intermediate results in memory for simplicity.

    The implementation focuses on correctness over performance and does not
    include optimizations like parallel execution or lazy evaluation.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        config: RunnerConfig | None = None,
    ) -> None:
        """Initialize the NaivePipelineRunner with a pipeline.

        Args:
            pipeline: The Pipeline object containing the DAG structure and nodes.
            config: Optional configuration for the runner.

        """
        self.pipeline = pipeline
        self.config = config if config is not None else RunnerConfig()
        self._node_outputs: dict[str, dict[str, Data | ContextData]] = {}

    def _compile_pipeline(self) -> None:
        """Ensure the pipeline is compiled before execution.

        This validates the DAG structure and prepares the pipeline for execution.
        """
        if not self.pipeline.compiled():
            # The pipeline should handle its own compilation logic
            # For now, we just validate and mark as compiled
            self.pipeline._validate()
            self.pipeline._compiled = True

    def _get_execution_order(self) -> list[str]:
        """Compute the execution order using topological sort.

        Returns:
            A list of node names in topological order.

        Raises:
            ValueError: If the pipeline graph is not a valid DAG.

        """
        if not nx.is_directed_acyclic_graph(self.pipeline._graph):
            message = "Pipeline graph must be a directed acyclic graph (DAG)."
            raise ValueError(message)

        return list(nx.topological_sort(self.pipeline._graph))

    def _gather_node_inputs(
        self, node_name: str
    ) -> dict[str, Data | ContextData]:
        """Gather input data for a node from its predecessors.

        Args:
            node_name: Name of the node to gather inputs for.

        Returns:
            Dictionary mapping input port names to data.

        """
        inputs: dict[str, Data | ContextData] = {}

        # Get all predecessor nodes (nodes with edges pointing to this node)
        predecessors = list(self.pipeline._graph.predecessors(node_name))

        for predecessor in predecessors:
            # Get edge data which contains port mappings
            edge_data = self.pipeline._graph.get_edge_data(predecessor, node_name)
            if edge_data and "ports_map" in edge_data:
                ports_map: dict[str, str] = edge_data["ports_map"]

                # Map outputs from predecessor to inputs of current node
                if predecessor in self._node_outputs:
                    predecessor_outputs = self._node_outputs[predecessor]
                    for source_port, target_port in ports_map.items():
                        if source_port in predecessor_outputs:
                            inputs[target_port] = predecessor_outputs[source_port]

        return inputs

    def _execute_node(
        self,
        node_name: str,
        inputs: dict[str, Data | ContextData],
        mode: ExecutionMode,
    ) -> dict[str, Data | ContextData] | None:
        """Execute a single node with the given inputs.

        Args:
            node_name: Name of the node to execute.
            inputs: Input data for the node.
            mode: Execution mode (fit, transform, or fit_transform).

        Returns:
            Node outputs if mode is transform or fit_transform, None for fit only.

        """
        node = self.pipeline.node_objects.get(node_name)
        if node is None:
            message = f"Node '{node_name}' not found in pipeline node objects."
            raise ValueError(message)

        if mode == ExecutionMode.FIT:
            node.node_fit(inputs)
            return None
        elif mode == ExecutionMode.TRANSFORM:
            return node.node_transform(inputs)
        else:  # FIT_TRANSFORM
            return node.node_fit_transform(inputs)

    def run(
        self, mode: ExecutionMode = ExecutionMode.FIT_TRANSFORM
    ) -> dict[str, dict[str, Data | ContextData]]:
        """Run the pipeline in the specified mode.

        Args:
            mode: Execution mode for the pipeline.

        Returns:
            Dictionary mapping node names to their output data.

        """
        # Ensure pipeline is compiled
        self._compile_pipeline()

        # Get execution order
        execution_order = self._get_execution_order()

        # Clear previous outputs
        self._node_outputs = {}

        # Execute nodes in topological order
        for node_name in execution_order:
            # Gather inputs from predecessors
            inputs = self._gather_node_inputs(node_name)

            # Execute the node
            outputs = self._execute_node(node_name, inputs, mode)

            # Store outputs for downstream nodes (except for fit-only mode)
            if outputs is not None:
                self._node_outputs[node_name] = outputs

        return self._node_outputs

    def train(self) -> None:
        """Train the pipeline by executing all nodes in fit_transform mode.

        This method fits all nodes in the pipeline and transforms data through them.
        """
        self.run(mode=ExecutionMode.FIT_TRANSFORM)

    def fit(self) -> None:
        """Fit the pipeline by executing all nodes in fit mode.

        This method only fits nodes without transforming data.
        """
        self.run(mode=ExecutionMode.FIT)

    def transform(self) -> dict[str, dict[str, Data | ContextData]]:
        """Transform data through the pipeline without fitting.

        Returns:
            Dictionary mapping node names to their output data.

        """
        return self.run(mode=ExecutionMode.TRANSFORM)

    def fit_transform(self) -> dict[str, dict[str, Data | ContextData]]:
        """Fit and transform data through the pipeline.

        Returns:
            Dictionary mapping node names to their output data.

        """
        return self.run(mode=ExecutionMode.FIT_TRANSFORM)
