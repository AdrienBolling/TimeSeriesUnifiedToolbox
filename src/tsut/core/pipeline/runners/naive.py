"""Naive implementation of a TSUT PipelineRunner.

This module provides a straightforward, sequential implementation of the PipelineRunner
that executes nodes in topological order without optimizations.

Key Design Principles:
- All nodes produce outputs in both TRAIN and PREDICT modes
- TRAIN mode: fits nodes and transforms data for downstream dependencies
- PREDICT mode: only transforms data (no fitting)
- EVALUATE mode: placeholder for future performance evaluation
"""

from enum import StrEnum

import networkx as nx

from tsut.core.common.data.types import ContextData, Data
from tsut.core.pipeline.base import Pipeline
from tsut.core.pipeline.runners.base import RunnerConfig


class ExecutionMode(StrEnum):
    """Define execution modes for the pipeline runner."""

    TRAIN = "train"
    PREDICT = "predict"
    EVALUATE = "evaluate"


class NaivePipelineRunner:
    """Naive implementation of a PipelineRunner.

    This runner executes pipeline nodes sequentially in topological order,
    ensuring dependencies are satisfied before execution. It stores all
    intermediate results in memory for simplicity.

    Key Features:
    - Sequential execution (no parallelization)
    - All nodes produce outputs in all modes (needed by downstream nodes)
    - TRAIN mode: fits nodes and produces outputs
    - PREDICT mode: only transforms (no fitting)
    - EVALUATE mode: placeholder for future evaluation logic

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
        
        Note:
            Directly accesses Pipeline's private attributes since no public compile()
            method is available. This is consistent with the runner's responsibility
            to manage pipeline execution state.
        """
        if not self.pipeline.compiled():
            # Validate the pipeline structure
            self.pipeline._validate()
            # Mark the pipeline as compiled
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
    ) -> dict[str, Data | ContextData]:
        """Execute a single node with the given inputs.

        In both TRAIN and PREDICT modes, nodes must produce outputs for downstream nodes.
        The difference is that TRAIN mode also fits the node before transforming.

        Args:
            node_name: Name of the node to execute.
            inputs: Input data for the node.
            mode: Execution mode (train, predict, or evaluate).

        Returns:
            Node outputs for downstream nodes.

        Raises:
            ValueError: If the specified node is not found in the pipeline.

        """
        node = self.pipeline.node_objects.get(node_name)
        if node is None:
            message = f"Node '{node_name}' not found in pipeline node objects."
            raise ValueError(message)

        if mode == ExecutionMode.TRAIN:
            # In train mode, fit the node then transform to produce outputs
            return node.node_fit_transform(inputs)
        if mode == ExecutionMode.PREDICT:
            # In predict mode, only transform (no fitting)
            return node.node_transform(inputs)
        # EVALUATE mode - for now, same as predict (placeholder for future)
        return node.node_transform(inputs)

    def run(
        self, mode: ExecutionMode = ExecutionMode.TRAIN
    ) -> dict[str, dict[str, Data | ContextData]]:
        """Run the pipeline in the specified mode.

        Args:
            mode: Execution mode for the pipeline (train, predict, or evaluate).

        Returns:
            Dictionary mapping node names to their output data.
            For sink nodes or leaf nodes, returns their final outputs.

        Raises:
            ValueError: If the pipeline graph is not a valid DAG.

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

            # Execute the node - always produces outputs for downstream nodes
            outputs = self._execute_node(node_name, inputs, mode)

            # Store outputs for downstream nodes
            self._node_outputs[node_name] = outputs

        return self._node_outputs

    def get_sink_outputs(self) -> dict[str, dict[str, Data | ContextData]]:
        """Get outputs from sink nodes or leaf nodes (nodes with no successors).

        Returns:
            Dictionary mapping sink/leaf node names to their output data.
        """
        sink_outputs = {}
        
        for node_name in self.pipeline.node_objects:
            node = self.pipeline.node_objects[node_name]
            
            # Check if it's a sink node by type
            if hasattr(node, 'node_type') and node.node_type.value == "sink":
                if node_name in self._node_outputs:
                    sink_outputs[node_name] = self._node_outputs[node_name]
            # Or if it's a leaf node (no successors)
            elif not list(self.pipeline._graph.successors(node_name)):
                if node_name in self._node_outputs:
                    sink_outputs[node_name] = self._node_outputs[node_name]
        
        return sink_outputs

    def train(self) -> dict[str, dict[str, Data | ContextData]]:
        """Train the pipeline by executing all nodes in train mode.

        This method fits all nodes in the pipeline and transforms data through them,
        ensuring each node gets the outputs from its predecessors.

        Returns:
            Dictionary mapping node names to their output data.
        """
        return self.run(mode=ExecutionMode.TRAIN)

    def predict(self) -> dict[str, dict[str, Data | ContextData]]:
        """Run prediction through the pipeline without fitting.

        This method only transforms data through the pipeline without fitting any nodes.

        Returns:
            Dictionary mapping node names to their output data.
        """
        return self.run(mode=ExecutionMode.PREDICT)

    def evaluate(self) -> dict[str, dict[str, Data | ContextData]]:
        """Evaluate the pipeline performance.

        This method is a placeholder for future evaluation logic.
        Currently behaves the same as predict.

        Returns:
            Dictionary mapping node names to their output data.
        """
        return self.run(mode=ExecutionMode.EVALUATE)
