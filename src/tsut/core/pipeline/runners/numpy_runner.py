"""Basic implementation of a PipelineRunner that executes the pipeline assuming all nodes take and return NumPy arrays, so there's no need for any translation of inputs/outputs between nodes."""

from typing import Any

import networkx as nx

from tsut.core.common.enums import NodeExecutionMode as Mode
from tsut.core.nodes.registry.node_registry import NODE_REGISTRY
from tsut.core.pipeline.pipeline import Edge, Pipeline
from tsut.core.pipeline.runners.pipeline_runner import PipelineRunner, RunnerConfig


class NumpyPipelineRunnerConfig(RunnerConfig):
    """Configuration for the NumpyPipelineRunner."""


class NumpyPipelineRunner(PipelineRunner):
    """PipelineRunner implementation for pipelines where all nodes take and return NumPy arrays."""

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        config: NumpyPipelineRunnerConfig | None = None,
    ) -> None:
        """Initialize the NumpyPipelineRunner with a pipeline.

        Args:
            pipeline: The Pipeline object containing the DAG structure and nodes.
            config: Optional configuration for the runner.

        """
        self.pipeline = pipeline
        self.config = config if config is not None else NumpyPipelineRunnerConfig()
        self._node_outputs: dict[str, Any] = {} # For memoization of node outputs during execution
        self._mode = Mode.DEFAULT
        self._node_objects = self._instantiate_nodes()

    def _instantiate_nodes(self) -> None:
        """Instantiate node objects from the pipeline configuration."""
        self._node_objects = {}
        for node_name, node_config in self.pipeline._config.nodes.items():
            node_class = NODE_REGISTRY[node_name]["node_class"]
            self._node_objects[node_name] = node_class(**node_config)

    def _validate_pipeline(self) -> None:
        """Validate that the pipeline is compiled before execution."""
        if not self.pipeline.compiled():
            message = "Pipeline must be compiled before running. Please compile the pipeline first using 'pipeline.compile()'."
            raise ValueError(message)

    def _get_execution_order(self) -> list[str]:
        """Compute the execution order using topological sort."""
        return list(nx.topological_sort(self.pipeline.graph()))

    def _predict(self) -> None:
        """Train the pipeline using NumPy."""

        # Take the last node in the execution order.
        execution_order = self._get_execution_order()
        last_node_name = execution_order[-1]

        # Validate the pipeline before execution
        self._validate_pipeline()

        # Execute the last node in the pipeline.
        result = self._call_node(last_node_name)

        return result

    def _call_node(self, node_name: str) -> Any:
        """Execute a specific node in the pipeline and return its output."""
        # Validate the pipeline before execution
        self._validate_pipeline()

        # Get the predecessors of the node to gather inputs
        predecessors = self.pipeline.graph().predecessors(node_name)
        inputs = {}
        for pred in predecessors:
            if pred not in self._node_outputs:
                # Recursively execute the predecessor node if its output is not already computed
                self._node_outputs[pred] = self._call_node(pred)
            # Get the edge data to translate the output keys to the expected input keys
            edge = self.pipeline.graph().get_edge(pred, node_name)
            inputs = {**inputs, **self._filter_and_translate_inputs(node_name, edge, self._node_outputs[pred])}

        # Execute the node with the gathered inputs
        output = self._execute_node(node_name, inputs)
        self._node_outputs[node_name] = output  # Cache the output for future use
        return output


    def _execute_node(self, node_name: str, t_inputs: dict[str, Any] | None) -> Any:
        if t_inputs is None:
            t_inputs = {}

        self._check_inputs_completeness(node_name, t_inputs)

        if self._mode == Mode.TRAINING:
            return self.node_objects[node_name].node_fit_predict(**t_inputs)
        elif self._mode == Mode.INFERENCE:
            return self.node_objects[node_name].node_predict(**t_inputs)
        elif self._mode == Mode.EVALUATION:
            return self.node_objects[node_name].node_evaluate(**t_inputs)

    def _check_inputs_completeness(self, node_name: str, t_inputs: dict[str, Any]) -> None:
        """Check if all required inputs for a node are present."""
        node = self.node_objects[node_name]
        required_inputs = set(node.in_ports.keys())
        provided_inputs = set(t_inputs.keys())

        missing_inputs = required_inputs - provided_inputs
        # Check if the missing inputs are required in the current mode, if not optional, raise an error
        for missing_input in missing_inputs:
            if node.in_ports[missing_input].mode != self._mode:
                continue
            message = f"Node '{node_name}' is missing required input: '{missing_input}'"
            raise ValueError(message)

    def _filter_and_translate_inputs(self, node_name: str, edge: Edge, inputs: dict[str, Any]) -> dict[str, Any]:
        """Translate the keys of outputs to match the node's expected input keys."""
        translated_inputs = {}
        node = self.node_objects[node_name]
        inputs_keys = node.in_ports.keys()
        for input_key, output_key in edge.ports_map.items():
            if input_key not in inputs_keys:
                message = f"Node '{node_name}' does not have an input port named '{input_key}' as specified in the edge from '{edge.source}' to '{edge.target}'."
                raise ValueError(message)
            if output_key not in inputs:
                message = f"The output key '{output_key}' specified in the edge from '{edge.source}' to '{edge.target}' is not present in the outputs of node '{edge.source}'."
                raise ValueError(message)
            translated_inputs[input_key] = inputs[output_key]
        return translated_inputs
