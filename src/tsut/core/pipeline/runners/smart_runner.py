"""SmartRunner implementation module."""
# TODO : Runner that takes TabularData as the base data type and uses conversion to pass it around.
# Rely on the jaxtyping check for validity and the enriched ports.

from typing import Any

import networkx as nx
from jaxtyping import AbstractDtype

from tsut.core.common.data.data import (
    DATA_CATEGORY_MAPPING,
    ArrayLike,
    ArrayLikeEnum,
)
from tsut.core.common.data.tabular_data import TabularData, TabularDataContext
from tsut.core.common.enums import NodeExecutionMode
from tsut.core.nodes.node import Node, NodeType, Port
from tsut.core.pipeline.pipeline import Edge, Pipeline
from tsut.core.pipeline.runners.pipeline_runner import PipelineRunner, RunnerConfig


class SmartRunnerConfig(RunnerConfig):
    """Configuration for the SmartRunner."""

class TabularSmartRunner(PipelineRunner[TabularData, Any]): # Work on some better typing for the metrics output in the future, maybe with a generic M for the metrics output type.
    # I think it works with any Data subclass, but there needs to be another layer of checking for the data "class" (e.g. time series, tabular, etc)
    """SmartRunner implementation for the TSUT Framework.

    The SmartRunner is a PipelineRunner that takes TabularData as the base data type and uses conversion to pass it around.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        config: SmartRunnerConfig | None = None,
    ) -> None:
        """Initialize the SmartRunner with a pipeline.

        Args:
            pipeline: The Pipeline object containing the DAG structure and nodes.
            config: Optional configuration for the runner.

        """
        if not pipeline.compiled:
            raise ValueError("Pipeline must be compiled before being passed to the SmartRunner.")
        self._pipeline = pipeline
        self._config = config if config is not None else SmartRunnerConfig()
        self._mode = NodeExecutionMode.DEFAULT
        self._node_outputs: dict[str, dict[str, TabularData]] = {} # For memoization of node outputs during execution
        self._metric_node_outputs: dict[str, dict[str, Any]] = {} # To store the outputs of metric nodes during evaluation

    # --- PipelineRunner API implementation ---

    def train(self) -> None:
        """Train the pipeline."""
        self._mode = NodeExecutionMode.TRAINING
        last_node = self.pipeline.get_sink_node_name()
        self._call_node(last_node)

    def evaluate(self) -> dict[str, Any]:
        """Evaluate the pipeline and return the metrics."""
        last_node_w_metrics = self._get_execution_order()[-1] # The last node in the execution order with metrics is supposed to be the one that outputs the metrics.
        self._mode = NodeExecutionMode.EVALUATION
        self._call_node(last_node_w_metrics)

        return {
            node_name: output for node_name, output in self._metric_node_outputs.items()
        }

    def infer(self) -> dict[str, TabularData]:
        """Run inference with the pipeline."""
        self._mode = NodeExecutionMode.INFERENCE
        last_node = self.pipeline.get_sink_node_name()
        return self._call_node(last_node)

    # --- Internal methods for node execution ---

    def _convert_from_node_output(self, output: tuple[ArrayLike, TabularDataContext]) -> TabularData:
        """Convert the arr_type output from a node to the base data type used by the SmartRunner."""
        context = output[1]
        return TabularData(data=output[0], columns=context.columns, dtypes=context.dtypes, categories=context.categories)

    def _convert_to_node_input(self, data: TabularData, arr_type: type[ArrayLike]) -> tuple[ArrayLike, TabularDataContext]:
        """Convert the base data type used by the SmartRunner to the input arr_type expected by a node."""
        match arr_type:
            case ArrayLikeEnum.PANDAS:
                return data.to_pandas()
            case ArrayLikeEnum.NUMPY:
                return data.to_numpy()
            case ArrayLikeEnum.TORCH:
                return data.to_tensor()
            case _:
                raise ValueError(f"Unsupported array type: {arr_type}. Supported types are: {ArrayLikeEnum.PANDAS}, {ArrayLikeEnum.NUMPY}, {ArrayLikeEnum.TORCH}.")

    def _get_execution_order(self) -> list[str]:
        """Compute the execution order using topological sort."""
        return list(nx.topological_sort(self.pipeline.graph))

    def _call_node(self, node_name: str) -> dict[str, TabularData]:
        """Recuservely calls the nodes then its predecessors, and returns the output of the node with the common data type."""
        # Get the predecessors of the node to gather inputs
        predecessors = list(self.pipeline.graph.predecessors(node_name))
        edges = [self.pipeline.get_edge(predecessor, node_name) for predecessor in predecessors]
        for pred in predecessors:
            if pred not in self._node_outputs:
                self._node_outputs[pred] = self._call_node(pred)
        value = self._execute_node(node_name, edges)
        if self.node_objects[node_name].config.node_type == NodeType.METRIC: # We don't memoize the output of the Sink node as it is supposed to be the last node in the pipeline and its output is not used by any other node.
            self._metric_node_outputs[node_name] = value
        return value

    def _execute_node(self, node_name: str, incoming_edges: list[Edge]) -> dict[str, TabularData]:
        """Execute a specific node in the pipeline and return its output in the common data type."""
        node = self.node_objects[node_name]
        # Gather the inputs for the node by converting the outputs of the predecessor nodes to the expected
        # input types of the node using the edge information.
        inputs = {}
        for edge in incoming_edges:
            pred_node = self.node_objects[edge.source]
            pred_output = self._node_outputs[edge.source]

            inputs.update({
                target_key: self._runtime_typecheck_edge(node, pred_node, pred_output, target_key, source_key) for target_key, source_key in edge.ports_map.items()
            })

        # Execute the node with the gathered inputs. The node will return its output in its expected arr_type, we convert it to the common data type before returning it.
        # First convert the inputs to the expected arr_type of the node, then execute the node, then convert the output to the common data type.
        node_inputs = {}
        for key, value in inputs.items():
            port = node.in_ports[key]
            node_inputs[key] = self._convert_to_node_input(value, port.arr_type)

        return self._get_node_outputs(node_name, node_inputs)


    def _runtime_typecheck_edge(self, target_node: Node, source_node: Node, source_output: dict[str, TabularData], target_key: str, source_key: str) -> TabularData:
        """Convert the output of the source node to the expected input type of the target node using the edge information."""
        # Get the expected input type for the target node from the edge information
        target_port = target_node.in_ports[target_key]
        source_port = source_node.out_ports[source_key]

        # Write the jaxtyping type hints with the information from the ports
        target_jaxtyping_type = self._get_jaxtyping_type_from_port(target_port)

        # Dynamically create the conversion function based on the jaxtypes. Wrap it into a jaxtyped decorator. This is used for runtime validation of the conversions.
        def runtime_type_check(data: TabularData) -> TabularData:
            if isinstance(data, target_jaxtyping_type):
                return data
            raise TypeError(
                f"Data type mismatch between source node '{source_node}' and target node '{target_node}' on edge with source port '{source_key}' and target port '{target_key}'. Expected type: {target_jaxtyping_type}, but got type: {type(data)}. (Expected source jaxtyping type: {self._get_jaxtyping_type_from_port(source_port)}). Data: shape - {data.shape}, dtype - {data.dtype}"
            )

        return runtime_type_check(source_output[source_key])


    def _get_jaxtyping_type_from_port(self, port: Port) -> AbstractDtype:
        """Get the jaxtyping type hint from the port information."""
        # For simplicity, we will only check the arr_type for now, but we could also check the data_category and data_shape if needed.
        arr = TabularData # That's where we implement the later logic of checking the data "class" (e.g. time series, tabular, etc) and not only the data type (e.g. pandas DataFrame, numpy array, etc)
        data_category = DATA_CATEGORY_MAPPING[port.data_category]
        shape_str = port.data_shape
        return data_category[arr, shape_str]

    def _get_node_outputs(self, node_name: str, inputs: dict[str, tuple[ArrayLike, TabularDataContext]]) -> dict[str, TabularData]:
        """Get the outputs of a node in the common data type."""
        self._check_inputs_completeness(node_name, inputs)
        node = self.node_objects[node_name]
        match self.mode:
            case NodeExecutionMode.TRAINING:
                output = node.node_fit_transform(inputs)
            case NodeExecutionMode.INFERENCE:
                output = node.node_transform(inputs)
            case NodeExecutionMode.EVALUATION:
                output = node.node_transform(inputs)
            case _:
                message = f"Unsupported execution mode: {self.mode}"
                raise ValueError(message)
        return {key: self._convert_from_node_output(value) for key, value in output.items()}

    def _check_inputs_completeness(self, node_name: str, inputs: dict[str, tuple[ArrayLike, TabularDataContext]]) -> None:
        """Check if all required inputs for a node are present."""
        node = self.node_objects[node_name]
        if not hasattr(node, "in_ports") or node.in_ports is None:
            return
        required_inputs = set(node.in_ports.keys())
        provided_inputs = set(inputs.keys())

        missing_inputs = required_inputs - provided_inputs
        # Check if the missing inputs are required in the current mode, if not optional, raise an error
        for missing_input in missing_inputs:
            if node.in_ports[missing_input].mode != self._mode:
                continue
            message = f"Node '{node_name}' ({node}) is missing required input: '{missing_input}'"
            raise ValueError(message)
