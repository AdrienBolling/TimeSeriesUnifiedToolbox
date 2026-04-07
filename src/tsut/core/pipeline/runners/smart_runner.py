"""SmartRunner implementation module."""

import time
from collections.abc import Mapping
from typing import Any

import networkx as nx
from jaxtyping import AbstractDtype

from tsut.core.common.data.data import (
    DATA_CATEGORY_MAPPING,
    ArrayLike,
    ArrayLikeEnum,
    Data,
    DataContext,
    TabularData,
    TabularDataContext,
)
from tsut.core.common.enums import NodeExecutionMode
from tsut.core.common.logging import Logger
from tsut.core.common.typechecking.typeguards import accepts_inputs_source_node
from tsut.core.nodes.node import Node, NodeType, Port
from tsut.core.pipeline.pipeline import Edge, Pipeline
from tsut.core.pipeline.runners.pipeline_runner import PipelineRunner, RunnerConfig


class SmartRunnerConfig(RunnerConfig):
    """Configuration for the SmartRunner."""


class SmartRunner(
    PipelineRunner
):  # Work on some better typing for the metrics output in the future, maybe with a generic M for the metrics output type.
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
            msg = "Pipeline must be compiled before being passed to the SmartRunner."
            raise ValueError(msg)  # No logger yet — __init__ hasn't finished
        self._pipeline = pipeline
        self._config = config if config is not None else SmartRunnerConfig()
        self._mode = NodeExecutionMode.DEFAULT
        self._node_outputs: dict[
            str, dict[str, Data]
        ] = {}  # For memoization of node outputs during execution
        self._metric_node_outputs: dict[
            str, dict[str, Any]
        ] = {}  # To store the outputs of metric nodes during evaluation
        self._log = Logger(
            "tsut.runner.smart",
            pipeline_name=pipeline.name,
            pipeline_version=pipeline.version,
        )
        self._input_data: Mapping[str, Mapping[str, Data]] | None = (
            None  # To store the input data passed to the runner, which can be used by the nodes during execution if needed.
        )

    # --- PipelineRunner API implementation ---

    def train(self, input_data: Mapping[str, Mapping[str, Data]] | None = None) -> None:
        """Train the pipeline."""
        self._reset_caches()
        self._input_data = input_data
        self._mode = NodeExecutionMode.TRAINING
        self._log.log_phase("training", "start")
        t0 = time.perf_counter()
        try:
            last_node = self.pipeline.get_validated_sink_node_name()
            self._call_node(last_node)
        except Exception as exc:
            self._log.exception("Training failed", exc)
            raise
        self._log.log_phase(
            "training", "end", duration_ms=(time.perf_counter() - t0) * 1000
        )

    def evaluate(
        self, input_data: Mapping[str, Mapping[str, Data]] | None = None
    ) -> dict[str, Any]:
        """Evaluate the pipeline and return the metrics."""
        self._reset_caches()
        self._input_data = input_data  # Store the input data for use during evaluation if needed by the nodes.
        self._mode = NodeExecutionMode.EVALUATION
        self._log.log_phase("evaluation", "start")
        t0 = time.perf_counter()
        try:
            metric_node_names = self.get_metric_node_names()
            for metric_node_name in metric_node_names:
                self._call_node(metric_node_name)
        except Exception as exc:
            self._log.exception("Evaluation failed", exc)
            raise
        self._log.log_phase(
            "evaluation", "end", duration_ms=(time.perf_counter() - t0) * 1000
        )
        return self._metric_node_outputs

    def infer(
        self, input_data: Mapping[str, Mapping[str, Data]] | None = None
    ) -> dict[str, Data]:
        """Run inference with the pipeline."""
        self._reset_caches()
        self._input_data = input_data  # Store the input data for use during inference if needed by the nodes.
        self._mode = NodeExecutionMode.INFERENCE
        self._log.log_phase("inference", "start")
        t0 = time.perf_counter()
        try:
            last_node = self.pipeline.get_validated_sink_node_name()
            result = self._call_node(last_node)
        except Exception as exc:
            self._log.exception("Inference failed", exc)
            raise
        self._log.log_phase(
            "inference", "end", duration_ms=(time.perf_counter() - t0) * 1000
        )
        return result

    def _reset_caches(self) -> None:
        """Clear memoized outputs so each phase recomputes from scratch."""
        self._node_outputs = {}
        self._metric_node_outputs = {}
        self._input_data = None

    # --- Internal methods for node execution ---

    def _tabular_convert_from_node_output(
        self, output: tuple[ArrayLike, TabularDataContext]
    ) -> TabularData:
        """Convert the arr_type output from a node to the base data type used by the SmartRunner."""
        context = output[1]
        return TabularData(
            data=output[0],
            columns=context.columns,
            dtypes=context.dtypes,
            categories=context.categories,
        )

    def _convert_from_node_output(self, output: tuple[ArrayLike, DataContext]) -> Data:
        """Convert the arr_type output from a node to the base data type used by the SmartRunner. This function is a wrapper that checks the type of the data and calls the appropriate conversion function."""
        (data, context) = output
        if isinstance(context, TabularDataContext):
            return self._tabular_convert_from_node_output((data, context))
        msg = f"Unsupported data context type: {type(context)}. Currently only TabularDataContext is supported as output data context type for the SmartRunner."
        self._log.error(msg)
        raise ValueError(msg)

    def _tabular_convert_to_node_input(
        self, data: TabularData, arr_type: ArrayLikeEnum
    ) -> tuple[ArrayLike, TabularDataContext]:
        """Convert the base data type used by the SmartRunner to the input arr_type expected by a node."""
        match arr_type:
            case ArrayLikeEnum.PANDAS:
                return data.to_pandas()
            case ArrayLikeEnum.NUMPY:
                return data.to_numpy()
            case ArrayLikeEnum.TORCH:
                return data.to_tensor()
            case _:
                msg = f"Unsupported array type: {arr_type}. Supported types are: {ArrayLikeEnum.PANDAS}, {ArrayLikeEnum.NUMPY}, {ArrayLikeEnum.TORCH}."
                self._log.error(msg)
                raise ValueError(msg)

    def _convert_to_node_input(
        self, data: Data, arr_type: ArrayLikeEnum
    ) -> tuple[ArrayLike, DataContext]:
        """Convert the base data type used by the SmartRunner to the input arr_type expected by a node. This function is a wrapper that checks the type of the data and calls the appropriate conversion function."""
        if isinstance(data, TabularData):
            return self._tabular_convert_to_node_input(data, arr_type)
        msg = f"Unsupported data type: {type(data)}. Currently only TabularData is supported as input data type for the SmartRunner."
        self._log.error(msg)
        raise ValueError(msg)

    def _get_execution_order(self) -> list[str]:
        """Compute the execution order using topological sort."""
        return list(nx.topological_sort(self.pipeline.graph))

    def _call_node(self, node_name: str) -> dict[str, Data]:
        """Recuservely calls the nodes then its predecessors, and returns the output of the node with the common data type."""
        self._log.log_node_call(node_name, self._mode)
        # Get the predecessors of the node to gather inputs
        predecessors = list(self.pipeline.graph.predecessors(node_name))
        edges = [
            self.pipeline.get_edge(predecessor, node_name)
            for predecessor in predecessors
        ]
        for pred in predecessors:
            if pred not in self._node_outputs:
                self._node_outputs[pred] = self._call_node(pred)

        t0 = time.perf_counter()
        value = self._execute_node(node_name, edges)
        duration_ms = (time.perf_counter() - t0) * 1000

        node_type = self.node_objects[node_name].config.node_type
        self._log.log_node_execution(
            node_name, self._mode, duration_ms=duration_ms, node_type=node_type
        )
        if (
            node_type == NodeType.METRIC
        ):  # We don't memoize the output of the Sink node as it is supposed to be the last node in the pipeline and its output is not used by any other node.
            self._metric_node_outputs[node_name] = value
        return value

    def _execute_node(
        self, node_name: str, incoming_edges: list[Edge]
    ) -> dict[str, Data]:
        """Execute a specific node in the pipeline and return its output in the common data type."""
        node = self.node_objects[node_name]
        # Gather the inputs for the node by converting the outputs of the predecessor nodes to the expected
        # input types of the node using the edge information.
        inputs = {}
        for edge in incoming_edges:
            pred_node = self.node_objects[edge.source]
            pred_output = self._node_outputs[edge.source]
            for source_key, target_key in edge.ports_map:
                checked = self._runtime_typecheck_edge(
                    node,
                    pred_node,
                    pred_output,
                    target_key,
                    source_key,
                    node_name=edge.target,
                    pred_node_name=edge.source,
                )
                inputs[target_key] = checked
                self._log.log_data_flow(
                    edge.source,
                    node_name,
                    data_shape=str(checked.shape),
                )

        # Execute the node with the gathered inputs. The node will return its output in its expected arr_type, we convert it to the common data type before returning it.
        # First convert the inputs to the expected arr_type of the node, then execute the node, then convert the output to the common data type.
        node_inputs = {}
        for key, value in inputs.items():
            port = node.in_ports[key]
            node_inputs[key] = self._convert_to_node_input(value, port.arr_type)

        return self._get_node_outputs(node_name, node_inputs)

    def _runtime_typecheck_edge(
        self,
        target_node: Node,
        source_node: Node,
        source_output: dict[str, Data],
        target_key: str,
        source_key: str,
        node_name: str,
        pred_node_name: str,
    ) -> Data:
        """Convert the output of the source node to the expected input type of the target node using the edge information."""
        # Get the expected input type for the target node from the edge information
        target_port = target_node.in_ports[target_key]
        source_port = source_node.out_ports[source_key]

        # Write the jaxtyping type hints with the information from the ports
        target_jaxtyping_type = self._get_jaxtyping_type_from_port(target_port)

        # Dynamically create the conversion function based on the jaxtypes. Wrap it into a jaxtyped decorator. This is used for runtime validation of the conversions.
        def runtime_type_check(data: Data) -> Data:
            if isinstance(data, target_jaxtyping_type):  # pyright: ignore[reportArgumentType] We ignore because it's runtime type checking
                return data
            msg = f"Data type mismatch between source node '{pred_node_name}' and target node '{node_name}' on edge with source port '{source_key}' and target port '{target_key}'."
            msg += f"\nExpected type: {target_jaxtyping_type}, but got type: {type(data)}. (Expected source jaxtyping type: {self._get_jaxtyping_type_from_port(source_port)})."
            msg += f"\nData: shape - {data.shape}, dtype - {data.dtype}"
            self._log.error(msg, node_name=node_name, source_node=pred_node_name, source_port=source_key, target_port=target_key)
            raise TypeError(msg)

        return runtime_type_check(source_output[source_key])

    def _get_jaxtyping_type_from_port(self, port: Port) -> AbstractDtype:
        """Get the jaxtyping type hint from the port information."""
        # For simplicity, we will only check the arr_type for now, but we could also check the data_category and data_shape if needed.
        arr = TabularData  # That's where we implement the later logic of checking the data "class" (e.g. time series, tabular, etc) and not only the data type (e.g. pandas DataFrame, numpy array, etc)
        data_category = DATA_CATEGORY_MAPPING[port.data_category]
        shape_str = port.data_shape
        return data_category[arr, shape_str]  # pyright: ignore[reportReturnType] We ignore because it's runtime type checking, we can't know the exact type at static analysis time.

    def _get_node_outputs(
        self, node_name: str, inputs: dict[str, tuple[ArrayLike, DataContext]]
    ) -> dict[str, Data]:
        """Get the outputs of a node in the common data type."""
        self._check_inputs_completeness(node_name, inputs)
        node = self.node_objects[node_name]
        if (
            accepts_inputs_source_node(node)
            and node.accepts_inputs
            and self._input_data is not None
        ):
            # For source nodes, we pass the input data from the runner to the node, as source nodes are supposed to be the entry point of data into the pipeline and they might need access to the raw input data.
            inputs = {
                key: self._convert_to_node_input(
                    self._input_data[node_name][key], ArrayLikeEnum.PANDAS
                )
                for key in self._input_data[node_name].keys()
            }
        if isinstance(node, Node):
            match self.mode:
                case NodeExecutionMode.TRAINING:
                    output = node.node_fit_transform(inputs)
                case NodeExecutionMode.INFERENCE:
                    output = node.node_transform(inputs)
                case NodeExecutionMode.EVALUATION:
                    output = node.node_transform(inputs)
                case _:
                    message = f"Unsupported execution mode: {self.mode}"
                    self._log.error(message, node_name=node_name)
                    raise ValueError(message)
        else:
            message = f"Node '{node_name}' is not an instance of the Node class. Got type: {type(node)}."
            self._log.error(message, node_name=node_name)
            raise TypeError(message)
        return {
            key: self._convert_from_node_output(value) for key, value in output.items()
        }

    def _check_inputs_completeness(
        self, node_name: str, inputs: dict[str, tuple[ArrayLike, DataContext]]
    ) -> None:
        """Check if all required inputs for a node are present.

        A missing input is tolerated when the port's ``mode`` list does not
        include the current execution mode (and does not include ``"all"``).
        """
        node = self.node_objects[node_name]
        if not hasattr(node, "in_ports") or node.in_ports is None:
            return
        required_inputs = set(node.in_ports.keys())
        provided_inputs = set(inputs.keys())

        missing_inputs = required_inputs - provided_inputs
        for missing_input in missing_inputs:
            port_modes = node.in_ports[missing_input].mode
            # Port is not active in the current mode → skip.
            if (
                self._mode not in port_modes and NodeExecutionMode.ALL not in port_modes
            ) or node.in_ports[missing_input].optional:
                continue
            message = f"Node '{node_name}' ({node}) is missing required input: '{missing_input}'"
            self._log.error(message, node_name=node_name, missing_input=missing_input)
            raise ValueError(message)
