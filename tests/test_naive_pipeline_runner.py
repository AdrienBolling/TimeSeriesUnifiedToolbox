"""Tests for the NaivePipelineRunner implementation."""

import pytest

from tsut.core.common.data.types import ContextData, Data
from tsut.core.nodes.base import Node, NodeConfig, NodeType, Port
from tsut.core.pipeline.base import Edge, Pipeline, PipelineConfig
from tsut.core.pipeline.runners.naive import ExecutionMode, NaivePipelineRunner


# Mock Node classes for testing
class MockData(Data):
    """Mock data class for testing."""

    value: int = 0


class MockNode(Node[MockData, MockData]):
    """Mock node for testing."""

    def __init__(self, *, config: NodeConfig) -> None:
        """Initialize the mock node."""
        super().__init__(config=config)
        self.fit_called = False
        self.transform_called = False

    def node_fit(self, data: dict[str, MockData | ContextData]) -> None:
        """Mock fit implementation."""
        self.fit_called = True

    def node_transform(
        self, data: dict[str, MockData | ContextData]
    ) -> dict[str, MockData | ContextData]:
        """Mock transform implementation."""
        self.transform_called = True
        # Simply pass through the data with incremented value
        output = {}
        for key, value in data.items():
            if isinstance(value, MockData):
                output[key] = MockData(value=value.value + 1)
            else:
                output[key] = value
        return output


class MockSourceNode(MockNode):
    """Mock source node that generates data."""

    def node_transform(
        self, data: dict[str, MockData | ContextData]
    ) -> dict[str, MockData | ContextData]:
        """Generate mock data."""
        self.transform_called = True
        return {"output": MockData(value=1)}


def create_simple_pipeline() -> Pipeline:
    """Create a simple pipeline with two nodes for testing.

    Structure: A -> B
    """
    # Create node configs
    node_a_config = NodeConfig(
        node_type=NodeType.SOURCE,
        in_ports={},
        out_ports={"output": Port(type=MockData, desc="Output", mode=["train"])},
    )

    node_b_config = NodeConfig(
        node_type=NodeType.TRANSFORM,
        in_ports={"input": Port(type=MockData, desc="Input", mode=["train"])},
        out_ports={"output": Port(type=MockData, desc="Output", mode=["train"])},
    )

    # Create pipeline config
    pipeline_config = PipelineConfig(
        nodes={"A": node_a_config, "B": node_b_config},
        edges=[Edge(source="A", target="B", ports_map={"output": "input"})],
    )

    # Create pipeline
    pipeline = Pipeline(config=pipeline_config)

    # Manually add node objects for testing
    pipeline.node_objects["A"] = MockSourceNode(config=node_a_config)
    pipeline.node_objects["B"] = MockNode(config=node_b_config)

    return pipeline


def create_complex_pipeline() -> Pipeline:
    """Create a more complex pipeline for testing.

    Structure:
        A -> B -> D
        A -> C -> D
    """
    # Create node configs
    node_configs = {}
    for name in ["A", "B", "C", "D"]:
        if name == "A":
            node_configs[name] = NodeConfig(
                node_type=NodeType.SOURCE,
                in_ports={},
                out_ports={"output": Port(type=MockData, desc="Output", mode=["train"])},
            )
        elif name == "D":
            node_configs[name] = NodeConfig(
                node_type=NodeType.TRANSFORM,
                in_ports={
                    "input1": Port(type=MockData, desc="Input 1", mode=["train"]),
                    "input2": Port(type=MockData, desc="Input 2", mode=["train"]),
                },
                out_ports={"output": Port(type=MockData, desc="Output", mode=["train"])},
            )
        else:
            node_configs[name] = NodeConfig(
                node_type=NodeType.TRANSFORM,
                in_ports={"input": Port(type=MockData, desc="Input", mode=["train"])},
                out_ports={"output": Port(type=MockData, desc="Output", mode=["train"])},
            )

    # Create pipeline config
    pipeline_config = PipelineConfig(
        nodes=node_configs,
        edges=[
            Edge(source="A", target="B", ports_map={"output": "input"}),
            Edge(source="A", target="C", ports_map={"output": "input"}),
            Edge(source="B", target="D", ports_map={"output": "input1"}),
            Edge(source="C", target="D", ports_map={"output": "input2"}),
        ],
    )

    # Create pipeline
    pipeline = Pipeline(config=pipeline_config)

    # Manually add node objects for testing
    pipeline.node_objects["A"] = MockSourceNode(config=node_configs["A"])
    for name in ["B", "C", "D"]:
        pipeline.node_objects[name] = MockNode(config=node_configs[name])

    return pipeline


class TestNaivePipelineRunner:
    """Tests for NaivePipelineRunner."""

    def test_initialization(self) -> None:
        """Test that the runner can be initialized."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        assert runner.pipeline == pipeline
        assert runner.config is not None
        assert runner._node_outputs == {}

    def test_compile_pipeline(self) -> None:
        """Test pipeline compilation."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        assert not pipeline.compiled()
        runner._compile_pipeline()
        assert pipeline.compiled()

    def test_get_execution_order_simple(self) -> None:
        """Test execution order for simple pipeline."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        execution_order = runner._get_execution_order()

        assert len(execution_order) == 2
        assert execution_order[0] == "A"
        assert execution_order[1] == "B"

    def test_get_execution_order_complex(self) -> None:
        """Test execution order for complex pipeline."""
        pipeline = create_complex_pipeline()
        runner = NaivePipelineRunner(pipeline)

        execution_order = runner._get_execution_order()

        assert len(execution_order) == 4
        # A must come first
        assert execution_order[0] == "A"
        # B and C must come before D
        assert execution_order.index("B") < execution_order.index("D")
        assert execution_order.index("C") < execution_order.index("D")

    def test_gather_node_inputs_no_predecessors(self) -> None:
        """Test gathering inputs for a node with no predecessors."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        inputs = runner._gather_node_inputs("A")
        assert inputs == {}

    def test_gather_node_inputs_with_predecessors(self) -> None:
        """Test gathering inputs for a node with predecessors."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        # Simulate node A having produced output
        runner._node_outputs["A"] = {"output": MockData(value=5)}

        inputs = runner._gather_node_inputs("B")
        assert "input" in inputs
        assert isinstance(inputs["input"], MockData)
        assert inputs["input"].value == 5

    def test_execute_node_fit_mode(self) -> None:
        """Test executing a node in fit mode."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        node = pipeline.node_objects["B"]
        inputs = {"input": MockData(value=1)}

        result = runner._execute_node("B", inputs, ExecutionMode.FIT)

        assert result is None
        assert node.fit_called
        assert not node.transform_called

    def test_execute_node_transform_mode(self) -> None:
        """Test executing a node in transform mode."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        node = pipeline.node_objects["B"]
        inputs = {"input": MockData(value=1)}

        result = runner._execute_node("B", inputs, ExecutionMode.TRANSFORM)

        assert result is not None
        assert not node.fit_called
        assert node.transform_called

    def test_execute_node_fit_transform_mode(self) -> None:
        """Test executing a node in fit_transform mode."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        node = pipeline.node_objects["B"]
        inputs = {"input": MockData(value=1)}

        result = runner._execute_node("B", inputs, ExecutionMode.FIT_TRANSFORM)

        assert result is not None
        assert node.fit_called
        assert node.transform_called

    def test_run_fit_mode(self) -> None:
        """Test running the pipeline in fit mode."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        outputs = runner.run(mode=ExecutionMode.FIT)

        # In fit mode, no outputs should be stored
        assert outputs == {}

        # But nodes should have been fitted
        assert pipeline.node_objects["A"].fit_called
        assert pipeline.node_objects["B"].fit_called

    def test_run_transform_mode(self) -> None:
        """Test running the pipeline in transform mode."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        outputs = runner.run(mode=ExecutionMode.TRANSFORM)

        # Outputs should be stored
        assert "A" in outputs
        assert "B" in outputs

        # Nodes should have transformed but not fitted
        assert pipeline.node_objects["A"].transform_called
        assert not pipeline.node_objects["A"].fit_called
        assert pipeline.node_objects["B"].transform_called
        assert not pipeline.node_objects["B"].fit_called

    def test_run_fit_transform_mode(self) -> None:
        """Test running the pipeline in fit_transform mode."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        outputs = runner.run(mode=ExecutionMode.FIT_TRANSFORM)

        # Outputs should be stored
        assert "A" in outputs
        assert "B" in outputs

        # Nodes should have both fit and transformed
        assert pipeline.node_objects["A"].fit_called
        assert pipeline.node_objects["A"].transform_called
        assert pipeline.node_objects["B"].fit_called
        assert pipeline.node_objects["B"].transform_called

    def test_train_method(self) -> None:
        """Test the train convenience method."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        runner.train()

        # Should call fit_transform
        assert pipeline.node_objects["A"].fit_called
        assert pipeline.node_objects["A"].transform_called

    def test_fit_method(self) -> None:
        """Test the fit convenience method."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        runner.fit()

        # Should only fit
        assert pipeline.node_objects["A"].fit_called
        assert not pipeline.node_objects["A"].transform_called

    def test_transform_method(self) -> None:
        """Test the transform convenience method."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        outputs = runner.transform()

        # Should only transform
        assert not pipeline.node_objects["A"].fit_called
        assert pipeline.node_objects["A"].transform_called
        assert outputs is not None

    def test_fit_transform_method(self) -> None:
        """Test the fit_transform convenience method."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        outputs = runner.fit_transform()

        # Should fit and transform
        assert pipeline.node_objects["A"].fit_called
        assert pipeline.node_objects["A"].transform_called
        assert outputs is not None

    def test_complex_pipeline_execution(self) -> None:
        """Test execution of a complex pipeline."""
        pipeline = create_complex_pipeline()
        runner = NaivePipelineRunner(pipeline)

        outputs = runner.run(mode=ExecutionMode.FIT_TRANSFORM)

        # All nodes should have outputs
        assert "A" in outputs
        assert "B" in outputs
        assert "C" in outputs
        assert "D" in outputs

        # All nodes should be executed
        for name in ["A", "B", "C", "D"]:
            assert pipeline.node_objects[name].fit_called
            assert pipeline.node_objects[name].transform_called

    def test_invalid_node_raises_error(self) -> None:
        """Test that referencing an invalid node raises an error."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        with pytest.raises(ValueError, match="not found"):
            runner._execute_node("NonExistent", {}, ExecutionMode.FIT)

    def test_cyclic_graph_raises_error(self) -> None:
        """Test that a cyclic graph raises an error."""
        # Create a pipeline with a cycle
        node_config = NodeConfig(
            node_type=NodeType.TRANSFORM,
            in_ports={"input": Port(type=MockData, desc="Input", mode=["train"])},
            out_ports={"output": Port(type=MockData, desc="Output", mode=["train"])},
        )

        pipeline_config = PipelineConfig(
            nodes={"A": node_config, "B": node_config},
            edges=[
                Edge(source="A", target="B", ports_map={"output": "input"}),
                Edge(source="B", target="A", ports_map={"output": "input"}),
            ],
        )

        # This should raise an error during validation
        with pytest.raises(ValueError, match="directed acyclic graph"):
            Pipeline(config=pipeline_config)
