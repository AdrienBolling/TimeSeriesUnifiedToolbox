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
    
    def node_fit_transform(
        self, data: dict[str, MockData | ContextData]
    ) -> dict[str, MockData | ContextData]:
        """Mock fit_transform implementation."""
        self.fit_called = True
        return self.node_transform(data)


class MockSourceNode(MockNode):
    """Mock source node that generates data."""

    def node_transform(
        self, data: dict[str, MockData | ContextData]
    ) -> dict[str, MockData | ContextData]:
        """Generate mock data."""
        self.transform_called = True
        return {"output": MockData(value=1)}
    
    def node_fit_transform(
        self, data: dict[str, MockData | ContextData]
    ) -> dict[str, MockData | ContextData]:
        """Mock fit_transform implementation."""
        self.fit_called = True
        return self.node_transform(data)


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

    def test_execute_node_train_mode(self) -> None:
        """Test executing a node in train mode."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        node = pipeline.node_objects["B"]
        inputs = {"input": MockData(value=1)}

        result = runner._execute_node("B", inputs, ExecutionMode.TRAIN)

        assert result is not None
        assert node.fit_called
        assert node.transform_called

    def test_execute_node_predict_mode(self) -> None:
        """Test executing a node in predict mode."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        node = pipeline.node_objects["B"]
        inputs = {"input": MockData(value=1)}

        result = runner._execute_node("B", inputs, ExecutionMode.PREDICT)

        assert result is not None
        assert not node.fit_called
        assert node.transform_called

    def test_execute_node_evaluate_mode(self) -> None:
        """Test executing a node in evaluate mode."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        node = pipeline.node_objects["B"]
        inputs = {"input": MockData(value=1)}

        result = runner._execute_node("B", inputs, ExecutionMode.EVALUATE)

        assert result is not None
        assert not node.fit_called
        assert node.transform_called

    def test_run_train_mode(self) -> None:
        """Test running the pipeline in train mode."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        outputs = runner.run(mode=ExecutionMode.TRAIN)

        # In train mode, all outputs should be stored
        assert "A" in outputs
        assert "B" in outputs

        # Nodes should have been fitted and transformed
        assert pipeline.node_objects["A"].fit_called
        assert pipeline.node_objects["A"].transform_called
        assert pipeline.node_objects["B"].fit_called
        assert pipeline.node_objects["B"].transform_called

    def test_run_predict_mode(self) -> None:
        """Test running the pipeline in predict mode."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        outputs = runner.run(mode=ExecutionMode.PREDICT)

        # Outputs should be stored
        assert "A" in outputs
        assert "B" in outputs

        # Nodes should have transformed but not fitted
        assert pipeline.node_objects["A"].transform_called
        assert not pipeline.node_objects["A"].fit_called
        assert pipeline.node_objects["B"].transform_called
        assert not pipeline.node_objects["B"].fit_called

    def test_train_method(self) -> None:
        """Test the train convenience method."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        outputs = runner.train()

        # Should fit and transform
        assert pipeline.node_objects["A"].fit_called
        assert pipeline.node_objects["A"].transform_called
        assert outputs is not None
        assert "A" in outputs
        assert "B" in outputs

    def test_predict_method(self) -> None:
        """Test the predict convenience method."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        outputs = runner.predict()

        # Should only transform
        assert not pipeline.node_objects["A"].fit_called
        assert pipeline.node_objects["A"].transform_called
        assert outputs is not None

    def test_evaluate_method(self) -> None:
        """Test the evaluate convenience method."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        outputs = runner.evaluate()

        # Should only transform (same as predict for now)
        assert not pipeline.node_objects["A"].fit_called
        assert pipeline.node_objects["A"].transform_called
        assert outputs is not None
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

        outputs = runner.run(mode=ExecutionMode.TRAIN)

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
            runner._execute_node("NonExistent", {}, ExecutionMode.TRAIN)

    def test_cyclic_graph_raises_error(self) -> None:
        """Test that a cyclic graph raises an error."""
        # Create separate node configs for each node to avoid ID conflicts
        node_a_config = NodeConfig(
            node_type=NodeType.TRANSFORM,
            in_ports={"input": Port(type=MockData, desc="Input", mode=["train"])},
            out_ports={"output": Port(type=MockData, desc="Output", mode=["train"])},
        )
        
        node_b_config = NodeConfig(
            node_type=NodeType.TRANSFORM,
            in_ports={"input": Port(type=MockData, desc="Input", mode=["train"])},
            out_ports={"output": Port(type=MockData, desc="Output", mode=["train"])},
        )

        pipeline_config = PipelineConfig(
            nodes={"A": node_a_config, "B": node_b_config},
            edges=[
                Edge(source="A", target="B", ports_map={"output": "input"}),
                Edge(source="B", target="A", ports_map={"output": "input"}),
            ],
        )

        # This should raise an error during validation
        with pytest.raises(ValueError, match="directed acyclic graph"):
            Pipeline(config=pipeline_config)

    def test_get_sink_outputs(self) -> None:
        """Test getting sink/leaf node outputs."""
        pipeline = create_simple_pipeline()
        runner = NaivePipelineRunner(pipeline)

        # Run the pipeline
        runner.run(mode=ExecutionMode.TRAIN)

        # Get sink outputs (B is a leaf node - no successors)
        sink_outputs = runner.get_sink_outputs()

        # B should be in sink outputs as it has no successors
        assert "B" in sink_outputs
        # A should not be in sink outputs as it has a successor (B)
        assert "A" not in sink_outputs

    def test_get_sink_outputs_complex(self) -> None:
        """Test getting sink outputs from complex pipeline."""
        pipeline = create_complex_pipeline()
        runner = NaivePipelineRunner(pipeline)

        # Run the pipeline
        runner.run(mode=ExecutionMode.TRAIN)

        # Get sink outputs
        sink_outputs = runner.get_sink_outputs()

        # D should be in sink outputs as it has no successors
        assert "D" in sink_outputs
        # A, B, C should not be in sink outputs as they have successors
        assert "A" not in sink_outputs
        assert "B" not in sink_outputs
        assert "C" not in sink_outputs
