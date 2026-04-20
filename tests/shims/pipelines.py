"""Pre-built compiled pipelines for use in tests.

These helpers wire a few common pipeline topologies (source → model → sink,
with or without a metric) and return a compiled :class:`Pipeline` ready to
train / evaluate / infer.
"""

from __future__ import annotations

from tsut.components.nodes.data_sources.inputs_passthrough import (
    InputsPassthroughConfig,
)
from tsut.core.common.data.data import (
    ArrayLikeEnum,
    DataCategoryEnum,
    DataStructureEnum,
)
from tsut.core.common.enums import NodeExecutionMode
from tsut.core.nodes.data_sink.sink import SinkConfig
from tsut.core.nodes.node import NodeConfig, Port
from tsut.core.nodes.registry.node_registry import NODE_REGISTRY
from tsut.core.pipeline.pipeline import Edge, Pipeline, PipelineConfig


def _inputs_passthrough_config(
    *,
    with_targets: bool = True,
) -> InputsPassthroughConfig:
    """Config with X and (optionally) y output ports."""
    ports: dict[str, Port] = {
        "X": Port(
            arr_type=ArrayLikeEnum.PANDAS,
            data_structure=DataStructureEnum.TABULAR,
            data_category=DataCategoryEnum.NUMERICAL,
            data_shape="batch feature",
            desc="Feature matrix forwarded from the caller.",
        ),
    }
    if with_targets:
        ports["y"] = Port(
            arr_type=ArrayLikeEnum.PANDAS,
            data_structure=DataStructureEnum.TABULAR,
            data_category=DataCategoryEnum.NUMERICAL,
            data_shape="batch 1",
            desc="Targets (training/evaluation only).",
            mode=[NodeExecutionMode.TRAINING, NodeExecutionMode.EVALUATION],
        )
    return InputsPassthroughConfig(out_ports=ports)


def _sink_config_with_pred() -> SinkConfig:
    """Sink with a ``pred`` in-port."""
    return SinkConfig(
        in_ports={
            "pred": Port(
                arr_type=ArrayLikeEnum.PANDAS,
                data_structure=DataStructureEnum.TABULAR,
                data_category=DataCategoryEnum.NUMERICAL,
                data_shape="batch 1",
                desc="Predicted values reaching the sink.",
            ),
        },
    )


def _default_config_for(node_type: str) -> NodeConfig:
    """Return a default ``NodeConfig`` instance for a registered node type."""
    return NODE_REGISTRY.get_node_config_class(node_type)()


def build_source_model_sink_pipeline(
    *,
    model_node_type: str = "LinearRegression",
    model_config: NodeConfig | None = None,
    with_metric: bool = False,
    metric_node_type: str = "MSE",
    metric_config: NodeConfig | None = None,
    name: str = "test_pipeline",
) -> Pipeline:
    """Build and compile a small source → model → sink pipeline.

    When *with_metric* is ``True`` a metric node is attached to the model's
    ``pred`` port and the source's ``y`` port, making the pipeline
    ``evaluate``-capable.

    The pipeline is constructed via ``PipelineConfig`` so the sink is
    discovered at construction time (``add_node`` would also work, but the
    config-first pattern mirrors the public examples).
    """
    model_cfg = model_config or _default_config_for(model_node_type)
    nodes: dict[str, tuple[str, NodeConfig]] = {
        "source": ("InputsPassthrough", _inputs_passthrough_config(with_targets=True)),
        "model": (model_node_type, model_cfg),
        "sink": ("Sink", _sink_config_with_pred()),
    }
    edges: list[Edge] = [
        Edge(source="source", target="model", ports_map=[("X", "X"), ("y", "y")]),
        Edge(source="model", target="sink", ports_map=[("pred", "pred")]),
    ]

    if with_metric:
        metric_cfg = metric_config or _default_config_for(metric_node_type)
        nodes["metric"] = (metric_node_type, metric_cfg)
        edges.append(
            Edge(source="model", target="metric", ports_map=[("pred", "pred")])
        )
        edges.append(
            Edge(source="source", target="metric", ports_map=[("y", "target")])
        )

    pipeline = Pipeline(
        config=PipelineConfig(name=name, nodes=nodes, edges=edges),
    )
    pipeline.compile()
    return pipeline
