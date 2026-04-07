"""Module for the Metric node wrapper."""

from abc import abstractmethod
from typing import TypeVar

from pydantic import BaseModel
from torchmetrics import Metric

from tsut.core.nodes.node import Node, NodeConfig, NodeMetadata, NodeType, Port


class MetricNodeRunningConfig(BaseModel):
    """Running configuration for a Metric Node in a TSUT Pipeline."""


R = TypeVar("R", bound=MetricNodeRunningConfig)


class MetricNodeConfig[R](NodeConfig):
    """Configuration for a Metric Node in a TSUT Pipeline.

    Generic over ``R``, which must be a :class:`MetricNodeRunningConfig`
    subclass carrying metric-specific runtime parameters.

    Attributes:
        node_type: Always ``NodeType.METRIC``.
        in_ports: Input port definitions for the metric node.
        out_ports: Output port definitions for the metric node.
        running_config: Optional runtime parameters for metric execution.

    """

    node_type: NodeType = NodeType.METRIC
    in_ports: dict[str, Port] = {}
    out_ports: dict[str, Port] = {}
    running_config: R | None = None


class MetricNodeMetadata(NodeMetadata):
    """Metadata for a Metric Node in a TSUT Pipeline."""


class MetricNode[D_I, D_C_I, D_O, D_C_O](Node[D_I, D_C_I, D_O, D_C_O]):
    """Node wrapper for an accumulator-style metric in a TSUT Pipeline.

    Metrics follow an **update/compute** pattern: :meth:`update` accumulates
    state from incoming batches and :meth:`compute` produces the final result.
    :meth:`node_transform` calls both in sequence.
    """

    metadata = MetricNodeMetadata()

    def __init__(self, *, config: MetricNodeConfig, metric: Metric) -> None:
        """Initialize the Metric Node with the given configuration and metric."""

    # --- Methods to implement for the metric node ---

    @abstractmethod
    def update(self, data: dict[str, tuple[D_I, D_C_I]]) -> None:
        """Accumulate metric state from the given batch of data.

        Args:
            data: Mapping of input port name to ``(data, context)`` tuples.

        """

    @abstractmethod
    def compute(self) -> dict[str, tuple[D_O, D_C_O]]:
        """Compute and return the final metric value from accumulated state.

        Returns:
            Mapping of output port name to ``(data, context)`` tuples.

        """

    # --- Node API override don't touch that ---

    def node_fit(self, data: dict[str, tuple[D_I, D_C_I]]) -> None:
        """Fit the Metric Node. This method will be called during the training phase of the pipeline."""

    def node_transform(
        self, data: dict[str, tuple[D_I, D_C_I]]
    ) -> dict[str, tuple[D_O, D_C_O]]:
        """Transform with the Metric Node. This method will be called during all phase of the pipeline."""
        # Metrics typically do not produce output data in the same way as other nodes, so this can be left empty or return an empty dict.
        self.update(data)
        return self.compute()
