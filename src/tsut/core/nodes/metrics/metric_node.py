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
    """Configuration for a Metric Node in a TSUT Pipeline."""

    node_type: NodeType = NodeType.METRIC
    in_ports: dict[str, Port] = {}  # Define the input ports for the metric node
    out_ports: dict[str, Port] = {}  # Define the output ports for the metric node
    running_config: R | None = None  # Running configuration for the metric node


class MetricNodeMetadata(NodeMetadata):
    """Metadata for a Metric Node in a TSUT Pipeline."""


class MetricNode[D_I, D_C_I, D_O, D_C_O](Node[D_I, D_C_I, D_O, D_C_O]):
    """A Node wrapper for a Metric in a TSUT Pipeline."""

    metadata = MetricNodeMetadata()

    def __init__(self, *, config: MetricNodeConfig, metric: Metric) -> None:
        """Initialize the Metric Node with the given configuration and metric."""

    # --- Methods to implement for the metric node ---

    @abstractmethod
    def update(self, data: dict[str, tuple[D_I, D_C_I]]) -> None:
        """Update the metric with the given data."""

    @abstractmethod
    def compute(self) -> dict[str, tuple[D_O, D_C_O]]:
        """Compute the metric with the given data."""

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
