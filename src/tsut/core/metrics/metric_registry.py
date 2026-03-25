"""Metric Registry for the TSUT framework."""

from tsut.core.common.registry import Registry


class MetricRegistry(Registry):
    """Registry for Metrics in the TSUT framework.

    This registry allows for registering and retrieving metrics by name, as well as listing all registered metrics.
    """

    def __init__(self) -> None:
        super().__init__(entity="metric")


METRIC_REGISTRY = MetricRegistry()
