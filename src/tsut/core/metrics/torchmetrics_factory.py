"""Common Factory to quickly convert torchmetrics to tsut Metrics."""

from typing import TypeVar

from torchmetrics import Metric as TorchMetric

from tsut.core.metrics.metric import MetadataMetricMixin, Metric, MetricMetadata

TMetric = TypeVar("TMetric", bound=type[TorchMetric])


def wrap_metric_class(
    metric_cls: type[TorchMetric],
    *,
    metadata: MetricMetadata,
    name: str | None = None,
) -> type[Metric]:
    """Wrap a torchmetrics.Metric class with the MetadataMetricMixin to create a TSUT Metric class.

    Args:
        metric_cls: The torchmetrics.Metric class to wrap.
        metadata: The MetricMetadata to attach to the wrapped class.
        name: Optional name for the wrapped class. If not provided, it will be "My{metric_cls.__name__}".

    Returns:
        A new class that inherits from both metric_cls and MetadataMetricMixin, with the provided metadata.

    Raises:
        TypeError: If metric_cls does not inherit from torchmetrics.Metric.

    """
    if not issubclass(metric_cls, TorchMetric):
        raise TypeError(f"{metric_cls} must inherit from torchmetrics.Metric")

    cls_name = name or f"{metric_cls.__name__}"

    return type(
        cls_name,
        (metric_cls, MetadataMetricMixin),
        {
            "metadata_model": type(metadata),
            "metadata": metadata,
            "__module__": metric_cls.__module__,
        },
    )

def metric_with_metadata(
    metadata: MetricMetadata,
    *,
    name: str | None = None,
):
    """Decorate to wrap a torchmetrics.Metric class with the MetadataMetricMixin to create a TSUT Metric class."""

    def decorator(metric_cls: type[TorchMetric]) -> type[Metric]:
        return wrap_metric_class(metric_cls, metadata=metadata, name=name)

    return decorator