"""Base Metric class for the TSUT framework."""


from typing import Protocol

from pydantic import BaseModel


class MetricMetadata(BaseModel):
    """Metadata for a Metric in the TSUT framework.

    This class can be extended to include any necessary metadata for a specific metric.
    For example, if a metric has a specific name, description, or other metadata, they can be added here.
    """

    name: str = "BaseMetric"
    description: str = "Base Metric for the TSUT framework."
    task_type: str = "unknown"  # e.g., "classification", "regression", etc.
    data_types: list[str] = []  # e.g., ["tabular", "image", "text", etc.]

class MetricConfig(BaseModel):
    """Configuration for a Metric in the TSUT framework.

    This class can be extended to include any necessary configuration parameters for a specific metric.
    For example, if a metric requires a specific threshold or other parameters, they can be added here.
    """

class MetadataMetricMixin:
    """Mixin class to provide metadata for a Metric in the TSUT framework.

    This mixin can be used to add metadata to any Metric class by simply inheriting from it and defining the `metadata` attribute.
    """

    # Needs to be defined in the subclass
    _metadata: MetricMetadata

    @property
    def metadata(self) -> MetricMetadata:
        """Return the metadata for this Metric."""
        return self._metadata


class Metric(Protocol):
    """The protocol a Metric needs to follow in the TSUT framework."""

    _metadata: MetricMetadata

    @property
    def metadata(self) -> MetricMetadata:
        """Return the metadata for this Metric."""
        return self._metadata
