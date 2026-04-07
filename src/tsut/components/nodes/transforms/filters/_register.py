"""Register outlier-filter transform nodes from this package into the NODE_REGISTRY."""

from tsut.core.nodes.registry.node_registry import NODE_REGISTRY

from .iqr_outlier_filter import (
    IQROutlierFilter,
    IQROutlierFilterConfig,
    IQROutlierFilterHyperParameters,
    IQROutlierFilterRunningConfig,
)
from .zscore_outlier_filter import (
    ZScoreOutlierFilter,
    ZScoreOutlierFilterConfig,
    ZScoreOutlierFilterHyperParameters,
    ZScoreOutlierFilterRunningConfig,
)


def register_nodes() -> None:
    """Register all outlier-filter nodes defined in this package."""
    NODE_REGISTRY.register(
        name="ZScoreOutlierFilter",
        node_class=ZScoreOutlierFilter,
        node_config_class=ZScoreOutlierFilterConfig,
        running_config_class=ZScoreOutlierFilterRunningConfig,
        hyperparameters_class=ZScoreOutlierFilterHyperParameters,
    )
    NODE_REGISTRY.register(
        name="IQROutlierFilter",
        node_class=IQROutlierFilter,
        node_config_class=IQROutlierFilterConfig,
        running_config_class=IQROutlierFilterRunningConfig,
        hyperparameters_class=IQROutlierFilterHyperParameters,
    )
