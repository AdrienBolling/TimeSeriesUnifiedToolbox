"""Module for registering feature selection nodes in the TSUT Framework."""

from tsut.core.nodes.registry.node_registry import NODE_REGISTRY

from .correlation_filter import (
    CorrelationFilter,
    CorrelationFilterConfig,
)
from .missing_rate_filter import (
    MissingRateFilterConfig,
    MissingRateFilterNode,
)
from .remove_categorical import (
    RemoveCategoricalConfig,
    RemoveCategoricalNode,
)
from .variance_filter import (
    VarianceFilter,
    VarianceFilterConfig,
)

# --- Registering the nodes in the registry ---
# Do this for all the nodes defined in this module, so that they can be easily accessed

def register_nodes() -> None:
    """Register all feature selection nodes defined in this module to the NODE_REGISTRY."""
    NODE_REGISTRY.register(
        name="CorrelationFilter",
        node_class=CorrelationFilter,
        node_config_class=CorrelationFilterConfig,
    )

    NODE_REGISTRY.register(
        name="MissingRateFilter",
        node_class=MissingRateFilterNode,
        node_config_class=MissingRateFilterConfig,
    )

    NODE_REGISTRY.register(
        name="VarianceFilter",
        node_class=VarianceFilter,
        node_config_class=VarianceFilterConfig,
    )

    NODE_REGISTRY.register(
        name="RemoveCategorical",
        node_class=RemoveCategoricalNode,
        node_config_class=RemoveCategoricalConfig,
    )

