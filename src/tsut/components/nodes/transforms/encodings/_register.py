"""Register encoding transform nodes from this package into the NODE_REGISTRY."""

from tsut.core.nodes.registry.node_registry import NODE_REGISTRY

from .one_hot_encoding import (
    OneHotEncoding,
    OneHotEncodingConfig,
    OneHotEncodingHyperParameters,
    OneHotEncodingRunningConfig,
)


def register_nodes() -> None:
    """Register all encoding nodes defined in this package."""
    NODE_REGISTRY.register(
        name="OneHotEncoding",
        node_class=OneHotEncoding,
        node_config_class=OneHotEncodingConfig,
        running_config_class=OneHotEncodingRunningConfig,
        hyperparameters_class=OneHotEncodingHyperParameters,
    )
