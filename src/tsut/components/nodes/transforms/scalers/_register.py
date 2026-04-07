"""Register scaler transform nodes from this package into the NODE_REGISTRY."""

from tsut.core.nodes.registry.node_registry import NODE_REGISTRY

from .standard_scaler import (
    StandardScaler,
    StandardScalerConfig,
    StandardScalerHyperParameters,
    StandardScalerRunningConfig,
)


def register_nodes() -> None:
    """Register all scaler nodes defined in this package."""
    NODE_REGISTRY.register(
        name="StandardScaler",
        node_class=StandardScaler,
        node_config_class=StandardScalerConfig,
        running_config_class=StandardScalerRunningConfig,
        hyperparameters_class=StandardScalerHyperParameters,
    )
