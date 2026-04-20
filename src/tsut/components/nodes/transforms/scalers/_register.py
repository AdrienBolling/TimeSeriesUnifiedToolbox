"""Register scaler transform nodes from this package into the NODE_REGISTRY."""

from tsut.core.nodes.registry.node_registry import NODE_REGISTRY

from .min_max_scaler import (
    MinMaxScaler,
    MinMaxScalerConfig,
    MinMaxScalerHyperParameters,
    MinMaxScalerRunningConfig,
)
from .robust_scaler import (
    RobustScaler,
    RobustScalerConfig,
    RobustScalerHyperParameters,
    RobustScalerRunningConfig,
)
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
    NODE_REGISTRY.register(
        name="MinMaxScaler",
        node_class=MinMaxScaler,
        node_config_class=MinMaxScalerConfig,
        running_config_class=MinMaxScalerRunningConfig,
        hyperparameters_class=MinMaxScalerHyperParameters,
    )
    NODE_REGISTRY.register(
        name="RobustScaler",
        node_class=RobustScaler,
        node_config_class=RobustScalerConfig,
        running_config_class=RobustScalerRunningConfig,
        hyperparameters_class=RobustScalerHyperParameters,
    )
