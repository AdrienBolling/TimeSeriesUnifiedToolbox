"""Register model nodes from this package into the NODE_REGISTRY."""

from tsut.core.nodes.registry.node_registry import NODE_REGISTRY

from .random_forest_regressor import (
    RandomForestRegressorConfig,
    RandomForestRegressorNode,
    RandomForestRegressorRunningConfig,
    RandomForestRegressorHyperParameters,
)


def register_nodes() -> None:
    """Register all model nodes defined in this package."""
    NODE_REGISTRY.register(
        name="RandomForestRegressor",
        node_class=RandomForestRegressorNode,
        node_config_class=RandomForestRegressorConfig,
        running_config_class=RandomForestRegressorRunningConfig,
        hyperparameters_class=RandomForestRegressorHyperParameters,
    )
