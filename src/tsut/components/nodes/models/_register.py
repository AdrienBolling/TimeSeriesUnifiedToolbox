"""Register model nodes from this package into the NODE_REGISTRY."""

from tsut.core.nodes.registry.node_registry import NODE_REGISTRY

from .cnn import (
    CNNConfig,
    CNNHyperParameters,
    CNNNode,
    CNNRunningConfig,
)
from .gradient_boosting_classifier import (
    GradientBoostingClassifierConfig,
    GradientBoostingClassifierHyperParameters,
    GradientBoostingClassifierNode,
    GradientBoostingClassifierRunningConfig,
)
from .gradient_boosting_regressor import (
    GradientBoostingRegressorConfig,
    GradientBoostingRegressorHyperParameters,
    GradientBoostingRegressorNode,
    GradientBoostingRegressorRunningConfig,
)
from .linear_regression import (
    LinearRegressionConfig,
    LinearRegressionHyperParameters,
    LinearRegressionNode,
    LinearRegressionRunningConfig,
)
from .mlp import (
    MLPConfig,
    MLPHyperParameters,
    MLPNode,
    MLPRunningConfig,
)
from .random_forest_classifier import (
    RandomForestClassifierConfig,
    RandomForestClassifierHyperParameters,
    RandomForestClassifierNode,
    RandomForestClassifierRunningConfig,
)
from .random_forest_regressor import (
    RandomForestRegressorConfig,
    RandomForestRegressorHyperParameters,
    RandomForestRegressorNode,
    RandomForestRegressorRunningConfig,
)


def register_nodes() -> None:
    """Register all model nodes defined in this package."""
    NODE_REGISTRY.register(
        name="LinearRegression",
        node_class=LinearRegressionNode,
        node_config_class=LinearRegressionConfig,
        running_config_class=LinearRegressionRunningConfig,
        hyperparameters_class=LinearRegressionHyperParameters,
    )
    NODE_REGISTRY.register(
        name="RandomForestRegressor",
        node_class=RandomForestRegressorNode,
        node_config_class=RandomForestRegressorConfig,
        running_config_class=RandomForestRegressorRunningConfig,
        hyperparameters_class=RandomForestRegressorHyperParameters,
    )
    NODE_REGISTRY.register(
        name="RandomForestClassifier",
        node_class=RandomForestClassifierNode,
        node_config_class=RandomForestClassifierConfig,
        running_config_class=RandomForestClassifierRunningConfig,
        hyperparameters_class=RandomForestClassifierHyperParameters,
    )
    NODE_REGISTRY.register(
        name="GradientBoostingRegressor",
        node_class=GradientBoostingRegressorNode,
        node_config_class=GradientBoostingRegressorConfig,
        running_config_class=GradientBoostingRegressorRunningConfig,
        hyperparameters_class=GradientBoostingRegressorHyperParameters,
    )
    NODE_REGISTRY.register(
        name="GradientBoostingClassifier",
        node_class=GradientBoostingClassifierNode,
        node_config_class=GradientBoostingClassifierConfig,
        running_config_class=GradientBoostingClassifierRunningConfig,
        hyperparameters_class=GradientBoostingClassifierHyperParameters,
    )
    NODE_REGISTRY.register(
        name="MLP",
        node_class=MLPNode,
        node_config_class=MLPConfig,
        running_config_class=MLPRunningConfig,
        hyperparameters_class=MLPHyperParameters,
    )
    NODE_REGISTRY.register(
        name="CNN",
        node_class=CNNNode,
        node_config_class=CNNConfig,
        running_config_class=CNNRunningConfig,
        hyperparameters_class=CNNHyperParameters,
    )
