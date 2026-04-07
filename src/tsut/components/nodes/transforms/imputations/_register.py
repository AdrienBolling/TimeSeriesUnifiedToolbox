"""Register imputation transform nodes from this package into the NODE_REGISTRY."""

from tsut.core.nodes.registry.node_registry import NODE_REGISTRY

from .categorical_imputation import (
    CategoricalImputation,
    CategoricalImputationConfig,
    CategoricalImputationHyperParameters,
    CategoricalImputationRunningConfig,
)
from .numerical_imputation import (
    NumericalImputation,
    NumericalImputationConfig,
    NumericalImputationHyperParameters,
    NumericalImputationRunningConfig,
)


def register_nodes() -> None:
    """Register all imputation nodes defined in this package."""
    NODE_REGISTRY.register(
        name="NumericalImputation",
        node_class=NumericalImputation,
        node_config_class=NumericalImputationConfig,
        running_config_class=NumericalImputationRunningConfig,
        hyperparameters_class=NumericalImputationHyperParameters,
    )
    NODE_REGISTRY.register(
        name="CategoricalImputation",
        node_class=CategoricalImputation,
        node_config_class=CategoricalImputationConfig,
        running_config_class=CategoricalImputationRunningConfig,
        hyperparameters_class=CategoricalImputationHyperParameters,
    )
