"""Module for registering data source nodes in the TSUT Framework."""

from tsut.core.nodes.registry.node_registry import NODE_REGISTRY

from .simple_numerical_imputation import (
    SimpleNumericalImputationConfig,
    SimpleNumericalImputationNode,
)

# --- Registering the nodes in the registry ---
# Do this for all the nodes defined in this module, so that they can be easily accessed

def register_nodes() -> None:
    """Register all imputation nodes defined in this module to the NODE_REGISTRY."""
    NODE_REGISTRY.register(
        name="SimpleNumericalImputation",
        node_class=SimpleNumericalImputationNode,
        node_config_class=SimpleNumericalImputationConfig,
    )

