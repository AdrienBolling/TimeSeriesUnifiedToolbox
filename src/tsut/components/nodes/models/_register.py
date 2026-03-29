"""Module for registering model nodes in the TSUT Framework."""

from tsut.core.nodes.registry.node_registry import NODE_REGISTRY

from .random_forest_regressor import (
    RandomForestRegressorConfig,
    RandomForestRegressorNode,
)

# --- Registering the nodes in the registry ---
# Do this for all the nodes defined in this module, so that they can be easily accessed and used in the TSUT Framework.

def register_nodes() -> None:
    """Register all model nodes defined in this module to the NODE_REGISTRY."""
    NODE_REGISTRY.register(
        name="RandomForestRegressor",
        node_class=RandomForestRegressorNode,
        node_config_class=RandomForestRegressorConfig,
    )