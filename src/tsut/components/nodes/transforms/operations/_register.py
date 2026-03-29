"""Module for registering data source nodes in the TSUT Framework."""

from tsut.core.nodes.registry.node_registry import NODE_REGISTRY

from .feature_concatenate import (
    FeatureConcatenate,
    FeatureConcatenateConfig,
)
from .row_concatenate import (
    RowConcatenate,
    RowConcatenateConfig,
)

# --- Registering the nodes in the registry ---
# Do this for all the nodes defined in this module, so that they can be easily accessed

def register_nodes() -> None:
    """Register all transform nodes defined in this module to the NODE_REGISTRY."""
    NODE_REGISTRY.register(
        name="FeatureConcatenate",
        node_class=FeatureConcatenate,
        node_config_class=FeatureConcatenateConfig,
    )

    NODE_REGISTRY.register(
        name="RowConcatenate",
        node_class=RowConcatenate,
        node_config_class=RowConcatenateConfig,
    )
