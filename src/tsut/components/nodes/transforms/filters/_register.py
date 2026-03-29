"""Module for registering filter nodes in the TSUT Framework."""

from tsut.core.nodes.registry.node_registry import NODE_REGISTRY
from .outlier_filter import (
    OutlierFilterConfig,
    OutlierFilter,
)

# --- Registering the nodes in the registry ---
# Do this for all the nodes defined in this module, so that they can be easily accessed

def register_nodes() -> None:
    """Register all filter nodes defined in this module to the NODE_REGISTRY."""
    NODE_REGISTRY.register(
        "OutlierFilter",
        OutlierFilter,
        OutlierFilterConfig,
    )