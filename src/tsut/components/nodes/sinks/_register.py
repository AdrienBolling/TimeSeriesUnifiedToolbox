"""Module for registering data source nodes in the TSUT Framework."""

from tsut.core.nodes.registry.node_registry import NODE_REGISTRY

from tsut.core.nodes.data_sink.sink import (
    SinkConfig,
    Sink,
)

# --- Registering the nodes in the registry ---
# Do this for all the nodes defined in this module, so that they can be easily accessed and used in the TSUT Framework.
def register_nodes() -> None:
    """Register all data source nodes defined in this module to the NODE_REGISTRY."""
    NODE_REGISTRY.register(
        name="Sink",
        node_class=Sink,
        node_config_class=SinkConfig,
    )


