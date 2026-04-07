"""TSUT — Time Series Unified Toolbox."""

### Start auto-discovery of components in the components package.
from tsut.core.nodes.registry.node_registry import NODE_REGISTRY as NODE_REGISTRY
print(NODE_REGISTRY)

if NODE_REGISTRY is None:
    raise ImportError("NODE_REGISTRY is not defined. Please ensure that the node registry is properly implemented and imported.")
from tsut.components import _auto_discovery

_auto_discovery()
