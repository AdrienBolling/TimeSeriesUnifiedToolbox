"""Register data-source nodes from this package into the NODE_REGISTRY."""

from tsut.core.nodes.registry.node_registry import NODE_REGISTRY

from .inputs_passthrough import (
    InputsPassthrough,
    InputsPassthroughConfig,
    InputsPassthroughRunningConfig,
)
from .tabular_csv_fetcher import (
    TabularCSVFetcher,
    TabularCSVFetcherConfig,
    TabularCSVFetcherRunningConfig,
)


def register_nodes() -> None:
    """Register all data-source nodes defined in this package."""
    NODE_REGISTRY.register(
        name="TabularCSVFetcher",
        node_class=TabularCSVFetcher,
        node_config_class=TabularCSVFetcherConfig,
        running_config_class=TabularCSVFetcherRunningConfig,
    )
    NODE_REGISTRY.register(
        name="InputsPassthrough",
        node_class=InputsPassthrough,
        node_config_class=InputsPassthroughConfig,
        running_config_class=InputsPassthroughRunningConfig,
    )
