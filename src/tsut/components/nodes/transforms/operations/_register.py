"""Register operation transform nodes from this package into the NODE_REGISTRY."""

from tsut.core.nodes.registry.node_registry import NODE_REGISTRY

from .column_order import ColumnOrder, ColumnOrderConfig, ColumnOrderRunningConfig
from .feature_concatenate import FeatureConcatenate, FeatureConcatenateConfig
from .row_concatenate import RowConcatenate, RowConcatenateConfig


def register_nodes() -> None:
    """Register all operation transform nodes defined in this package."""
    NODE_REGISTRY.register(
        name="RowConcatenate",
        node_class=RowConcatenate,
        node_config_class=RowConcatenateConfig,
    )
    NODE_REGISTRY.register(
        name="FeatureConcatenate",
        node_class=FeatureConcatenate,
        node_config_class=FeatureConcatenateConfig,
    )
    NODE_REGISTRY.register(
        name="ColumnOrder",
        node_class=ColumnOrder,
        node_config_class=ColumnOrderConfig,
        running_config_class=ColumnOrderRunningConfig,
    )
