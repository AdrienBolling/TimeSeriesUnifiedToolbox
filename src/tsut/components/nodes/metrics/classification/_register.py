"""Register classification metric nodes from this package into the NODE_REGISTRY."""

from tsut.core.nodes.registry.node_registry import NODE_REGISTRY

from .accuracy import AccuracyConfig, AccuracyNode, AccuracyRunningConfig
from .auroc import AUROCConfig, AUROCNode, AUROCRunningConfig
from .f1_score import F1, F1ScoreConfig, F1ScoreRunningConfig


def register_nodes() -> None:
    """Register all classification metric nodes defined in this package."""
    NODE_REGISTRY.register(
        name="Accuracy",
        node_class=AccuracyNode,
        node_config_class=AccuracyConfig,
        running_config_class=AccuracyRunningConfig,
    )
    NODE_REGISTRY.register(
        name="F1Score",
        node_class=F1,
        node_config_class=F1ScoreConfig,
        running_config_class=F1ScoreRunningConfig,
    )
    NODE_REGISTRY.register(
        name="AUROC",
        node_class=AUROCNode,
        node_config_class=AUROCConfig,
        running_config_class=AUROCRunningConfig,
    )
