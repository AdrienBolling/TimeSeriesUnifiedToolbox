"""Register classification metric nodes from this package into the NODE_REGISTRY."""

from tsut.core.nodes.registry.node_registry import NODE_REGISTRY

from .accuracy import AccuracyConfig, AccuracyNode, AccuracyRunningConfig
from .auroc import AUROCConfig, AUROCNode, AUROCRunningConfig
from .f1_score import F1, F1ScoreConfig, F1ScoreRunningConfig
from .precision import PrecisionConfig, PrecisionNode, PrecisionRunningConfig
from .recall import RecallConfig, RecallNode, RecallRunningConfig


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
    NODE_REGISTRY.register(
        name="Precision",
        node_class=PrecisionNode,
        node_config_class=PrecisionConfig,
        running_config_class=PrecisionRunningConfig,
    )
    NODE_REGISTRY.register(
        name="Recall",
        node_class=RecallNode,
        node_config_class=RecallConfig,
        running_config_class=RecallRunningConfig,
    )
