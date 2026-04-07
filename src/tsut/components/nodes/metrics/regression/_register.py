"""Register regression metric nodes from this package into the NODE_REGISTRY."""

from tsut.core.nodes.registry.node_registry import NODE_REGISTRY

from .mae import MAE, MAEConfig, MAERunningConfig
from .mse import MSE, MSEConfig, MSERunningConfig
from .r2_score import R2, R2ScoreConfig, R2ScoreRunningConfig


def register_nodes() -> None:
    """Register all regression metric nodes defined in this package."""
    NODE_REGISTRY.register(
        name="MSE",
        node_class=MSE,
        node_config_class=MSEConfig,
        running_config_class=MSERunningConfig,
    )
    NODE_REGISTRY.register(
        name="MAE",
        node_class=MAE,
        node_config_class=MAEConfig,
        running_config_class=MAERunningConfig,
    )
    NODE_REGISTRY.register(
        name="R2Score",
        node_class=R2,
        node_config_class=R2ScoreConfig,
        running_config_class=R2ScoreRunningConfig,
    )
