"""Node Registry implementation."""

from pydantic import BaseModel

from tsut.core.common.registry import Registry
from tsut.core.nodes.node import Node, NodeConfig


class NodeRegistry(Registry):
    """Registry for Nodes in the TSUT framework.

    This registry allows for registering and retrieving nodes by name, as well as listing all registered nodes.
    """

    def __init__(self) -> None:
        super().__init__(entity="node")

    def register(
        self,
        name: str,
        node_class: type[Node],
        node_config_class: type[NodeConfig],
        running_config_class: type[BaseModel] | None= None,
        hyperparameters_class: type[BaseModel] | None = None,
    ) -> None:
        """Register a node in the registry.

        Args:
            name (str): The name of the node to register.
            node_class (type): The class of the node to register.
            node_config_class (type): The class of the node's configuration to register.
            running_config_class (type): The class of the node's running configuration to register.
            hyperparameters_class (type): The class of the node's hyperparameters to register.
        """
        self._register(
            name=name,
            class_callable=node_class,
            config_class_callable=node_config_class,
            extra_fields={**node_class.metadata.model_dump(), "running_config_class": running_config_class, "hyperparameters_class": hyperparameters_class}
        )


NODE_REGISTRY = NodeRegistry()
