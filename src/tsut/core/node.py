"""Define the atomic component of a TSUT Pipeline: the Node."""

from enum import StrEnum

from pydantic import BaseModel


class NodeType(StrEnum):
    """Define the types of Nodes available in a TSUT Pipeline."""

    BASE = "base"
    SOURCE = "source"
    SINK = "sink"
    TRANSFORM = "transform"
    MODEL = "model"


class NodeConfig(BaseModel):
    """Configuration for a Node in a TSUT Pipeline."""

    node_type: NodeType = NodeType.BASE
    in_ports: list[str] = []
    out_ports: list[str] = []


class Node:
    """Base class for a Node in a TSUT Pipeline."""

    def __init__(self, config: NodeConfig) -> None:
        """Initialize the Node with the given configuration.

        The only common denomination across all Nodes are their input and output ports.
        """
        self.node_type: NodeType = config.node_type
        self.in_ports: list[str] = config.in_ports
        self.out_ports: list[str] = config.out_ports
