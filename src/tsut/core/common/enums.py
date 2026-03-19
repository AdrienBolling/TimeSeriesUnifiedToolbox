from enum import Enum, StrEnum, auto

class NodeExecutionMode(StrEnum):
    """Define execution modes for nodes in the pipeline."""

    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
    DEFAULT = "default" # Placeholder for unset mode to raise errors if used without being set