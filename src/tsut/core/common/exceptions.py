"""Exceptions module."""


# --- Pipeline exceptions ---
class PipelineError(Exception):
    """Base class for all exceptions raised by the Pipeline."""


class PipelineValidationError(PipelineError):
    """Exception raised when the pipeline configuration is invalid."""


class PipelineCompilationError(PipelineError):
    """Exception raised when there is an error during pipeline compilation."""


class PipelineGraphError(PipelineError):
    """Exception raised when there is an error in the pipeline graph (e.g., cycles, disconnected components)."""


# --- Node exceptions ---
class NodeError(Exception):
    """Base class for all exceptions raised by Nodes."""
