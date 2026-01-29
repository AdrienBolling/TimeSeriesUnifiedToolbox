"""Define the base class for a TSUT PipelineRunner.

This class serves as a blueprint for implementing various pipeline execution strategies.
"""

from typing import Protocol

from pydantic import BaseModel

from tsut.core.pipeline.base import Pipeline


class RunnerConfig(BaseModel):
    """Define the configuration schema for a PipelineRunner."""


class PipelineRunner(Protocol):
    """Define the interface for a TSUT PipelineRunner.

    The goal of the PipelineRunner is to encapsulate all logic related to effectively run a pipeline. (Such as training, tuning, evaluating, etc.)
    It is the object effectively upon from which an user effectively interacts with a pipeline. (train/evaluate/tune/etc.)
    """

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        config: RunnerConfig,
    ) -> None:
        """Initialize the PipelineRunner with a pipeline and configuration."""
        ...

    def train(self) -> None:
        """Train the pipeline."""
        ...
