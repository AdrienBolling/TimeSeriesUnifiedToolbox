"""Define the base class for a TSUT PipelineRunner.

This class serves as a blueprint for implementing various pipeline execution strategies.
"""

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from tsut.core.common.enums import NodeExecutionMode
from tsut.core.pipeline.pipeline import Pipeline

D_O = TypeVar("D_O")  # Output data type of the pipeline execution
M = TypeVar("M")  # Metrics output type for evaluation results

class RunnerConfig(BaseModel):
    """Define the configuration schema for a PipelineRunner."""


class PipelineRunner[D_O, M](ABC):
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
        self._pipeline = pipeline
        self._config = config
        self._mode = NodeExecutionMode.DEFAULT

    # --- Convenience API

    @property
    def pipeline(self) -> Pipeline:
        """Get the pipeline associated with this runner."""
        return self._pipeline

    @property
    def config(self) -> RunnerConfig:
        """Get the configuration of this runner."""
        return self._config

    @property
    def mode(self) -> str:
        """Get the current execution mode of the runner. The mode is supposed to be set automatically according to the function used."""
        return self._mode

    # --- API to implement for any PipelineRunner implementation ---
    # INFO : Note the absence of a 'tune' method, as I think it is best suited to be part of a Wrapper for PipelineRunners, rather than the PipelineRunner itself. 
    # We will see where it belongs on the long run.

    @abstractmethod
    def train(self) -> None:
        """Train the pipeline."""
        ...

    @abstractmethod
    def evaluate(self) -> M:
        """Evaluate the pipeline."""
        ...

    @abstractmethod
    def infer(self) -> D_O:
        """Run inference with the pipeline."""
        ...
