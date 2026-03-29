"""Base wrapper class for PipelineRunners."""

from abc import ABC, abstractmethod

from tsut.core.pipeline.runners.pipeline_runner import PipelineRunner
from pydantic_settings import BaseSettings
from tsut.core.nodes.node import Node

from typing import Any

class PipelineRunnerWrapperSettings(BaseSettings):
    """Define the configuration schema for a PipelineRunnerWrapper."""

class PipelineRunnerWrapper[D_O, M](PipelineRunner[D_O, M], ABC):
    """Define the interface for a PipelineRunnerWrapper.

    The goal of the PipelineRunnerWrapper is to wrap around a PipelineRunner and add additional functionality to it. (Such as tuning, logging, etc.)
    It is the object upon which an user interacts with when they want to use the additional functionality provided by the wrapper.
    """

    def __init__(
        self,
        pipeline_runner: PipelineRunner[D_O, M],
        *,
        settings: PipelineRunnerWrapperSettings,
    ) -> None:
        """Initialize the PipelineRunnerWrapper with a pipeline runner and configuration."""
        self._pipeline_runner = pipeline_runner
        self._settings = settings

    # --- Convenience API

    @property
    def pipeline_runner(self) -> PipelineRunner[D_O, M]:
        """Get the pipeline runner associated with this wrapper."""
        return self._pipeline_runner

    @property
    def unwrapped(self) -> PipelineRunner:
        """Get the original, unwrapped pipeline runner."""
        return self._pipeline_runner.unwrapped if hasattr(self._pipeline_runner, 'unwrapped') else self._pipeline_runner

    @property
    def settings(self) -> PipelineRunnerWrapperSettings:
        """Get the settings of this wrapper."""
        return (self._settings)

    @property
    def config(self) -> PipelineRunnerWrapperSettings:
        """Get the configuration of this wrapper."""
        return self._pipeline_runner.config

    @property
    def mode(self) -> str:
        """Get the current execution mode of the underlying runner. The mode is supposed to be set automatically according to the function used."""
        return self._pipeline_runner.mode

    @property
    def node_objects(self) -> dict[str, Node]:
        """Get the mapping of node names to their instantiated objects in the underlying pipeline."""
        return self._pipeline_runner.node_objects

    def get_params(self) -> dict[str, dict[str, any]]:
        """Get the parameters of all nodes in the underlying pipeline."""
        return self._pipeline_runner.get_params()

    def set_params(self, params: dict[str, dict[str, any]]) -> None:
        """Set the parameters of all nodes in the underlying pipeline."""
        self._pipeline_runner.set_params(params=params)

    # --- API to implement for any PipelineRunnerWrapper implementation ---

    def train(self) -> None:
        """Train the underlying pipeline runner. This method should be implemented by any PipelineRunnerWrapper implementation."""
        return self._pipeline_runner.train()

    def infer(self) -> D_O:
        """Infer with the underlying pipeline runner. This method should be implemented by any PipelineRunnerWrapper implementation."""
        return self._pipeline_runner.infer()

    def evaluate(self) -> M:
        """Evaluate with the underlying pipeline runner. This method should be implemented by any PipelineRunnerWrapper implementation."""
        return self._pipeline_runner.evaluate()