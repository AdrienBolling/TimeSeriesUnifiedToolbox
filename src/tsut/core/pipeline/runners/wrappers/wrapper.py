"""Base wrapper class for PipelineRunners."""

from abc import ABC, abstractmethod

from tsut.core.pipeline.runners.pipeline_runner import PipelineRunner
from pydantic import BaseModel
from tsut.core.nodes.node import Node

from typing import Any

class PipelineRunnerWrapper[D_O, M, C](PipelineRunner[D_O, M], ABC):
    """Abstract wrapper that adds functionality around a :class:`PipelineRunner`.

    Subclasses can layer cross-cutting concerns (tuning, logging, caching,
    etc.) on top of any runner without modifying its implementation.

    Type Parameters:
        D_O: Data output type of the wrapped runner.
        M: Metrics output type of the wrapped runner.
        C: Configuration model for the wrapper itself.
    """

    def __init__(
        self,
        pipeline_runner: PipelineRunner[D_O, M],
        *,
        config: C,
    ) -> None:
        """Initialize the PipelineRunnerWrapper.

        Args:
            pipeline_runner: The runner instance to wrap.
            config: Wrapper-specific configuration.

        """
        self._pipeline_runner = pipeline_runner
        self._config = config

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
    def config(self) -> C:
        """Get the configuration of this wrapper."""
        return self._config

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