"""MLFlow Runner Wrapper for the PipelineRunner. This wrapper will log the parameters and metrics of the pipeline to MLFlow."""

from abc import ABC

from tsut.core.pipeline.wrappers.wrapper import PipelineRunnerWrapper, PipelineRunnerWrapperConfig

class MLFlowLoggerWrapperConfig(PipelineRunnerWrapperConfig):
    """Define the configuration schema for the MLFlowLoggerWrapper."""
    # TODO : Add MLFlow tracking URI, experiment name, run name, etc. to the config.

class MLFlowLoggerWrapper(PipelineRunnerWrapper, ABC):
    """Interface for pipeline-runner wrappers that log run artifacts to MLflow.

    Implementations can decide whether to only log metrics/parameters or to also
    handle model registration lifecycle concerns.
    """
    pass
