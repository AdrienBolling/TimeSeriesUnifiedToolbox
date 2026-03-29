"""MLFlow Runner Wrapper for the PipelineRunner. This wrapper will log the parameters and metrics of the pipeline to MLFlow."""

from tsut.core.pipeline.wrappers.wrapper import PipelineRunnerWrapper, PipelineRunnerWrapperConfig
import mlflow

class MLFlowLoggerWrapperConfig(PipelineRunnerWrapperConfig):
    """Define the configuration schema for the MLFlowLoggerWrapper."""
    # TODO : Add MLFlow tracking URI, experiment name, run name, etc. to the config.

class MLFlowLoggerWrapper(PipelineRunnerWrapper, ABC):
    # TODO : to discuss depending on how you want to use MLFlow precisely (model registering or just metric logs ?)
    """Define the interface for the MLFlowLoggerWrapper."""
    pass