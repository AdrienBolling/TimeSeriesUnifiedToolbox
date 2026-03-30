"""Wrapper for Ray Tune capabilities."""

import tempfile
from typing import Any, Callable

from pydantic_settings import BaseSettings
from ray import tune

from build.lib.tsut.core.pipeline.runners.pipeline_runner import PipelineRunner
from tsut.core.pipeline.runners.wrappers.wrapper import PipelineRunnerWrapper


class TunerWrapperConfig(BaseSettings):
    """Settings for the TuneWrapper."""

class TuneWrapper(PipelineRunnerWrapper):
    """Wrapper for Ray Tune capabilities."""

    def __init__(self, pipeline_runner: PipelineRunner, config: TunerWrapperConfig):
        """Initialize the TuneWrapper with a pipeline runner and configuration."""
        super().__init__(pipeline_runner=pipeline_runner, config=config)
        self._result: tune.ResultGrid | None = None

    @property
    def result(self) -> tune.ResultGrid:
        """Get the result of the Ray Tune run."""
        if self._result is None:
            raise ValueError("No result available. Please run the tune method first.")
        return self._result

    def _trainable(self, optimization_metric: str | None = None, metric_aggregator: Callable | None = None) -> PipelineRunner:
        """Get the function to be used as the trainable for Ray Tune."""
        if optimization_metric is not None and metric_aggregator is None:
            def _meta_metric(metrics: dict) -> float:
                """Aggregate the metrics into a single value for optimization."""
                # This is a placeholder implementation. The actual implementation would depend on how the metrics are structured and how the optimization metric is defined.
                return metrics[optimization_metric]
        elif metric_aggregator is not None and optimization_metric is None:
            def _meta_metric(metrics: dict) -> float:
                """Aggregate the metrics into a single value for optimization."""
                # This is a placeholder implementation. The actual implementation would depend on how the metrics are structured and how the optimization metric is defined.
                return metric_aggregator(metrics)
        else:
            raise ValueError("Either optimization_metric or metric_aggregator must be provided, but not both or neither.")
        def trainable(config):
            # Verify that if the data source comes from a file, the path is absolute, since Ray Tune can change the working directory when running trials.
            self._verify_data_source_paths(config)
            # Set the parameters of the underlying pipeline runner according to the config provided by Ray Tune.
            self._set_pipeline_hyperparameters(config)

            if tune.get_checkpoint():
                self._load_checkpoint(tune.get_checkpoint()) # This is a placeholder implementation. The actual implementation would depend on how checkpoints are defined and loaded in the pipeline runner.

            self.train()
            metrics = self.evaluate()
            # Convert the metrics to a format that can be returned to Ray Tune. (e.g. convert any Data/DataContext objects to dictionaries or other serializable formats)
            metrics = self._convert_metrics(metrics)
            metric = _meta_metric(metrics)
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint = self._save_checkpoint(tmpdir) # This is a placeholder implementation. The actual implementation would depend on how checkpoints are defined and saved in the pipeline runner.
                tune.report(metrics=metrics, checkpoint=checkpoint, opt_metric=metric)
        return trainable

    def _verify_data_source_paths(self, config: dict) -> None:
        """Verify that if the data source comes from a file, the path is absolute, since Ray Tune can change the working directory when running trials."""
        # This is a placeholder implementation. The actual implementation would depend on how the data sources are defined in the pipeline and how they are represented in the config.

    def _set_pipeline_hyperparameters(self, config: dict) -> None:
        """Set the parameters of the underlying pipeline runner according to the config provided by Ray Tune."""
        # This is a placeholder implementation. The actual implementation would depend on how the hyperparameters are defined in the pipeline and how they are represented in the config.

    def _convert_metrics(self, metrics: dict) -> dict:
        """Convert the metrics to a format that can be returned to Ray Tune. (e.g. convert any Data/DataContext objects to dictionaries or other serializable formats)"""
        # This is a placeholder implementation. The actual implementation would depend on the format of the metrics returned by the evaluate method and the format expected by Ray Tune.
        return metrics

    def tune(self, param_space: dict[str, Any], tune_config:tune.TuneConfig, run_config:tune.RunConfig, resources=dict[str, Any]) -> None:
        """Run Ray Tune with the given parameter space and configuration.

        The metric to optimize for Ray Tune is expected to be returned by the trainable function as "opt_metric" in the "metrics" dictionary, and the trainable function is expected to be defined such that it trains the underlying pipeline runner with the hyperparameters provided by Ray Tune, evaluates it, and returns the evaluation metrics in a format that can be processed by the provided metric_aggregator or optimization_metric.
        """
        tune_config.metric = "opt_metric" # Enforce the use of the "opt_metric" as the optimization metric for Ray Tune, since this is what our trainable function returns.
        self._result = tune.run(
            self._trainable(),
            config=param_space,
            tune_config=tune_config,
            run_config=run_config
        )

        # Get the best trial and the configuration that led to the best result, and set the underlying pipeline runner's parameters accordingly.
        best_trial = self._result.get_best_trial()

        # Get the best hyperparameters and set the underlying pipeline runner's parameters accordingly.
        best_config = best_trial.config
        self._set_pipeline_hyperparameters(best_config)

        # Retrain the underlying pipeline runner with the best hyperparameters.
        self.train()
